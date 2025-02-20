import logging
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList

from msclap import CLAP
from .base_model_ import BaseModel

from .utils import count_parameters, noise_injection, StoppingCriteriaSub, load_memory


class AudioFreeMLLM(BaseModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.amp = False
        self.noise_variance = config.get("noise_variance", 0.015)
        self.uniform_noise = config.get("uniform_noise", False)
        logging.info('noise variance: {}'.format(self.noise_variance))
        
        # prepare prompts
        self.prompt_dict = self.prepare_prompts() # contains the <SpeechHere>
        
        # loading llama
        self.llama_model, self.llama_tokenizer = self.load_llama()

        if self.use_memory_bank:
            # k-means
            if self.is_kmeans:
                self.memory_bank_kmeans, self.means = self.kmeans_memory_bank(self.group_nums)
            else:
                self.memory_bank = self.load_memory_bank()
                self.memory_bank = self.memory_bank.unsqueeze(0)    # 1, num_mb, dim
        
        # loading clap text encoder
        self.clap_wrapper, self.audio_encoder, self.clap_text_encoder = self.load_clap()

        self.ln_audio = nn.LayerNorm(self.clap_dim) 
        
        # loading Mapper
        qdim = self.llama_model.config.hidden_size     
        self.audio_mapper, self.audio_llama_proj = self.load_mapper(qdim) # llama_proj is a Linear
    
    def audio_llm_connector(self, audio_embeds):
        with self.maybe_autocast(): 
            audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)
            audio_embeds = self.audio_mapper(audio_embeds) #[B, prefix_length, C]
            audio_embeds = audio_embeds.view(audio_embeds.size(0), self.prefix_length, -1)
            audio_embeds = self.audio_llama_proj(audio_embeds)
            
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device) #[B, prefix_length]
        
        return audio_embeds, audio_atts
    
    def encode_text(self, captions):
        # captions: a list of captions
        with self.maybe_autocast():
            text_embeds = self.clap_wrapper.get_text_embeddings(captions) #[B, D] # if use laion is 512
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            text_embeds = noise_injection(text_embeds, self.noise_variance, self.device, self.uniform_noise)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            audio_embeds, audio_atts = self.audio_llm_connector(text_embeds)
        
        return audio_embeds, audio_atts

    def encode_audio(self, raw_wav, is_longer=None):
        # raw_wav: [B, sample_rate * audio_duration]
        with self.maybe_autocast():
            
            audio_embeds, _ = self.audio_encoder(raw_wav) #[B, C]

            if self.use_memory_bank:
                if self.is_kmeans:
                    
                    cluster_distances = torch.cdist(audio_embeds, self.means.to(audio_embeds.device), p=2)  # (B, num_clusters)
                    closest_clusters = torch.argmin(cluster_distances, dim=1)  # (B,)
                    top_k_samples = []

                    for idx, cluster_idx in enumerate(closest_clusters):
                        cluster_samples = torch.stack(self.memory_bank_kmeans[cluster_idx.item()]).to(audio_embeds.device)  # (num_samples_in_cluster, D)

                        if cluster_samples.shape[0] < self.k_nums:
                            padding_count = self.k_nums - cluster_samples.shape[0]
                            padding_samples = cluster_samples[:1].repeat(padding_count, 1)  # 取第一个样本重复填充
                            cluster_samples = torch.cat([cluster_samples, padding_samples], dim=0)  # 合并填充样本
                        
                        sample_distances = torch.cdist(audio_embeds[idx].unsqueeze(0), cluster_samples, p=2).squeeze(0)  # (num_samples_in_cluster,)
                        
                        top_k_indices = torch.topk(sample_distances, k=self.k_nums, largest=False).indices  # (k,)
                        top_k_samples.append(cluster_samples[top_k_indices])  # (k, D)

                    top_k_samples = torch.stack(top_k_samples)  # (B, k, D)                    
                else:
                    temp_audio_embeds = audio_embeds.clone()
                    temp_audio_embeds = temp_audio_embeds.to(self.memory_bank.device)
                    temp_audio_embeds = temp_audio_embeds.unsqueeze(1)  # bs, 1, dim
                    
                    dist = torch.norm(temp_audio_embeds - self.memory_bank, dim=2)  # bs, num_mb

                    topk_dist, topk_indices = torch.topk(dist, self.k_nums, dim=1, largest=False, sorted=True)

                    top_k_samples = self.memory_bank.squeeze(0)[topk_indices]
                    top_k_samples = top_k_samples.to(audio_embeds.device)

                top_k_samples = noise_injection(top_k_samples, self.noise_variance, self.device, self.uniform_noise)

                # (bs, 1024) * (bs, k, 1024) -> (bs, k)
                dot_products = torch.einsum('nd,nkd->nk', audio_embeds, top_k_samples)  # (n, k)

                # (bs, k)
                weights = torch.softmax(dot_products * self.mb_temperature, dim=-1)  
            
                weighted_text_embedds = weights.unsqueeze(-1) * top_k_samples

                text_embedds = weighted_text_embedds.sum(dim=1)  # (n, d)

                audio_embeds = audio_embeds * (1 - self.memory_lambda) + text_embedds * self.memory_lambda
                
            audio_embeds, audio_atts = self.audio_llm_connector(audio_embeds)
                 
        return audio_embeds, audio_atts
    
    def forward(self, samples, mode='train', verbose=True):
        # prepare prompts
        special_tokens = samples['special_token'] # bs
        prompt = []
        for i, task in enumerate(samples['task']):
            x = random.choice(self.prompt_dict[task])
            x = re.sub(r'(<Speech><SpeechHere></Speech>)(.*)', rf'\1 {special_tokens[i]}\2', x) # 在speech之后加上special token
            prompt.append(x)
            
        if 'Q' in samples:
            prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples['Q'])] # for the qa pairs, replace the {} with the question
        
        # use text/audio encoder to encode text/audio
        if mode == 'train':
            captions = samples["caption"]
            speech_embeds, speech_atts = self.encode_text(captions) #[B, 40, D]
        else:
            raw_wav = samples["raw_wav"] #[B, T * sr]
            
            speech_embeds, speech_atts = self.encode_audio(raw_wav)
            
        # wrap embeds with prompts
        if self.prompt_dict and prompt:
            # emebedding the prompt
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt)

        # prepare inputs/targets for LLMs, output + </s>
        text = [t + self.end_sym for t in samples['output']]
        # the length more than the max_len, then will be trucated
        to_regress_tokens = self.llama_tokenizer(text, return_tensors='pt', padding='longest',
                                                 truncation=True, max_length=self.max_txt_len, 
                                                 add_special_tokens=False).to(self.device) #[B, L2]
        
        to_regress_embeds = self.llama_embed_tokens(to_regress_tokens.input_ids)
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        ) #[B, L2]
        empty_targets = torch.ones(
            [speech_atts.shape[0], speech_atts.shape[1] + 1],  # bs, seq_len+1
            dtype=torch.long
        ).to(self.device).fill_(-100) #[B, L1+1]
        
        targets = torch.cat([empty_targets, targets], dim=1) #[B, 1+L1+L2]
        
        # inputs for LLM
        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], 
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device
        ) * self.llama_tokenizer.bos_token_id #[B, 1]
        
        bos_embeds = self.llama_embed_tokens(bos) # emebedding
        atts_bos = speech_atts[:, :1]
        
        # bos+caption+(output+eos), the caption's attention_mask is all one 
        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1) #[B, 1+L1+L2, C]
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)
        # calculate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
        if torch.isnan(loss):
            import pdb; pdb.set_trace()
        # calculate accuracy
        if verbose:
            logits = outputs.logits[:, empty_targets.size(1) -  1: -1] #[B, L2, V], output
            labels = targets[:, empty_targets.size(1):] #[B, L2]
            acc, total = self.compute_accuracy(logits, labels)
            return {'loss': loss, 'acc': acc, 'total': total}

        return {'loss': loss}

    def generate(self, samples, generate_cfg, prompt):
        if self.use_laion:
            batch_size = len(samples["raw_wav"])
        else:
            batch_size = samples["raw_wav"].shape[0]
        raw_wav = samples["raw_wav"]
            
        if self.use_laion:
            is_longer = samples['is_longer']
            speech_embeds, speech_atts = self.encode_audio(raw_wav, is_longer)
        else:
            speech_embeds, speech_atts = self.encode_audio(raw_wav)

        if prompt is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        
        bos_embeds = self.llama_embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)  
        
        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 70),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_k=generate_cfg.get("top_k", 50),
            no_repeat_ngram_size=generate_cfg.get("no_repeat_ngram_size", 2),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 2.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        text = [t.capitalize().strip() for t in text]
        return text
