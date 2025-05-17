import logging
import contextlib
import os
from einops import rearrange, repeat
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList, ClapModel, ClapProcessor, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.bert.configuration_bert import BertConfig
from msclap import CLAP

from .mapper import build_mapper
from .modeling_llama import LlamaForCausalLM
from .utils import StoppingCriteriaSub, load_json, count_parameters


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.llama_path = config.get("llama_path")
        self.clap_path = config.get("clap_path")
        self.freeze_clap = config.get("freeze_clap", True)

        self.clap_dim = config.get("clap_dim", 1024)
        self.mapper_dim = config.get("mapper_dim", 1024) 

        # memory bank
        self.use_memory_bank = config.get('use_memory_bank', False)
        self.is_kmeans = config.get("is_kmeans", True)
        self.k_nums = config.get('k_nums', 16)
        self.memory_lambda = config.get('lambda', 0.5)
        self.group_nums = config.get('group_nums', 1000)
        self.save_memory_dir = config.get('save_memory_dir', None)
        self.iter_kmeans = config.get('iter_kmeans', 30)
        self.mb_temperature = config.get('temperature', 1)
        self.mb_number = config.get('number', 5172)
        
        self.prefix_length = config.get("prefix_length", 40) 
        self.clip_length = config.get("clip_length", 40)
        self.num_layers = config.get("num_layers", 8)
        self.mapping_type = config.get("mapping_type")
        
        self.lora = config.get("lora", True)
        self.lora_rank = config.get("lora_rank", 8)
        self.lora_alpha = config.get("lora_alpha", 32)
        self.lora_dropout = config.get("lora_dropout", 0.1)

        self.max_txt_len = config.get("max_txt_len", 128)
        self.end_sym = config.get("end_sym", "</s>")
        self.amp = False
    
    def load_llama(self):
        logging.info('loading LLAMA Tokenizer')
        
        llama_tokenizer = LlamaTokenizer.from_pretrained(self.llama_path, use_fast=False) # fast is Rust version
        llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # for variable length
        if self.config.get('evaluation'):
            llama_tokenizer.padding_side = 'left'
        else:
            llama_tokenizer.padding_side = 'right'
        logging.info('Padding Strategy is {}'.format(llama_tokenizer.padding_side))
       
        logging.info('Loading LLaMA Model')
        llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            torch_dtype=torch.float16,
        )
        
        # this func means that it can change the vocabulary size, not the real embedding
        llama_model.resize_token_embeddings(len(llama_tokenizer)) # vocab_size + added special token
    
        # fix llama, frozen
        for name, param in llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')
        
        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=self.lora_rank, 
                lora_alpha=self.lora_alpha, 
                lora_dropout=self.lora_dropout,
            )
            llama_model = get_peft_model(llama_model, self.peft_config)
            llama_model.print_trainable_parameters()
            logging.info('LoRA Training')
            
        return llama_model, llama_tokenizer
    
    def load_clap(self):   
        # loading clap model
        logging.info('Loading CLAP Model')
        clap_wrapper = CLAP(model_fp=self.clap_path, use_cuda=True)
        audio_encoder = clap_wrapper.clap.audio_encoder
        clap_text_encoder = clap_wrapper.clap.caption_encoder
    
        if self.freeze_clap:
            for name, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()
            logging.info('freeze CLAP Audio Encoder')
            for name, param in clap_text_encoder.named_parameters():
                param.requires_grad = False
            clap_text_encoder.eval()
            logging.info('freeze CLAP Text Encoder')
            
        count_parameters(audio_encoder, name='CLAP Audio Encoder')
        count_parameters(clap_text_encoder, name='CLAP Text Encoder')
        
        return clap_wrapper, audio_encoder, clap_text_encoder
    
    
    def load_mapper(self, qdim):
        logging.info('Loading Audio Mapper')
        audio_mapper = build_mapper(
            mapping_type=self.mapping_type, 
            prefix_dim=self.clap_dim, 
            llm_embedding_size=self.mapper_dim, 
            prefix_length=self.prefix_length, 
            clip_length=self.clip_length,
            num_layers=self.num_layers
        )
        count_parameters(audio_mapper, name='mapper')
        
        logging.info('Loading Mapper LLAMA proj')
        audio_llama_proj = nn.Linear(self.mapper_dim, qdim) # proj the 1024 to the llama dim
        count_parameters(audio_llama_proj, name='projector')
        
        return audio_mapper, audio_llama_proj
    
    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_memory_bank(self):
        logging.info("load memory bank...")
        assert os.path.exists(self.save_memory_dir), 'the memory bank does not exist!'
        memory_bank = torch.load(self.save_memory_dir)
        if isinstance(memory_bank, list):
            memory_bank = torch.stack(memory_bank)
            memory_bank = memory_bank[:self.mb_number]
            random.shuffle(memory_bank)
        logging.info('there are {:.2}k text embedding!'.format(len(memory_bank) / 1e3))
        logging.info("memory bank done!")
        return memory_bank

    def sample_vector(self, nums_center, all_text_embedding):
        num_samples = all_text_embedding.shape[0]
        
        indices = torch.randperm(num_samples)[:nums_center] # 生成0~num_samples-1的随机排列

        return all_text_embedding[indices]

    def kmeans_memory_bank(self, nums_center: int):
        logging.info('k-means start...')
        logging.info('the number of cluster is {}'.format(nums_center))
        assert os.path.exists(self.save_memory_dir), 'the memory bank does not exist!'
        all_text_embedding = torch.load(self.save_memory_dir)
        if isinstance(all_text_embedding, list):
            all_text_embedding = torch.stack(all_text_embedding)
            random.shuffle(all_text_embedding)
            all_text_embedding = all_text_embedding[:self.mb_number]
        dim = all_text_embedding.shape[-1]
        logging.info('there {} samples in memory bank'.format(len(all_text_embedding)))
        dtype = all_text_embedding.dtype
        
        assert nums_center <= len(all_text_embedding), 'the number of the centers should be less than the number of all text embedding!'
        means = self.sample_vector(nums_center, all_text_embedding)

        clusters_ = [[] for _ in range(nums_center)]
        
        for iter in range(self.iter_kmeans):
            diffs = rearrange(all_text_embedding, "n d -> n () d") - rearrange(
                means, "c d -> () c d"
            )
            dists = -(diffs ** 2).sum(dim=-1)

            buckets = dists.max(dim=-1).indices 
            bins = torch.bincount(buckets, minlength=nums_center)
            zero_mask = bins == 0
            bins_min_clamped = bins.masked_fill(zero_mask, 1)

            new_means = buckets.new_zeros(nums_center, dim, dtype=dtype)
            new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), all_text_embedding)
            new_means = new_means / bins_min_clamped[..., None]

            means = torch.where(zero_mask[..., None], means, new_means)
        
        # final cluster
        for embedding in all_text_embedding:
            distances = torch.norm(means - embedding, dim=1)
            min_index = torch.argmin(distances)
            clusters_[min_index].append(embedding)

        logging.info('k-means done!')
        return clusters_, means
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = (self.device != torch.device("cpu") and self.amp)

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def llama_embed_tokens(self, tokens):
        if not self.lora:
            embeds = self.llama_model.model.embed_tokens(tokens)
        else:
            embeds = self.llama_model.model.model.embed_tokens(tokens) # lora return the pert model
        return embeds
    
    def compute_accuracy(self, logits, labels):
        # logits: [B, L, V]
        # labels: [B, L]
        B = logits.size(0) 
        preds = logits.contiguous().view(-1, logits.size(-1)).argmax(dim=-1) #[B * L2]
        labels = labels.contiguous().view(-1)
        mask = (labels != -100)
        correct = (preds[mask] == labels[mask]).float().sum()
        total = len(labels[mask])
        acc = (correct / total).item()
        return acc, total
        
    # caption attention could search the info after current pos
    def prompt_wrap(self, embeds, atts, prompt):
        # embeds: [B, M, C]
        # atts: [B, M]
        # prompt: [B prompt] 
        p_before = []
        p_after = []
        for i, p in enumerate(prompt):
            b, a = p.split('<SpeechHere>')
            p_before.append(b)
            p_after.append(a)
            
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors='pt', add_special_tokens=False
        ).to(embeds.device)  #[B, l1]
        
        # prompts_embeds are padded to the same length here
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors='pt', padding="longest", add_special_tokens=False
        ).to(embeds.device)  #[B, l2]
        
        p_before_embeds = self.llama_embed_tokens(p_before_tokens.input_ids) #[B, l1, C]
        p_after_embeds = self.llama_embed_tokens(p_after_tokens.input_ids)
        
        wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
        wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
    
        return wrapped_embeds, wrapped_atts  
    
    @classmethod
    def from_config(cls, config):
      
        model = cls(config)

        return model

