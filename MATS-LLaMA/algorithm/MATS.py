import unidecode
import logging
import re

import torch
import numpy as np
import random
import torchaudio

from .runner import Runner
from models import load_model
from torch.nn.utils.rnn import pad_sequence



all_tasks = {
    "caption": "<Speech><SpeechHere></Speech> Describe the following audio in a caption.",
    "cla_label": "<Speech><SpeechHere></Speech> Identify the sounds in the audio clip, only display audio tags.",
    "musis_caption": "<Speech><SpeechHere></Speech> Listen to this music clip and describe the music.",
    "QA": "<Speech><SpeechHere></Speech> {}"
}

def read_and_resample_audio(index, root, annotations, resampling_rate, resample=True):
    ann = annotations[index]
    audio, sr = torchaudio.load(ann['audio_id']) # [1, T * sample_rate]    
    
    if sr != resampling_rate and resample:
        resampler = torchaudio.transforms.Resample(sr, resampling_rate)
        audio = resampler(audio) #[1, T * resample_rate]
        sr = resampling_rate
    return audio, sr


def load_audio_into_tensor(index, root, annotations, resampling_rate, audio_duration, resample=True):
    audio, sr = read_and_resample_audio(index, root, annotations, resampling_rate, resample)
    audio_time_series = audio.reshape(-1) #[T * sr]
    
    target_length = audio_duration * sr
    
    if target_length >= audio_time_series.size(0):
        repeat_factor = int(np.ceil(target_length / audio_time_series.size(0)))
        audio_time_series = audio_time_series.repeat(repeat_factor)
        audio_time_series = audio_time_series[0:target_length]
    else:
        start_index = random.randrange(audio_time_series.size(0) - target_length)
        audio_time_series = audio_time_series[start_index: start_index + target_length]
    return torch.FloatTensor(audio_time_series), sr    

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples

def unwrap_dist_model(model, use_distributed):
    if use_distributed:
        return model.module
    else:
        return model

def post_process(caption):
    caption = unidecode.unidecode(caption)
    caption = caption.replace(',', ' , ') 
    caption = re.sub(' +', ' ', caption)
    caption = caption.replace(' ,', ',')
    caption = caption.strip()
    return caption

class MATS(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        model_config = cfg.config.model
        model_config.evaluation = True
        
        # build model
        self.model = load_model(model_config)
        
        # load ckpt
        if model_config.get('ckpt', "") != "":
            ckpt_path = model_config.ckpt
            logging.info("Load MATS ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt['model'], strict=False)
    
    @torch.no_grad()
    def generate(self, audio_fp, task, question=None):
        audios = [{"audio_id":audio_fp}]
        if task not in all_tasks.keys():
            raise ValueError("Task {} not in {}".format(task, all_tasks.keys()))
        prompt = all_tasks[task]
        if task == "QA":
            if question is None:
                raise ValueError("Question is None, but task is QA")
            prompt = prompt.format(question)
        prompt = "USER: {}\nASSISTANT:".format(prompt)
        prompts = [prompt]
        samples = {"raw_wav":[]}
        # process the audio
        audio, sr = load_audio_into_tensor(0, audios, self.config.datasets.resampling_rate, self.config.datasets.audio_duration, self.config.datasets.resample)
        audio = audio.reshape(-1)
        audio = [torch.tensor(audio)]
        raw_wav = pad_sequence(audio, batch_first=True, padding_value=0)

        samples = {"raw_wav":raw_wav}

        self.model.eval()

        samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled) # to cuda

        text = self.model.generate(samples, self.config.generate, prompts)
        response = post_process(text[0])

        return response