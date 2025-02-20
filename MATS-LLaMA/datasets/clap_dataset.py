import random
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import librosa
import soundfile as sf
import numpy as np
import logging
import torchaudio
from transformers import WhisperFeatureExtractor

from tools.utils import load_json

task_transpose = {
    'caption': ['audiocaption', 'caption'],
    'cla_label': ['cla_label', 'instrument_label', 'genres_label', 'vocal_label', 'scene_label', 'emotion event'],
    'QA': ['audio_qa', 'open-ended question'],
    'music_description': ['musiccaption', 'captioning'], 
}


def read_and_resample_audio(index, root, annotations, resampling_rate, resample=True):
    ann = annotations[index]
    # AIRBench数据集的audio的路径叫做path
    if 'audio_id' not in ann.keys():
        # logging.info('enter the AIRBench, the path is {}'.format(os.path.join(root, "AIRBench","Chat", ann['task_name'] + "_" + ann['dataset_name'], ann['path'])))
        audio, sr = torchaudio.load(os.path.join(root, "AIRBench","Chat", ann['task_name'] + "_" + ann['dataset_name'], ann['path']))
    else :
        # logging.info('no enter, the ann is {}'.format(ann.keys()))
        audio, sr = torchaudio.load(os.path.join(root, ann['audio_id'])) # [1, T * sample_rate]
    
    if audio.numel() != 0 and audio.size(1) != 0:
        pass
    else:
        logging.error("empty file...{}".format(ann['audio_id'] if 'audio_id' in ann.keys() else ann['path']))
        
    
    if sr != resampling_rate and resample:
        resampler = torchaudio.transforms.Resample(sr, resampling_rate)
        audio = resampler(audio) #[1, T * resample_rate]
        sr = resampling_rate
    return audio, sr


def load_audio_into_tensor(index, root, annotations, resampling_rate, audio_duration, resample=True):
    audio, sr = read_and_resample_audio(index, root, annotations, resampling_rate, resample)
    audio_time_series = audio.reshape(-1) #[T * sr]
    
    target_length = audio_duration * sr
    
    # audio_time_series is shorter than prefined audio duration
    # so audio_time_serires is extended
    if target_length >= audio_time_series.size(0):
        repeat_factor = int(np.ceil(target_length / audio_time_series.size(0)))
        audio_time_series = audio_time_series.repeat(repeat_factor)
        audio_time_series = audio_time_series[0:target_length]
    else:
        start_index = random.randrange(audio_time_series.size(0) - target_length)
        audio_time_series = audio_time_series[start_index: start_index + target_length]
    return torch.FloatTensor(audio_time_series), sr      

AIRBENCH_TASK_NAMES = ["sound_QA", 
                       "sound_generation_QA", 
                       "music_QA", 
                       "music_generation_analysis_QA",
                    #    "speech_dialogue_QA",
                    #    "speech_QA",
                    #    "speech_and_music_QA",
                    #    "speech_and_sound_QA"
                       ]

class CLAPDataset(Dataset):
    def __init__(self, ann_path, resampling_rate, audio_duration, audiofree, root, special_token, resample=True, sample_ratio=1.0, isAIRBench=False):
        super().__init__()
        self.resampling_rate = resampling_rate
        self.audio_duration = audio_duration
        self.resample = resample
        self.audiofree = audiofree
        self.root = root
        self.special_token = special_token
        self.annotations = load_json(ann_path, AIRBENCH_TASK_NAMES=None if not isAIRBench else AIRBENCH_TASK_NAMES, root=root)
        self.sample_ratio = sample_ratio

        self.isAIRBench = isAIRBench
        if self.isAIRBench:
            logging.info('Testing for AIRBench...')

        logging.info('{} have {:.2f}K samples'.format(ann_path, len(self.annotations) * 1.0 / 1000))
        
        task2Salmonntask = {}
        for key in task_transpose.keys():
            for value in task_transpose[key]:
                task2Salmonntask[value] = key
        self.task2Salmonntask = task2Salmonntask
    
    def __len__(self):
        return len(self.annotations)
    
    def collater(self, samples):  
        output = [s['output'] for s in samples]
        task = [s['task'] for s in samples]
        Q = [s['Q'] for s in samples]
        special_token = [s['special_token'] for s in samples]

        if self.isAIRBench:
            meta_info = [s['meta_info'] for s in samples]
            task_name = [s['task_name'] for s in samples]
            dataset_name = [s['dataset_name'] for s in samples]
            uniq_id = [s['uniq_id'] for s in samples]
        
        if not self.audiofree:     
            id = [s['id'] for s in samples] 
            raw_wav = [torch.tensor(s['raw_wav']) for s in samples]
            raw_wav_length = torch.tensor([len(s['raw_wav']) for s in samples])
            raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0) #[B, L2]
            paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1) #[B, L2]

            if self.isAIRBench:
                return {
                    # for AIRBench
                    "meta_info": meta_info,
                    "task_name": task_name,
                    'dataset_name': dataset_name,
                    'uniq_id': uniq_id,
                    
                    "id": id,
                    "raw_wav": raw_wav,
                    "padding_mask": paddding_mask,
                    "output": output,
                    "task": task,
                    "Q": Q,
                    "special_token": special_token
                }
            return {
                "id": id,
                "raw_wav": raw_wav,
                "padding_mask": paddding_mask,
                "output": output,
                "task": task,
                "Q": Q,
                "special_token": special_token
            }
        else:
            caption = [s['caption'] for s in samples]
            return {
                "caption": caption,
                "output": output,
                "task": task,
                "Q": Q,
                "special_token": special_token
            }
    
    def __getitem__(self, index):
        ann = self.annotations[index]
        # task = ann['task']
        # task = self.task2Salmonntask[task]

        if self.isAIRBench:
            task = 'audio_qa'
        else:
            task = ann['task']

        task = self.task2Salmonntask[task]
        
        if not self.audiofree:
            audio, sr = load_audio_into_tensor(index, self.root, self.annotations, 
                                            self.resampling_rate, self.audio_duration, 
                                            self.resample) 
            audio = audio.reshape(-1) #[T * sr]

            if self.isAIRBench:
                return {
                    # AIRBench needs
                    "meta_info": ann['meta_info'],
                    "task_name": ann['task_name'],
                    "dataset_name": ann['dataset_name'],
                    "uniq_id": ann['uniq_id'],

                    # origin needs
                    'id': ann['audio_id'] if not self.isAIRBench else ann['path'],
                    'raw_wav': audio,
                    'output': ann['answer_gt'],
                    'task': task,
                    'Q': ann.get('question', ''),
                    'special_token': self.special_token,
                }
            
            return  {
                'id': ann['audio_id'],
                'raw_wav': audio,
                'output': ann['output'],
                'task': task,
                'Q': ann.get('instruction', ''),
                'special_token': self.special_token,
            }
        else:
            caption = ann['caption']
                
            return  {
                'caption': caption,
                'output': ann['output'],
                'task': task,
                'Q': ann.get('instruction', ''),
                'special_token': self.special_token,
            }
            
