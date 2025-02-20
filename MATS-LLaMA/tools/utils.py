import json
import logging
import time
import os
import random
import sys
import os.path as osp
import unidecode
import re

import torch
import torchaudio
from torch.utils.data import DataLoader, DistributedSampler
import soundfile as sf
import numpy as np

from .dist_utils import is_main_process, get_world_size, get_rank


def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M")


def setup_logger(fpath):
    if fpath is not None:
        mkdir_if_missing(os.path.dirname(fpath))
        
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(fpath)
        ],
    )


def load_json(path, AIRBENCH_TASK_NAMES=None, root=None):
    try:
        if AIRBENCH_TASK_NAMES is None:
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = []
            with open(path, 'r') as f:
                samples = json.load(f)
                for sample in samples:
                    if sample['task_name'] in AIRBENCH_TASK_NAMES:
                        audio, sr = torchaudio.load(os.path.join(root, "AIRBench","Chat", sample['task_name'] + "_" + sample['dataset_name'], sample['path']))
                        if audio.numel() != 0 and audio.size(1) != 0:
                            data.append(sample)
            data = json.load(f)
    except:
        print(f"Failed to load {path}! Try to use utf-8 encoding.")
        if AIRBENCH_TASK_NAMES is None:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                for sample in samples:
                    if sample['task_name'] in AIRBENCH_TASK_NAMES:
                        audio, sr = torchaudio.load(os.path.join(root, "AIRBench","Chat", sample['task_name'] + "_" + sample['dataset_name'], sample['path']))
                        if audio.numel() != 0 and audio.size(1) != 0:
                            data.append(sample)
    return data


def save_json(data, path):
    try:
        json.dump(data, open(path, 'w'), ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Error saving {path}. Error: {e}")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=is_train,
            num_replicas=get_world_size(),
            rank=get_rank()
        )
    else:
        sampler = None
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=False,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        collate_fn=dataset.collater,
        drop_last=is_train,
    )
    
    if is_train:
        loader = IterLoader(loader, use_distributed=use_distributed)

    return loader

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

    # TODO fp16 support

    return samples

def unwrap_dist_model(model, use_distributed):
    if use_distributed:
        return model.module
    else:
        return model

class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)
    

class MultiIterLoader:
    """
    A simple wrapper for iterating over multiple iterators.

    Args:
        loaders (List[Loader]): List of Iterator loaders.
        ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
    """

    def __init__(self, loaders, ratios=None):
        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]
        self.loaders = loaders
        self.ratios = ratios
        
        rounded_ratios = [round(num, 2) for num in self.ratios]
        logging.info('rations: {}'.format(rounded_ratios))
    
    def __next__(self):
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        return next(self.loaders[loader_idx])
    
    def __len__(self):
        length = 0
        for loader in self.loaders:
            length += len(loader)
        return length


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        os.makedirs(directory) 

def get_logger(fpath, local_rank=0, name=''):
    # Creat logger
    logger = logging.getLogger(name)
    level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logger.setLevel(level=level)

    # Output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=level) 
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Output to file
    if fpath is not None:
        mkdir_if_missing(os.path.dirname(fpath))
    file_handler = logging.FileHandler(fpath, mode='w')
    file_handler.setLevel(level=level)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger

class Logger(object):
    """
    Write console output to external text file.
    """
    def __init__(self, fpath=None, mode='a'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def post_process(caption):
    caption = unidecode.unidecode(caption)
    caption = caption.replace(',', ' , ') 
    caption = re.sub(' +', ' ', caption)
    caption = caption.replace(' ,', ',')
    caption = caption.strip()
    return caption