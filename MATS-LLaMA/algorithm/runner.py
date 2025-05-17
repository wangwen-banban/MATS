import logging

import torch

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config = self.cfg.config
        
        # settings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.audiofree = self.config.datasets.get('audiofree', True)
        logging.info("current mode is {}".format('audio_free' if self.audiofree else 'audio_input'))
        
        self.use_distributed = False
        self.evaluate_only = self.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")
        
        # scaler
        self.use_amp = self.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def load_checkpoint(self, ckpt_path):
        if ckpt_path:
            logging.info("Load CLAPMLLP ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt['model'], strict=False)