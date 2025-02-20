import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from pathlib import Path

from tools.utils import *
from tools.config import Config
from tools.dist_utils import get_rank, init_distributed_mode, setup_seeds, is_dist_avail_and_initialized
from models import load_model
from algorithm.MATS import MATS

def parse_args():
    parser = argparse.ArgumentParser(description='test parameters')
    parser.add_argument("--cfg-path", type=str, default='configs/audiofree_config_open.yaml')
    parser.add_argument("--cfg-path1", type=str, default='configs/decode_config.yaml')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()

def main():
    # load config
    cfg = Config(parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.config.gpu)
    
    #ckpt_root = cfg.config.model["ckpt"]
    ckpt_root = "results/CyCLAP/stage2_model_noise_0.005_no_mb"
    # ckpt_root = "results_laion/speech_noise_0.015_mapper_1024"
    cfg.config.run.output_dir = ckpt_root
    cfg.config.run.log_file = os.path.join(cfg.config.run.output_dir, 'log_test_file.txt')
   
    # initialize distributed training
    init_distributed_mode(cfg.config.run)
    setup_seeds(cfg.config.run.seed)
    setup_logger(cfg.config.run.log_file) # set after init_distributed_mode() to only log on master.
    
    # print config
    cfg.pretty_print()
    
    # build runner
    runner = MATS(cfg)
    
    # test
    for epoch in range(29, 30):
        ckpt_path = os.path.join(ckpt_root, f'checkpoint_{epoch}.pth')
        
        # ckpt_path = os.path.join(ckpt_root, f'checkpoint_last.pth')
        assert os.path.exists(ckpt_path)
        runner.load_checkpoint(ckpt_path)
        
        runner.valid_generate_epoch(epoch, "test")
    


if __name__ == "__main__":
    main()
