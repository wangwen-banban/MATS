import argparse
import random
import pdb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from pathlib import Path

from tools.utils import *
from tools.config import Config
from tools.dist_utils import get_rank, init_distributed_mode, setup_seeds, is_dist_avail_and_initialized
from algorithm.MATS import MATS

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, default='')
    parser.add_argument("--cfg-path1", type=str, default='')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    return parser.parse_args()

def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    #job_id = now()
    
    # load config
    cfg = Config(parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.config.gpu)
    
    #cfg.config.run.output_dir = os.path.join(cfg.config.run.output_dir, job_id)
    if cfg.config.model.use_laion:
        cfg.config.run.output_dir = cfg.config.run.laion_output_dir
    os.makedirs(cfg.config.run.output_dir, exist_ok=True)
    cfg.config.run.log_file = os.path.join(cfg.config.run.output_dir, 'log_file.txt')
    
    # initialize distributed training
    init_distributed_mode(cfg.config.run)
    setup_seeds(cfg.config.run.seed)
    setup_logger(cfg.config.run.log_file) # set after init_distributed_mode() to only log on master.
    
    # print config
    cfg.pretty_print()

    # build runner
    runner = MATS(cfg)
    
    # train
    runner.train()


if __name__ == "__main__":
    main()