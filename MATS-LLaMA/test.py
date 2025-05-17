import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from tools.config import Config
from algorithm.MATS import MATS
import os

def setup_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

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
    
    setup_seeds(cfg.config.run.seed)
    
    # print config
    cfg.pretty_print()
    
    # build runner
    runner = MATS(cfg)
    
    response = runner.generate(
        audio_fp=cfg.config.datasets.test_file,
        task=cfg.config.datasets.task,
        question=cfg.config.datasets.question
    )

    print("Response: ", response)

if __name__ == "__main__":
    main()
