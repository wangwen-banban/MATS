import os
import json
import time
import datetime
from pathlib import Path
import logging
import shutil
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from tools.dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from tools.utils import load_json, save_json, get_dataloader, prepare_sample, unwrap_dist_model
from tools.optims import get_optimizer, LinearWarmupCosineLRScheduler
from tools.logger import MetricLogger, SmoothedValue

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config = self.cfg.config
        #self.anno_path = self.config.datasets.ann_path
        #log
        self.output_dir = self.config.run.output_dir
        self.log_writter = SummaryWriter(self.output_dir)
        
        # settings
        #self.device = torch.device(self.config.run.device)
        self.local_rank = get_rank()
        self.device = torch.device('cuda:' + str(self.local_rank))
        self.audiofree = self.config.datasets.get('audiofree', True)
        logging.info("current mode is {}".format('audio_free' if self.audiofree else 'audio_input'))
        
        self.use_distributed = self.config.run.use_distributed
        # self.start_epoch = 0
        self.start_epoch = self.config.run.get('start_epoch', 0)
        logging.info("start epoch is {}".format(self.start_epoch))
        self.save_epoch = self.config.run.save_epoch
        self.max_epoch = self.config.run.optims.max_epoch
        self.evaluate_only = self.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")
        
        # scaler
        self.use_amp = self.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def DDP_model(self, model):
        self._model = model
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self._model, device_ids=[self.local_rank], find_unused_parameters=True,
                output_device=[self.local_rank]
            )
        else:
            self.model = self._model
    
    def load_anno_captions(self, out_captions_pred):
        anno_path = self.anno_path
        anno_data = load_json(anno_path)
        audio_to_caption = defaultdict(list)
        for item in anno_data:
            audio_path, caption = item['audio_name'], item['caption']
            audio_to_caption[audio_path].append(caption)
        
        out_captions_gt = []
        for item in out_captions_pred:
            audio_path = item['file_name']
            captions = audio_to_caption[audio_path]
            out_captions_gt.append({
                'file_name': audio_path,
                'caption_reference_01': captions[0],
                'caption_reference_02': captions[1],
                'caption_reference_03': captions[2],
                'caption_reference_04': captions[3],
                'caption_reference_05': captions[4]
            })
        print('ground truth captions for {} audios'.format(len(out_captions_gt)))
        return out_captions_gt
    
    def save_results(self, result, result_dir, filename):
        result_file = os.path.join(result_dir, '{}_rank{}.json'.format(filename, get_rank()))
        final_result_file = os.path.join(result_dir, '{}.json'.format(filename))
        
        save_json(result, result_file)
        
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        if is_main_process():
            logging.info("rank {} starts merging results.".format(get_rank()))
            result = []
            
            for rank in range(get_world_size()):
                result_file = os.path.join(result_dir, '{}_rank{}.json'.format(filename, rank))
                res = load_json(result_file)
                result += res
            
            save_json(result, final_result_file)
            logging.info(f"result file saved to {final_result_file}")
        
    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.cfg.to_dict(), indent=4) + "\n")
    
    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass
    
    def load_checkpoint(self, ckpt_path):
        if ckpt_path:
            logging.info("Load CLAPMLLP ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt['model'], strict=False)
    
    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False, is_last=False):
        model_no_ddp = unwrap_dist_model(self.model, use_distributed=self.use_distributed)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        if is_last:
            save_to = os.path.join(
                self.output_dir,
                "checkpoint_last.pth",
            )
        else:
            save_to = os.path.join(
                self.output_dir,
                "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
            )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
        
        # if not is_best :
        #     save_last = os.path.join(self.output_dir, "checkpoint_last.pth")
        #     shutil.copy(save_to, save_last)