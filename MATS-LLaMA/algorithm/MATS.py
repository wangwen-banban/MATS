import os
import json
import time
import datetime
from pathlib import Path
import logging
import re

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import shutil

from .runner import Runner
from models import load_model
from datasets import load_dataset, load_dataloader, load_laion_dataset
from datasets.clap_dataset import load_audio_into_tensor
from torch.nn.utils.rnn import pad_sequence

from tools.dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_world_size
from tools.utils import load_json, get_dataloader, prepare_sample, unwrap_dist_model, post_process
from tools.optims import get_optimizer, LinearWarmupCosineLRScheduler
from tools.logger import MetricLogger, SmoothedValue


class MATS(Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        model_config = cfg.config.model
        data_config = cfg.config.datasets
        self.isAIRBench = cfg.config.datasets.get('isAIRBench', False)
        run_config = cfg.config.run

        model_config.evaluation = run_config.get('evaluate', False)
        self.metrics = {}
        
        # build dataset
        datasets = load_dataset(data_config)
        
        
        self.train_loader, self.valid_loader, self.test_loader = \
            load_dataloader(datasets=datasets, run_config=run_config, use_distributed=self.use_distributed)
        logging.info('A train loader have {} iteractions'.format(len(self.train_loader)))

        if self.config.run.epoch_based:
            self.iters_per_epoch = len(self.train_loader)
        else:
            self.iters_per_epoch = self.config.run.iters_per_epoch * self.config.run.accum_grad_iters
        
        self.config.run.optims.warmup_steps = self.config.run.optims.warmup_steps * self.config.run.accum_grad_iters
        logging.info('A train loader need {:.1f} epochs'.format(len(self.train_loader) * 1.0 / self.iters_per_epoch))
        logging.info('Warmup iteractions: {}'.format(self.config.run.optims.warmup_steps))
        
        # build test prompt
        self.prompt_template = self.config.model.get("prompt_template", "") # "USER: {}\nASSISTANT:"
        test_prompt_dict = self.config.model.get("test_prompt_path", "") 
        self.test_prompt_dict = load_json(test_prompt_dict)
        for k in self.test_prompt_dict.keys():
            self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])
        
        # build model
        model = load_model(model_config)
        
        # load ckpt
        # ckpt_path = 'results/CyCLAP/stage2_model_noise_0.005_cyclap_100_epochs/checkpoint_last.pth'
        # logging.info("Load CLAPMLLP ckpt from: {}".format(ckpt_path))
        # ckpt = torch.load(ckpt_path, map_location="cpu")
        # model.load_state_dict(ckpt['model'], strict=False)
        
        self.DDP_model(model)
        
        # optimizer & scheduler
        self.optimizer = get_optimizer(self.model, self.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.run.optims.min_lr,
            init_lr=self.config.run.optims.init_lr,
            warmup_steps=self.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.run.optims.get("warmup_start_lr", -1),
        )

    def train(self):
        start_time = time.time()
        # the upper bound of the loss
        best_agg_metric = 20
        best_epoch = 0
        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break
            
            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")
            
            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid")
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log['loss']
                    if agg_metrics < best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch
                        self.save_checkpoint(cur_epoch, is_best=True)
                    
                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
            
            # save the model
            if cur_epoch % self.save_epoch == 0 or cur_epoch == self.max_epoch - 1:
                self.save_checkpoint(cur_epoch, is_best=False)
        
            self.save_checkpoint(cur_epoch, is_best=False, is_last=True)        
                
            #同步所有进程 
            if self.use_distributed:
                dist.barrier()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
    
    def train_epoch(self, epoch):
        self.model.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=20, fmt="{avg:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{avg:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=20, fmt="{avg:.1f}"))
        
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch)
        )
        header = "Train: data epoch: [{}]".format(epoch)
        start_step = epoch * self.iters_per_epoch
        
        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.run.log_freq, header=header, logger=self.log_writter, start_step=start_step):
            if i >= self.iters_per_epoch:
                break
            
            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled) # to cuda
            metric_logger.data_time.update(time.time() - metric_logger.end)
            
            self.scheduler.step(cur_epoch=epoch, cur_step=i)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self.audiofree:
                    forward_result = self.model(samples, verbose=True)
                else:
                    forward_result = self.model(samples, mode='val', verbose=True)
            loss = forward_result.get("loss", 0)
            acc = forward_result.get("acc", 0)
            
            if self.use_amp:
                self.scaler.scale(loss).backward() # mix up the float16 and float32
            else:
                loss.backward()
            
            if (i + 1) % self.config.run.accum_grad_iters == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                #self.scheduler.step(cur_epoch=epoch, cur_step=i)
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(acc=acc * 100)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
         
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    
    @torch.no_grad()
    def valid_epoch(self, epoch, split):
        model = unwrap_dist_model(self.model, use_distributed=self.use_distributed)
        model.eval()
        
        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{avg:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=20, fmt="{avg:.1f}"))
        header = "Eval: data epoch: [{}]".format(epoch)
        
        results = []
        # import pdb; pdb.set_trace()
        for samples in metric_logger.log_every(dataloader, self.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled) # to cuda
            metric_logger.data_time.update(time.time() - metric_logger.end)
        
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                forward_result = model(samples, mode='val', verbose=True)
            loss = forward_result.get("loss", 0)
            acc = forward_result.get("acc", 0)
            total = forward_result.get("total", 1)
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(acc=acc * 100)
            
            res = {
                "ground_truth": samples["output"],
                "loss": loss.item(),
                "acc": acc,
                "total": total,
            }
            
            results.append(res)
        
        metric_logger.synchronize_between_processes()
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        res = {
            "loss": torch.tensor(0).float().cuda(),
            "n_sample": torch.tensor(0).float().cuda(),
            "correct": torch.tensor(0).float().cuda(),
            "n_token": torch.tensor(0).float().cuda(),
        }
        for item in results:
            res['loss'] += item['loss'] * len(item['ground_truth'])
            res['n_sample'] += len(item['ground_truth'])
            res['correct'] += item['acc'] * item['total']
            res['n_token'] += item['total']
        
        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])
        
        ret = {"epoch": epoch, "loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item()
        ret["agg_metrics"] = (res["correct"] / res["n_token"]).item()
        return ret
    
    @torch.no_grad()
    def generate(self, audio_fp, prompt):
        audios = [{"audio_id":audio_fp}]
        prompts = [prompt]
        samples = {"raw_wav":[]}
        # process the audio
        audio, sr = load_audio_into_tensor(0, self.config.datasets.root, audios, self.config.datasets.resampling_rate, self.config.datasets.audio_duration, self.config.datasets.resample)
        audio = audio.reshape(-1)
        audio = [torch.tensor(audio)]
        raw_wav = pad_sequence(audio, batch_first=True, padding_value=0)

        samples = {"raw_wav":raw_wav}

        model = unwrap_dist_model(self.model, use_distributed=self.use_distributed)
        model.eval()

        samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled) # to cuda

        text = model.generate(samples, self.config.generate, prompts)
        response = post_process(text[0])

        return response


    
    @torch.no_grad()
    def valid_generate_epoch(self, epoch, split):
        save_file = self.config.datasets.save_file + f'_ep_{epoch}'
        logging.info("saving to {}".format(save_file))
        
        model = unwrap_dist_model(self.model, use_distributed=self.use_distributed)
        model.eval()
        
        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval Generate: data epoch: [{}]".format(epoch)
        logging.info("\n=====  Generating Parameters    =====")
        logging.info(self.config.generate)
        
        out_captions_pred = []
        for samples in metric_logger.log_every(dataloader, self.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled) # to cuda
            metric_logger.data_time.update(time.time() - metric_logger.end)
        
            prompts = None
            if model.prompt_dict and self.test_prompt_dict is not None:
                prompts = []
                special_tokens = samples['special_token']
                for i, task in enumerate(samples['task']):
                    x = self.test_prompt_dict[task]
                    x = re.sub(r'(<Speech><SpeechHere></Speech>)(.*)', rf'\1 {special_tokens[i]}\2', x)
                    prompts.append(x)
                
                if "Q" in samples:
                    prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
            
            # print(prompts[0])
            text = model.generate(samples, self.config.generate, prompts)
            
            if not self.isAIRBench:
                for i in range(len(samples['id'])):
                    out_captions_pred.append({
                        'Q': samples["Q"][i],
                        'audio_name': samples["id"][i],
                        "pred": post_process(text[i]),
                        "target": post_process(samples["output"][i])
                    })
            else:
                for i in range(len(samples['id'])):
                    out_captions_pred.append({
                        "meta_info": samples['meta_info'][i],
                        "question": samples['Q'][i],
                        "answer_gt": post_process(samples["output"][i]),
                        "path": samples["id"][i],
                        "task_name": samples['task_name'][i],
                        "dataset_name": samples['dataset_name'][i],
                        "response": post_process(text[i]),
                        "uniq_id": samples["uniq_id"][i],
                    })
            
            self.save_results(out_captions_pred, self.output_dir, save_file)
        
        self.save_results(out_captions_pred, self.output_dir, save_file)
