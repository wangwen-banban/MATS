# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/logger.py

import datetime
import logging
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from .dist_utils import is_dist_avail_and_initialized, is_main_process

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg: .4f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0.0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += 1
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_time = None
        self.data_time = None
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)
    
    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None, logger=None, start_step=None):
        i = 0
        if not header:
            header = ""
        self.start_time = time.time()
        self.end = time.time()
        self.iter_time = SmoothedValue(fmt="{avg:.4f}")
        self.data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]", #[{0:2d}/{1}]
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        for obj in iterable:
            #self.data_time.update(time.time() - end)
            yield obj
            # after one iteration of training 
            self.iter_time.update(time.time() - self.end) 
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                if is_main_process() and logger is not None: # tensorboard
                    assert start_step is not None, "start_step is needed to compute global_step!"
                    for name, meter in self.meters.items():
                        logger.add_scalar(name, float(str(meter)), global_step=start_step+i)       
                eta_seconds = self.iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds))) #h:m:s
                logging.info(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time),
                        data=str(self.data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    )
                )
            
            i += 1
            self.end = time.time()
        
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )
            