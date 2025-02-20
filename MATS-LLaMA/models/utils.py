import logging
import math
import torch
import json
from transformers import StoppingCriteria
import torch.nn.functional as F

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[0][-len(stop):] == stop).item():
                return True      
        return False
    
def load_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except:
        print(f"Failed to load {path}! Try to use utf-8 encoding.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data

def count_parameters(model, name):
    """
        Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    trainable_params = trainable_params / 1e9 
    all_param = all_param / 1e9
    logging.info(
        "{} trainable params: {:.4f} B || all params: {:.4f} B || trainable: {:.3%}".format(
            name, trainable_params, all_param, trainable_params / all_param)
    )


def get_uniform_ball_noise(input_shape, device, radius=0.1):
    uniform_noise_ball = torch.randn(input_shape, device=device)  # normal distribution
    uniform_noise_sphere = torch.nn.functional.normalize(uniform_noise_ball, dim=1)
    u = torch.rand(input_shape[0], device=device)  # unified distribution
    u = u ** (1. / input_shape[1])
    uniform_noise_ball = (uniform_noise_sphere.T * u * radius).T
    return uniform_noise_ball

def noise_injection(x, noise_variance, device, uniform_noise=False):
    # x: clip text features [B, C]
    variance = noise_variance
    if variance == 0.0:
        return x
    
    std = math.sqrt(variance)
    x = F.normalize(x, p=2, dim=-1)
    
    if uniform_noise:
        x = x + get_uniform_ball_noise(x.shape, device, radius=std)
    else:
        x = x + torch.randn(x.shape, device=device) * std
    
    x = F.normalize(x, p=2, dim=-1)
    return x

def load_memory(clap_wrapper):
    path = '/home/houruibing/code/AUDIO/data/audio_free_AudioCaps_stage1.json'
    data = load_json(path)["annotation"]
    train_captions = []
    for item in data:
        train_captions.append(item['caption'])
    
    memory_text_features = []
    batch_size = 64
    for i in range(len(train_captions) // batch_size + 1):
        if (i * batch_size) >= len(train_captions):
            break 
        inputs = train_captions[i * batch_size: (i + 1) * batch_size]
        text_embeds = clap_wrapper.get_text_embeddings(inputs)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        memory_text_features.append(text_embeds)
    memory_text_features = torch.cat(memory_text_features, dim=0)
    return memory_text_features
        
        

    
