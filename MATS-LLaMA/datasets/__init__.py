import logging
from tools.utils import get_dataloader, MultiIterLoader
from .clap_dataset import CLAPDataset
# from .laion_clap_datasets import LaionCLAPDataset

def load_dataset(config):
    resampling_rate = config.resampling_rate
    audio_duration = config.audio_duration
    resample = config.resample
    audiofree = config.audiofree
    root = config.root
    train_config = config.train_data

    config.isAIRBench = config.get('isAIRBench', False)
    
    train_dataset = []
    for name in train_config:
        train_path = train_config[name]["path"]
        sample_ratio = train_config[name]["sample_ratio"]
        special_token = train_config[name]['special_token']
        dataset = CLAPDataset(train_path, 
                            resampling_rate=resampling_rate, 
                            audio_duration=audio_duration, 
                            resample=resample,
                            audiofree=audiofree,
                            root=root,
                            sample_ratio=sample_ratio,
                            special_token=special_token)
        train_dataset.append(dataset)
        
    valid_dataset = CLAPDataset(config.valid_ann_path, 
                                resampling_rate=resampling_rate, 
                                audio_duration=audio_duration, 
                                resample=resample,
                                audiofree=False,
                                root=root,
                                special_token=config.special_token)
    test_dataset = CLAPDataset(config.test_ann_path, 
                               resampling_rate=resampling_rate, 
                               audio_duration=audio_duration, 
                               resample=resample,
                               audiofree=False,
                               root=root,
                               special_token=config.special_token,
                               isAIRBench=config.isAIRBench)

    logging.info(train_dataset[0].task2Salmonntask)
    
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
    }
    return datasets

def load_laion_dataset(config):
    resampling_rate = config.resampling_rate
    audio_duration = config.audio_duration
    resample = config.resample
    audiofree = config.audiofree
    root = config.root
    train_config = config.train_data
    
    train_dataset = []
    for name in train_config:
        train_path = train_config[name]["path"]
        sample_ratio = train_config[name]["sample_ratio"]
        special_token = train_config[name]['special_token']
        dataset = LaionCLAPDataset(train_path, 
                            resampling_rate=resampling_rate, 
                            audio_duration=audio_duration, 
                            resample=resample,
                            audiofree=audiofree,
                            root=root,
                            sample_ratio=sample_ratio,
                            special_token=special_token)
        train_dataset.append(dataset)
        
    valid_dataset = LaionCLAPDataset(config.valid_ann_path, 
                                resampling_rate=resampling_rate, 
                                audio_duration=audio_duration, 
                                resample=resample,
                                audiofree=False,
                                root=root,
                                special_token=config.special_token)
    test_dataset = LaionCLAPDataset(config.test_ann_path, 
                               resampling_rate=resampling_rate, 
                               audio_duration=audio_duration, 
                               resample=resample,
                               audiofree=False,
                               root=root,
                               special_token=config.special_token)

    logging.info(train_dataset[0].task2Salmonntask)
    
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
    }
    return datasets
    
def load_dataloader(datasets, run_config, use_distributed):
    # dataloaders
    train_loader_list = []
    dataset_ratios = []
    for d in datasets["train"]:
        loader = get_dataloader(
            d, 
            run_config, 
            is_train=True, 
            use_distributed=use_distributed
        )
        train_loader_list.append(loader)
        dataset_ratios.append(d.sample_ratio)
    
    train_loader = MultiIterLoader(train_loader_list, dataset_ratios)
    
    valid_loader = get_dataloader(
        datasets["valid"], 
        run_config, 
        is_train=False, 
        use_distributed=use_distributed
    )
    test_loader = get_dataloader(
        datasets["test"], 
        run_config, 
        is_train=False, 
        use_distributed=use_distributed
    )
    return train_loader, valid_loader, test_loader