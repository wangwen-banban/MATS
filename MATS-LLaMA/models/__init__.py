from .audiofree_salmonn import AudioFreeMLLM

def load_model(config):
    if config.name == 'audiofree':
        model = AudioFreeMLLM.from_config(config)
    else:
        raise NotImplementedError
    return model