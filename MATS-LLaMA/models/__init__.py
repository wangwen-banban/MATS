from .mats import AudioFreeMLLM

def load_model(config):
    model = AudioFreeMLLM.from_config(config)
    return model