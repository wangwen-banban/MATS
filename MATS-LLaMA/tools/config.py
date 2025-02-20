
import json
import logging

from omegaconf import OmegaConf


class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        if args.cfg_path1:
            config1 = OmegaConf.load(self.args.cfg_path1)
            config = OmegaConf.merge(config, config1)
        config = OmegaConf.merge(config, user_config)
        self.config = config
    
    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts
        
        out = []
        for opt, value in zip(opts[0::2], opts[1::2]):
            out.append((opt + "=" + str(value)))
            
        return out

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)
    
    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.datasets))

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)