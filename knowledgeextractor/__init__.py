import os
import json


class KGEConfig(object):
    def __init__(self, config_file=None):
        self.config={}
        if config_file:
            self.loadConfig(config_file=config_file)

    def loadConfig(self,config_file):
        with open(config_file, "r", encoding="utf-8") as f:
           config=json.load(f) 
        
        self.config=config
        def loadConfigCore(config_dict):

            for k, v in self.config.items():
                if type(v)==dict():
                    setattr(self, k, loadConfigCore(v))
                else:
                    setattr(self, k, v)

        loadConfigCore(self.config)
        
        