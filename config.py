import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key):
        return self.config.get(key)