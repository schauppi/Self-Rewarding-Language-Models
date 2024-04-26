from src.utils.logging.logging_config import setup_logging
import logging
import yaml

class ConfigLoader:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        setup_logging()
        self.logger = logging.getLogger()
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as config_file:
            try:
                self.logger.info("Reading the config file")
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                self.logger.error(f"Error reading the config file: {exc}")
                raise

        return self.config