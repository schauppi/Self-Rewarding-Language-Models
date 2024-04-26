from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader

import logging
import os

setup_logging()
logger = logging.getLogger()

config = ConfigLoader().config

config['base_path'] = os.path.dirname(__file__)
config['data_path'] = os.path.normpath(os.path.join(config['base_path'], config['data_directory']))
config['model_dir_path'] = os.path.normpath(os.path.join(config['base_path'], config['model_directory']))

os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']

logger.info(f"Visible CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

model = ModelLoader(config).model
