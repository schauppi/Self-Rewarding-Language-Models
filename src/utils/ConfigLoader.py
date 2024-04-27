from src.utils.logging.logging_config import setup_logging
import logging
import yaml
from pathlib import Path
import os


class ConfigLoader:
    def __init__(self, config_file="config.yaml"):
        setup_logging()
        self.logger = logging.getLogger()
        self.config = self.load_config(config_file)
        self.setup_paths()

    def load_config(self, config_file):
        # Compute the path to the configuration file
        config_path = Path(
            os.getenv(
                "PROJECT_CONFIG_PATH",
                Path(__file__).resolve().parent.parent.parent / config_file,
            )
        )
        self.logger.info(f"Attempting to load config file at: {config_path}")

        if not config_path.exists():
            self.logger.error(f"Config file not found at: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def setup_paths(self):
        base_path = Path(
            os.getenv("PROJECT_BASE_PATH", Path(__file__).resolve().parent.parent)
        )
        self.logger.info(f"Base path set to: {base_path}")
        self.config["data_path"] = (base_path / self.config["data_directory"]).resolve()
        self.config["model_dir_path"] = (
            base_path / self.config["model_directory"]
        ).resolve()

        self.logger.info(f"Data path set to: {self.config['data_path']}")
        self.logger.info(
            f"Model directory path set to: {self.config['model_dir_path']}"
        )
