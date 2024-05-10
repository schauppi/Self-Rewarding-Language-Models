import logging
import yaml
from pathlib import Path
import os
import datetime

from src.utils.logging.logging_config import setup_logging


class ConfigLoader:
    """
    Class for loading the configuration file and setting up paths.

    Methods:
    load_config: Loads the configuration file.
    setup_paths: Sets up the paths for the project based on the configuration.
    setup_experiment_dir: Sets up the experiment directory based on the configuration.
    """

    def __init__(self, config_file: str = "config.yaml"):
        try:

            setup_logging()
            self.logger = logging.getLogger()
        except ImportError:
            print(
                "Module 'src.utils.logging.logging_config' not found. Logging setup skipped."
            )

        self.config = self.load_config(config_file)
        self.setup_paths()
        self.setup_experiment_dir()

    def load_config(self, config_file: str) -> dict:
        """
        Loads the configuration file.

        Args:
            config_file: The name of the configuration file.

        Returns:
            The loaded configuration as a dictionary.
        """
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
        """
        Sets up the paths for the project based on the configuration.
        """
        base_path = Path(
            os.getenv(
                "PROJECT_BASE_PATH", Path(__file__).resolve().parent.parent.parent
            )
        )
        self.logger.info(f"Base path set to: {base_path}")
        self.config["ift_data_path"] = (
            base_path / self.config["data_directory"]
        ).resolve()
        self.config["model_dir_path"] = (
            base_path / self.config["model_directory"]
        ).resolve()

        self.logger.info(f"IFT Data path set to: {self.config['ift_data_path']}")
        self.logger.info(
            f"Model directory path set to: {self.config['model_dir_path']}"
        )

    def setup_experiment_dir(self):
        """
        Sets up the experiment directory based on the configuration.
        """
        try:
            base_path = Path(
                os.getenv(
                    "PROJECT_BASE_PATH", Path(__file__).resolve().parent.parent.parent
                )
            )
            current_datetime = os.getenv(
                "CURRENT_DATETIME", ""
            ) or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.config["experiment_dir"] = (
                base_path / "results" / f"results_{current_datetime}"
            ).resolve()
            self.logger.info(
                f"Experiment directory set to: {self.config['experiment_dir']}"
            )
            os.makedirs(self.config["experiment_dir"], exist_ok=True)
            self.config["data_path"] = self.config["experiment_dir"] / "data"
            self.logger.info(f"Data path set to: {self.config['data_path']}")
            os.makedirs(self.config["data_path"], exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e
