from pathlib import Path
import os
import logging
from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader
from src.utils.create_sft_dataset import create_sft_dataset
from src.utils.Trainer.SFTTrainer import TrainerSFT
from src.utils.Trainer.DPOTrainer import TrainerDPO
from src.utils.generate_prompts import generate_new_prompts
from src.utils.generate_responses import generate_responses
from src.utils.generate_scores import generate_scores
from src.utils.generate_preferences import generate_preferences
from src.utils.generate_dpo_dataset import generate_dpo_dataset


class TrainingPipeline:
    """
    Pipeline for training the SFT and DPO models

    Methods:
    - setup_environment: Sets up the environment variables
    - train_sft_model: Trains the SFT model
    - run_iteration: Runs an iteration of the pipeline
    - run: Runs the pipeline
    """

    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger()
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config
        self.setup_environment()
        self.loader = None
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.sft_adapter_path = None
        self.dpo_adapter_path = None

    def setup_environment(self):
        """
        Sets up the environment variables
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["cuda_visible_devices"]
        if self.config["wandb_enable"] == True:
            os.environ["WANDB_PROJECT"] = self.config["wandb_project"]
        else:
            os.environ["WANDB_MODE"] = "disabled"
        self.logger.info(f"WandB Enabled: {self.config['wandb_enable']}")

    def train_sft_model(self):
        self.logger.info("Step 0: Training SFT model")
        self.loader = ModelLoader(self.config)
        self.model, self.tokenizer, self.lora_config = (
            self.loader.model,
            self.loader.tokenizer,
            self.loader.lora_config,
        )
        dataset = create_sft_dataset(
            dataset_path=(self.config["ift_data_path"] / self.config["ift_dataset"]),
            tokenizer=self.tokenizer,
        )
        sft_trainer = TrainerSFT(config=self.config, iteration=0)
        self.sft_adapter_path = sft_trainer.output_dir
        sft_trainer = sft_trainer.train(
            model=self.model,
            tokenizer=self.tokenizer,
            lora_config=self.lora_config,
            dataset=dataset,
        )

    def run_iteration(self, iteration):
        """
        Runs an iteration of the pipeline

        Args:
        - iteration: The iteration number
        """
        self.logger.info(f"Starting iteration {iteration}")

        self.logger.info(f"Step 1 | iteration {iteration}: Generating new prompts")
        if iteration == 0:
            self.loader = ModelLoader(
                self.config, adapter=True, adapter_path=self.sft_adapter_path
            )
        else:
            self.loader = ModelLoader(
                self.config, adapter=True, adapter_path=self.dpo_adapter_path
            )
        self.model, self.tokenizer, self.lora_config = (
            self.loader.model,
            self.loader.tokenizer,
            self.loader.lora_config,
        )
        prompts_path = generate_new_prompts(
            self.model, self.tokenizer, self.config, iteration
        )

        self.logger.info(f"Step 2 | iteration {iteration}: Generating responses")
        responses_path = generate_responses(
            self.model, self.tokenizer, self.config, iteration, prompts_path
        )

        self.logger.info(f"Step 3 | iteration {iteration}: Generating scores")
        scores_path = generate_scores(
            self.model, self.tokenizer, self.config, iteration, responses_path
        )

        self.logger.info(f"Step 4 | iteration {iteration}: Generating preferences")
        preferences_path = generate_preferences(self.config, iteration, scores_path)

        self.logger.info(f"Step 5 | iteration {iteration}: Training DPO model")
        dpo_dataset = generate_dpo_dataset(preferences_path, self.tokenizer)

        dpo_trainer = TrainerDPO(config=self.config, iteration=iteration)
        self.dpo_adapter_path = dpo_trainer.output_dir
        dpo_trainer = dpo_trainer.train(
            model=self.model,
            tokenizer=self.tokenizer,
            lora_config=self.lora_config,
            dataset=dpo_dataset,
        )

    def run(self):
        """
        Runs the pipeline
        """
        try:
            self.train_sft_model()
            for iteration in range(self.config["iterations"]):
                self.run_iteration(iteration)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            raise e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
