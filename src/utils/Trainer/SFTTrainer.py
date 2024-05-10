from typing import Any, Dict
from transformers import TrainingArguments
from transformers import PreTrainedTokenizer, PreTrainedModel
from trl import SFTTrainer
import os
import wandb
from accelerate import Accelerator
import logging

from src.utils.logging.logging_config import setup_logging


class TrainerSFT:
    """
    Trainer class for SFT model

    Methods:
    train: Trains the SFT model
    """

    def __init__(self, config: Dict[str, Any], iteration: int):
        """
        Initializes the TrainerSFT class

        Args:
            config: The configuration for the model
            iteration: The iteration number
        """
        self.output_dir = str(
            config["experiment_dir"] / "sft" / f"iteration_{iteration}"
        )
        self.accelerator = Accelerator()
        self.sft_training_params = config["sft_training"]
        if config["wandb_enable"] == True:
            self.report_to = "wandb"
        else:
            self.report_to = None

        try:
            setup_logging()
            self.logger = logging.getLogger()
        except ImportError:
            print(
                "Module 'src.utils.logging.logging_config' not found. Logging setup skipped."
            )

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        lora_config: Dict[str, Any],
        dataset: Any,
    ):
        """
        Trains the SFT model

        Args:
            model: The model to be trained
            tokenizer: The tokenizer for the model
            lora_config: The configuration for the model
            dataset: The dataset to be used for training

        Returns:
            None
        """
        self.logger.info("Training SFT model")
        try:
            learning_rate = float(self.sft_training_params["learning_rate"])
            batch_size = self.sft_training_params["batch_size"]
            max_seq_length = self.sft_training_params["max_seq_length"]
            self.logger.info(f"Training parameters loaded: {self.sft_training_params}")
        except KeyError as e:
            self.logger.error(f"KeyError: {e}")
            return

        try:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                gradient_accumulation_steps=4,
                warmup_steps=30,
                weight_decay=0.001,
                logging_steps=1,
                num_train_epochs=1,
                lr_scheduler_type="cosine",
                optim="paged_adamw_32bit",
                report_to=self.report_to,
            )
        except Exception as e:
            self.logger.error(f"Error in TrainingArguments: {e}")
            return

        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=lora_config,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                args=training_args,
                dataset_text_field="text",
            )
        except Exception as e:
            self.logger.error(f"Error during trainer setup: {e}")
            return

        try:
            model, trainer = self.accelerator.prepare(model, trainer)
            trainer.train()
            trainer.model.save_pretrained(self.output_dir)
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
