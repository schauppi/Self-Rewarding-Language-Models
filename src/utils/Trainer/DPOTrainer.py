from typing import Any, Dict
from transformers import TrainingArguments
from transformers import PreTrainedTokenizer, PreTrainedModel
from trl import DPOTrainer
from accelerate import Accelerator
import logging

from src.utils.logging.logging_config import setup_logging


class TrainerDPO:
    """
    Trainer class for DPO model

    Methods:
    train: Trains the DPO model
    """

    def __init__(self, config: Dict[str, Any], iteration: int):
        """
        Initializes the TrainerDPO class

        Args:
            config: The configuration for the model
            iteration: The iteration number
        """
        self.output_dir = str(
            config["experiment_dir"] / "dpo" / f"iteration_{iteration}"
        )
        self.accelerator = Accelerator()
        self.dpo_training_params = config["dpo_training"]
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
        Trains the DPO model

        Args:
            model: The model to be trained
            tokenizer: The tokenizer for the model
            lora_config: The configuration for the model
            dataset: The dataset to be used for training

        Returns:
            None
        """
        self.logger.info("Training DPO model")
        try:
            learning_rate = float(self.dpo_training_params["learning_rate"])
            batch_size = self.dpo_training_params["batch_size"]
            max_seq_length = self.dpo_training_params["max_seq_length"]
            max_prompt_length = self.dpo_training_params["max_prompt_length"]
            self.logger.info(f"Training parameters loaded: {self.dpo_training_params}")
        except KeyError as e:
            print(f"Key {e} not found in training parameters.")
            return

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

        try:
            trainer = DPOTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=lora_config,
                max_length=max_seq_length,
                max_prompt_length=max_prompt_length,
                tokenizer=tokenizer,
                args=training_args,
            )
        except Exception as e:
            print(f"Error during trainer setup: {e}")
            return

        try:
            model, trainer = self.accelerator.prepare(model, trainer)
            trainer.train()
            trainer.model.save_pretrained(self.output_dir)
        except Exception as e:
            print(f"Error during training: {e}")
