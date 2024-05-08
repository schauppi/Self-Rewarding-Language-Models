import os
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import wandb
from accelerate import Accelerator


class TrainerSFT:
    def __init__(self, config, iteration=0):
        self.output_dir = str(
            config["experiment_dir"] / "sft" / f"iteration_{iteration}"
        )
        self.accelerator = Accelerator()
        self.sft_training_params = config["sft_training"]
        if config["wandb_enable"] == True:
            self.report_to = "wandb"
        else:
            self.report_to = None

    def train(self, model, tokenizer, lora_config, dataset):
        learning_rate = float(self.sft_training_params["learning_rate"])
        batch_size = self.sft_training_params["batch_size"]
        max_seq_length = self.sft_training_params["max_seq_length"]

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

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
        )

        model, trainer = self.accelerator.prepare(model, trainer)

        trainer.train()
        trainer.model.save_pretrained(self.output_dir)
