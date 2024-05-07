import os
from transformers import TrainingArguments
from trl import DPOTrainer
import os
import wandb
from accelerate import Accelerator


class TrainerDPO:
    def __init__(self, config):
        self.output_dir = str(config["experiment_dir"] / "dpo")
        self.accelerator = Accelerator()
        self.dpo_training_params = config["dpo_training"]

    def train(self, model, tokenizer, lora_config, dataset):
        learning_rate = float(self.dpo_training_params["learning_rate"])
        batch_size = self.dpo_training_params["batch_size"]
        max_seq_length = self.dpo_training_params["max_seq_length"]
        max_prompt_length = self.dpo_training_params["max_prompt_length"]

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
            save_steps=50,
            report_to="wandb",
        )

        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_length=max_seq_length,
            max_prompt_length=max_prompt_length,
            tokenizer=tokenizer,
            args=training_args,
        )

        model, trainer = self.accelerator.prepare(model, trainer)

        trainer.train()

        output_dir = os.path.join(self.output_dir, "model")
        trainer.model.save_pretrained(output_dir)
