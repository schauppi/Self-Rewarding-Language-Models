import os
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import wandb
from accelerate import Accelerator
import datetime

os.environ["WANDB_PROJECT"]="Self Rewarding Language Models"

class Trainer:
    def __init__(self):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f'./results/results_{current_datetime}'

    def train(self, model, tokenizer, lora_config, dataset, accelerator):
        learning_rate = 2e-4
        batch_size = 8
        max_seq_length = 1024

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            logging_steps=1,
            num_train_epochs=1,
            save_steps=50,
            report_to="wandb",)

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text")

        model, trainer = accelerator.prepare(model, trainer)

        trainer.train()

        output_dir = os.path.join(self.output_dir, "model")
        trainer.model.save_pretrained(output_dir)