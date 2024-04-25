import os
from transformers import TrainingArguments
from trl import DPOTrainer
import os
import wandb
from accelerate import Accelerator
import datetime

os.environ["WANDB_PROJECT"]="Self Rewarding Language Models"

class TrainerDPO:
    def __init__(self):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f'./results/results_dpo_{current_datetime}'

    def train(self, model, tokenizer, lora_config, dataset, accelerator):
        learning_rate=5e-5
        batch_size = 4
        max_length = 1536
        max_prompt_length = 1024

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
            report_to="wandb",)

        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            tokenizer=tokenizer,
            args=training_args,)

        model, trainer = accelerator.prepare(model, trainer)

        trainer.train()

        output_dir = os.path.join(self.output_dir, "model")
        trainer.model.save_pretrained(output_dir)