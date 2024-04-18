# Assuming your script is set up as mentioned previously:
from src.utils.model import load_model, create_peft_model
from src.utils.trainer import Trainer
import os
from datasets import load_dataset
from accelerate import Accelerator

base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, '..', 'data/ift.jsonl')
dataset_path = os.path.normpath(dataset_path)

accelerator = Accelerator()

def create_chat_template(tokenizer, x):
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": x["prompt"]},
        {"role": "assistant", "content": x["completion"]},
    ], tokenize=False)
    return {"text": text}

model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1")

dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset["train"].shuffle(seed=42)
dataset = dataset.map(lambda x: create_chat_template(tokenizer, x))

model, lora_config = create_peft_model(model)
trainer = Trainer()
trainer.train(model, tokenizer, lora_config, dataset, accelerator)
