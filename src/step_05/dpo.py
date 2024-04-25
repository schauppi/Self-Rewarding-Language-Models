import os

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from src.utils.dpo_trainer import TrainerDPO
from accelerate import Accelerator

accelerator = Accelerator()

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, '..', 'data/')
dpo_dataset_path = os.path.normpath(data_path) + '/preference_pairs.jsonl'
model_dir_path = os.path.join(base_path, '..', '..', 'results/')
model_dir_path = os.path.normpath(model_dir_path) + '/results_2024-04-18_12-44-14/checkpoint-150'

dataset = load_dataset('json', data_files={'train': dpo_dataset_path})
dataset = dataset['train'].shuffle(seed=42)

def get_prompt(example, tokenizer):
    prompt_sample = [
        {"role": "user", "content": example['prompt']}
    ]

    model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example['prompt'] = model_prompt
    example['chosen'] = example['chosen']
    example['rejected'] = example['rejected']

    return example

def get_bnb_config():
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    return config

def load_fine_tuned(model_dir):
    bnb_config = get_bnb_config()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)

    return model, tokenizer

model, tokenizer = load_fine_tuned(model_dir_path)
dataset = dataset.map(lambda data: get_prompt(data, tokenizer))

def create_peft_model(model):
    lora_dropout=0.5
    lora_alpha=16
    lora_r=16

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_model(model)
trainer = TrainerDPO()
trainer.train(model, tokenizer, lora_config, dataset, accelerator)