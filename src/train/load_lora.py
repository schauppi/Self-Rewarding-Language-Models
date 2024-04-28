from pathlib import Path
import os
import logging
from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader
from src.utils.create_sft_dataset import create_sft_dataset
from src.utils.SFTTrainer import TrainerSFT

from peft import load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

setup_logging()
logger = logging.getLogger()


config_loader = ConfigLoader()
config = config_loader.config

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
os.environ["WANDB_PROJECT"] = config["wandb_project"]

loader = ModelLoader(
    config,
    adapter=True,
    adapter_path="/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-04-28_16-03-28/sft",
)
model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config


"""def set_peft_model_state_dict(model, state_dict):

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    return model


model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
)

lora_weights = load_peft_weights(
    "/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-04-28_16-03-28/sft"
)
model = set_peft_model_state_dict(model, lora_weights)"""
