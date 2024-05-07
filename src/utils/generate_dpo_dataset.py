import os
import logging
from datasets import load_dataset

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def get_prompt(example, tokenizer):
    prompt_sample = [{"role": "user", "content": example["prompt"]}]

    model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example["prompt"] = model_prompt
    example["chosen"] = example["chosen"]
    example["rejected"] = example["rejected"]

    return example


def generate_dpo_dataset(preferences_path, tokenizer):
    try:
        logger.info(f"Generating DPO dataset from preferences: {preferences_path}")
        dataset = load_dataset("json", data_files={"train": preferences_path})
        dataset = dataset["train"].shuffle(seed=42)
        dataset = dataset.map(lambda data: get_prompt(data, tokenizer))
        return dataset
    except Exception as e:
        logger.error(f"Error generating DPO dataset: {e}")
