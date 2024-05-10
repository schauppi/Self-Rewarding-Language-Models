import os
import logging
from datasets import load_dataset
from typing import Dict, Any
from datasets import Dataset

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def get_prompt(example: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """
    Gets the prompt from the example using the provided tokenizer.

    Args:
        example: The example to get the prompt from.
        tokenizer: The tokenizer to be used.

    Returns:
        The example with the prompt added.
    """
    try:
        prompt_sample = [{"role": "user", "content": example["prompt"]}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        example["prompt"] = model_prompt
        example["chosen"] = example["chosen"]
        example["rejected"] = example["rejected"]
        return example
    except Exception as e:
        logger.error(f"Error: {e}")
        return {}


def generate_dpo_dataset(preferences_path: str, tokenizer: Any) -> Dataset:
    """
    Generates a DPO dataset from the provided preferences path and tokenizer.

    Args:
        preferences_path: The path to the preferences.
        tokenizer: The tokenizer to be used.

    Returns:
        The generated DPO dataset.
    """
    try:
        logger.info(f"Generating DPO dataset from preferences: {preferences_path}")
        preferences_path_str = str(preferences_path)
        dataset = load_dataset("json", data_files={"train": preferences_path_str})
        dataset = dataset["train"].shuffle(seed=42)
        dataset = dataset.map(lambda data: get_prompt(data, tokenizer))
        return dataset
    except Exception as e:
        logger.error(f"Error generating DPO dataset: {e}")
        return None
