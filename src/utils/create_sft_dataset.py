import logging
from datasets import load_dataset
from typing import Dict, Any
from transformers import PreTrainedTokenizer
from datasets import Dataset

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def create_chat_template(
    tokenizer: PreTrainedTokenizer, x: Dict[str, Any]
) -> Dict[str, str]:
    """
    Creates a chat template using the provided tokenizer and input dictionary.

    Args:
        tokenizer: The tokenizer to be used.
        x: The input dictionary containing the 'prompt' and 'completion'.

    Returns:
        A dictionary containing the text created from the chat template.
    """
    try:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": x["prompt"]},
                {"role": "assistant", "content": x["completion"]},
            ],
            tokenize=False,
        )
        return {"text": text}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"text": ""}


def create_sft_dataset(dataset_path: str, tokenizer: Any) -> Dataset:
    """
    Creates a SFT dataset from the provided dataset path and tokenizer.

    Args:
        dataset_path: The path to the dataset.
        tokenizer: The tokenizer to be used.

    Returns:
        The created SFT dataset.
    """
    try:
        dataset_path = str(dataset_path)
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset["train"].shuffle(seed=42)
        dataset = dataset.map(lambda x: create_chat_template(tokenizer, x))
        return dataset
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
