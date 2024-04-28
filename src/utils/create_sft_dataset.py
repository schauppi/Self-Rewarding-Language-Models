import logging
from datasets import load_dataset

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def create_chat_template(tokenizer, x):
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


def create_sft_dataset(dataset_path: str, tokenizer: object):
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
