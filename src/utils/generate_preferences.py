import logging
import os
import pandas as pd
import json
from typing import Dict, Any, List, Union
from pathlib import Path


from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def generate(scores_path: str, output_path: str) -> str:
    """
    Generates preference pairs from the scores and saves them to the output path.

    Args:
        scores_path: The path to the scores.
        output_path: The path to save the preference pairs.

    Returns:
        The output path where the preference pairs were saved.
    """
    prompts: Dict[str, List[Dict[str, Any]]] = {}

    try:
        with open(scores_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]
                if prompt_id not in prompts:
                    prompts[prompt_id] = []
                prompts[prompt_id].append(data)

        pairs: List[Dict[str, Any]] = []

        for prompt_id, prompt_group in prompts.items():
            best_prompt, worst_prompt = None, None
            max_score, min_score = float("-inf"), float("inf")

            for prompt in prompt_group:
                if prompt["score"] > max_score:
                    max_score = prompt["score"]
                    best_prompt = prompt
                if prompt["score"] < min_score:
                    min_score = prompt["score"]
                    worst_prompt = prompt

            if best_prompt and worst_prompt:
                pairs.append(
                    {
                        "prompt_id": best_prompt["prompt_id"],
                        "prompt": best_prompt["prompt"],
                        "chosen": best_prompt["completion"],
                        "rejected": worst_prompt["completion"],
                        "score_chosen": best_prompt["score"],
                        "score_rejected": worst_prompt["score"],
                    }
                )

        df_pairs = pd.DataFrame(pairs)
        df_pairs.to_json(output_path, lines=True, orient="records")
        return output_path
    except Exception as e:
        logger.error(f"Error in generate: {e}")
        return ""


def generate_preferences(
    config: Dict[str, Union[str, Path]], iteration: int, scores_path: str
) -> Union[str, None]:
    """
    Generates preference pairs for the given iteration and saves them to the output path.

    Args:
        config: The configuration dictionary.
        iteration: The current iteration.
        scores_path: The path to the scores.

    Returns:
        The output path where the preference pairs were saved, or None if an error occurred.
    """
    try:
        logger.info(f"Generating preference pairs for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "preference_pairs.jsonl"
        logger.info(f"Output path: {output_path}")
        return generate(scores_path=scores_path, output_path=output_path)
    except Exception as e:
        logger.error(f"Error in generate_preferences: {e}")
        return None
