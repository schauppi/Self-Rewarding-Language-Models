import logging
import os
import pandas as pd
import json


from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def generate(scores_path, output_path):
    prompts = {}

    with open(scores_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data["prompt_id"]
            if prompt_id not in prompts:
                prompts[prompt_id] = []
            prompts[prompt_id].append(data)

        pairs = []

    for prompt_id, prompt_group in prompts.items():
        best_prompt = None
        worst_prompt = None
        max_score = float("-inf")
        min_score = float("inf")

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


def generate_preferences(config, iteration, scores_path):
    try:
        logger.info(f"Generating preference pairs for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "preference_pairs.jsonl"
        logger.info(f"Output path: {output_path}")
        generate(scores_path=scores_path, output_path=output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_preferences: {e}")
