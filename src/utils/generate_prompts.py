import pandas as pd
import re
import uuid
import logging
from transformers import TextStreamer
import os
from typing import List, Dict, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.utils.read_write_jsonl import read_jsonl_file, write_jsonl_file
from src.utils.prompts import prompt_step_01
from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def get_random_prompts(data: pd.DataFrame, num_prompts: int = 5) -> List[str]:
    """
    Gets a random sample of prompts from the data.

    Args:
        data: The data to get the prompts from.
        num_prompts: The number of prompts to get.

    Returns:
        A list of random prompts.
    """
    try:
        return data.sample(n=num_prompts)["prompt"].tolist()
    except Exception as e:
        logger.error(f"Error getting random prompts: {e}")
        return []


def generate_prompt(examples: List[str]) -> str:
    """
    Generates a prompt from the examples.

    Args:
        examples: The examples to generate the prompt from.

    Returns:
        The generated prompt.
    """
    global prompt_step_01
    try:
        for _, item in enumerate(examples):
            prompt_step_01 += f"<task>|{item}</task>\n"
        return prompt_step_01
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        return prompt_step_01


def extract_prompt(answer: str) -> List[str]:
    """
    Extracts the prompts from the answer.

    Args:
        answer: The answer to extract the prompts from.

    Returns:
        A list of extracted prompts.
    """
    prompts = []
    try:
        extracted_prompts = re.findall(r"<task>\|(.*?)</task>", answer, re.DOTALL)
        for prompt in extracted_prompts:
            prompts.append(prompt)
    except Exception as e:
        logger.error(f"Error extracting prompts: {e}")
    return prompts


def do_sample(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_prompts: List[str]
) -> str:
    """
    Samples from the model using the task prompts.

    Args:
        model: The model to sample from.
        tokenizer: The tokenizer to use.
        task_prompts: The task prompts to use.

    Returns:
        The sampled text.
    """
    try:
        prompt = generate_prompt(task_prompts)
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tokenizer)
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=256,
            streamer=streamer,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0]
    except Exception as e:
        logger.error(f"Error during sampling: {e}")
        return ""


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    ift_data: pd.DataFrame,
    new_prompts_to_generate: int,
) -> List[Dict[str, Any]]:
    """
    Generates new prompts using the model and tokenizer.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        ift_data: The IFT data to use.
        new_prompts_to_generate: The number of new prompts to generate.

    Returns:
        A list of new prompts.
    """
    uniq_prompts = set()
    new_prompts = []
    try:
        while len(uniq_prompts) < new_prompts_to_generate:
            task_prompts = get_random_prompts(ift_data)
            answer = do_sample(model, tokenizer, task_prompts)
            prompts = extract_prompt(answer)
            for prompt in prompts:
                if prompt not in uniq_prompts:
                    uniq_prompts.add(prompt)
                    prompt_id = str(uuid.uuid4())
                    new_prompts.append(
                        {"id": prompt_id, "prompt": prompt, "source": "generated"}
                    )
        return new_prompts
    except Exception as e:
        logger.error(f"Error generating new prompts: {e}")
        return []


def generate_new_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    iteration: int,
) -> Union[str, None]:
    """
    Generates new prompts for the given iteration and saves them to the output path.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        config: The configuration dictionary.
        iteration: The current iteration.

    Returns:
        The output path where the new prompts were saved, or None if an error occurred.
    """
    try:
        logger.info(f"Generating new prompts for iteration {iteration}")
        ift_data = read_jsonl_file(config["ift_data_path"] / config["ift_dataset"])
        new_prompts = generate(
            model=model,
            tokenizer=tokenizer,
            ift_data=ift_data,
            new_prompts_to_generate=config["generate_prompts"]["new_prompts"],
        )
        logger.info(f"Generated {len(new_prompts)} new prompts")
        new_prompts_df = pd.DataFrame(new_prompts)
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_prompts.jsonl"
        write_jsonl_file(new_prompts_df, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_new_prompts: {e}")
