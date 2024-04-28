import pandas as pd
import re
import uuid
import logging
from transformers import TextStreamer
import os

from src.utils.read_write_jsonl import read_jsonl_file, write_jsonl_file
from src.utils.prompts import prompt_step_01
from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def get_random_prompts(data, num_prompts=8):
    try:
        return data.sample(n=num_prompts)["prompt"].tolist()
    except Exception as e:
        logger.error(f"Error getting random prompts: {e}")
        return []


def generate_prompt(examples):
    global prompt_step_01
    try:
        for _, item in enumerate(examples):
            prompt_step_01 += f"<task>|{item}</task>\n"
        return prompt_step_01
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        return prompt_step_01


def extract_prompt(answer):
    prompts = []
    try:
        extracted_prompts = re.findall(r"<task>\|(.*?)</task>", answer, re.DOTALL)
        for prompt in extracted_prompts:
            prompts.append(prompt)
    except Exception as e:
        logger.error(f"Error extracting prompts: {e}")
    return prompts


def do_sample(model, tokenizer, task_prompts):
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


def generate(model, tokenizer, ift_data, new_prompts_to_generate):
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


def generate_new_prompts(model, tokenizer, config, iteration):
    try:
        logger.info(f"Generating new prompts for iteration {iteration}")
        ift_data = read_jsonl_file(config["data_path"] / config["ift_dataset"])
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
        write_jsonl_file(new_prompts_df, (output_dir / "gen_prompts.jsonl"))
    except Exception as e:
        logger.error(f"Error in generate_new_prompts: {e}")
