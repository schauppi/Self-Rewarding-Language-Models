import os
import logging
import re
import pandas as pd

from src.utils.prompts import judge_prompt
from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def do_sample(model, tokenizer, prompt):
    try:
        prompt_sample = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        model_inputs = tokenizer(model_prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=100,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        logger.error(f"Error in do_sample: {e}")
        return ""


def extract_scores(answer):
    try:
        pattern = r"[Ss]core: ([0-5])"
        matches = re.findall(pattern, answer)
        score = int(matches[0]) if matches else -1
        return score
    except Exception as e:
        logger.error(f"Error in extract_scores: {e}")
        return -1


def generate(model, tokenizer, gen_respones, output_path):
    results = []
    try:
        for _, row in gen_respones.iterrows():
            prompt = row["prompt"]
            prompt_id = row["prompt_id"]
            completion = row["completion"]

            formatted_llm_prompt = judge_prompt.format(
                prompt=prompt, response=completion
            )

            answer = do_sample(model, tokenizer, formatted_llm_prompt)
            score = extract_scores(answer)
            results.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "completion": completion,
                    "score": score,
                    "reasoning": answer,
                }
            )

            df_results = pd.DataFrame(results)
            df_results.to_json(output_path, orient="records", lines=True)
    except Exception as e:
        logger.error(f"Error in generate: {e}")


def generate_scores(model, tokenizer, config, iteration, responses_path):
    try:
        logger.info(f"Generating scores for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_scores.jsonl"
        logger.info(f"Output path: {output_path}")

        gen_responses = pd.read_json(responses_path, lines=True)

        generate(
            model=model,
            tokenizer=tokenizer,
            gen_respones=gen_responses,
            output_path=output_path,
        )

        return output_path
    except Exception as e:
        logger.error(f"Error in generate_scores: {e}")
