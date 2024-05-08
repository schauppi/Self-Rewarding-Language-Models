import logging
import os
from transformers import TextStreamer
import pandas as pd

from src.utils.logging.logging_config import setup_logging

setup_logging()
logger = logging.getLogger()


def trim_completion(completion):
    try:
        if "\n" in completion:
            last_newline = completion.rfind("\n")
            completion = completion[:last_newline]
            return completion.strip()
        else:
            return completion
    except Exception as e:
        logger.error(f"Error in trim_completion: {e}")
        return ""


def extract_completion(answer):
    try:
        pattern = f"[/INST]"
        parts = answer.split(pattern)
        if len(parts) > 1:
            return parts[-1]
        else:
            return ""
    except Exception as e:
        logger.error(f"Error in extract_completion: {e}")
        return ""


def do_sample(model, tokenizer, prompt):
    try:
        prompt_sample = [{"role": "user", "content": prompt}]
        model_prompt = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        model_inputs = tokenizer(model_prompt, return_tensors="pt").to("cuda")
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

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        logger.error(f"Error in do_sample: {e}")
        return ""


def generate(model, tokenizer, gen_prompts, responses_to_generate, output_path):
    logger.info(f"Generating {responses_to_generate} responses per prompt")
    completions = []
    try:
        for _, row in gen_prompts.iterrows():
            prompt = row["prompt"]
            prompt_id = row["id"]

            for completion_sample in range(responses_to_generate):
                logger.info(
                    f"Processing completion {completion_sample + 1} of {responses_to_generate}"
                )

                answer = do_sample(model, tokenizer, prompt)
                completion = extract_completion(answer)
                trimmed_completion = trim_completion(completion)

                completions.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "completion": trimmed_completion,
                    }
                )

                df_completions = pd.DataFrame(completions)
                df_completions.to_json(output_path, orient="records", lines=True)
    except Exception as e:
        logger.error(f"Error in generate: {e}")


def generate_responses(model, tokenizer, config, iteration, prompts_path):
    try:
        logger.info(f"Generating responses for iteration {iteration}")
        output_dir = config["data_path"] / f"{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "gen_responses.jsonl"
        logger.info(f"Output path: {output_path}")

        gen_prompts = pd.read_json(prompts_path, lines=True)
        generate(
            model=model,
            tokenizer=tokenizer,
            gen_prompts=gen_prompts,
            responses_to_generate=config["response_prompts"]["new_prompts"],
            output_path=output_path,
        )
        return output_path
    except Exception as e:
        logger.error(f"Error in generate_responses: {e}")
