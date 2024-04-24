import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import re

from src.step_03.utils.llm_as_judge import judge_prompt

def get_bnb_config():
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    return config

def load_fine_tuned(model_dir):
    bnb_config = get_bnb_config()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)

    return model, tokenizer

def do_sample(model, tokenizer, prompt):
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt}
        ]

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
            max_new_tokens=100,
            streamer=streamer,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer


base_path = os.path.dirname(__file__)

data_path = os.path.join(base_path, '..', 'data/')

gen_response_path = os.path.normpath(data_path) + '/gen_responses.jsonl'

gen_score_path = os.path.normpath(data_path) + '/gen_scores.jsonl'

model_dir_path = os.path.join(base_path, '..', '..', 'results/')
model_dir_path = os.path.normpath(model_dir_path) + '/results_2024-04-18_12-44-14/checkpoint-150'

response_df = pd.read_json(gen_response_path, lines=True)

model, tokenizer = load_fine_tuned(model_dir_path)
model.eval()

pattern = r"[Ss]core: ([0-5])"
results = []

for index, row in response_df.iterrows():
    print(f"Processing prompt {index + 1} of {len(response_df)}")
    prompt = row['prompt']
    prompt_id = row['prompt_id']
    completion = row['completion']

    formatted_llm_prompt = judge_prompt.format(prompt=prompt, response=completion)

    answer = do_sample(model, tokenizer, formatted_llm_prompt)
    matches = re.findall(pattern, answer)
    score = int(matches[0]) if matches else -1
    print(f"Score: {score}")

    results.append({
        "prompt_id": prompt_id,
        "prompt": prompt,
        "completion": completion,
        "score": score,
        "reasoning": answer
    })

    df_results = pd.DataFrame(results)
    df_results.to_json(gen_score_path, lines=True, orient='records')

