import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd

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
            max_new_tokens=256,
            streamer=streamer,
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer
    
def extract_completion(answer):
    pattern = f"[/INST]"
    parts = answer.split(pattern)
    if len(parts) > 1:
        return parts[-1]
    else:
        return ""
    
def trim_completion(completion):
    if "\n" in completion:
        last_newline = completion.rfind("\n")
        completion = completion[:last_newline]
        return completion.strip()
    else:
        return completion

base_path = os.path.dirname(__file__)

data_path = os.path.join(base_path, '..', 'data/')
gen_prompts_path = os.path.normpath(data_path) + '/gen_prompts.jsonl'

gen_response_path = os.path.normpath(data_path) + '/gen_responses.jsonl'

model_dir_path = os.path.join(base_path, '..', '..', 'results/')
model_dir_path = os.path.normpath(model_dir_path) + '/results_2024-04-18_12-44-14/checkpoint-150'


if not os.path.exists(model_dir_path):
    raise ValueError(f"The specified model directory does not exist: {model_dir_path}")

if not os.path.exists(data_path):
    raise ValueError(f"The specified data path does not exist: {data_path}")


df_prompts = pd.read_json(gen_prompts_path, lines=True)
df_prompts = df_prompts.sample(frac=1).reset_index(drop=True)


model, tokenizer = load_fine_tuned(model_dir_path)
model.eval()

completions = []

for index, row in df_prompts.iterrows():
    print(f"Processing prompt {index + 1} of {len(df_prompts)}")

    prompt = row['prompt']
    prompt_id = row['id']

    for completion_sample in range(4):
        print("----------------")
        print(f"Processing completion {completion_sample + 1} of 4")

        answer = do_sample(model, tokenizer, prompt)
        completion = extract_completion(answer)
        trimmed_completion = trim_completion(completion)

        completions.append({"prompt_id": prompt_id, "prompt": prompt , "completion": trimmed_completion})

        df_completions = pd.DataFrame(completions)

        df_completions.to_json(gen_response_path, orient='records', lines=True)