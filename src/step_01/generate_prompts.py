import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.utils.model import load_model
from src.utils.read_write_jsonl import read_jsonl_file, write_jsonl_file
from src.utils.prompts import prompt_01

from transformers import TextStreamer
import re
import uuid
import pandas as pd

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, '..', 'data/')
ift_path = os.path.normpath(data_path) + '/ift.jsonl'
gen_prompts_path = os.path.normpath(data_path) + '/gen_prompts.jsonl'

model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.2")

ift_df = read_jsonl_file(ift_path)
uniq_prompts = set([])
new_prompts = []
new_prompts_to_generate = 1000

def get_random_prompts(data, num_prompts=8):
    return data.sample(n=num_prompts)["prompt"].tolist()

def generate_prompt(examples):
    global prompt_01
    for _, item in enumerate(examples):
        prompt_01 += f"<task>|{item}</task>\n"
    
    return prompt_01
    
def do_sample(model, tokenizer, task_prompts):
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
        streamer=streamer,)
    
    
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answer = decoded[0]
    return answer

def extract_prompt(answer):
    prompts = []
    extracted_prompts = re.findall(r'<task>\|(.*?)</task>', answer, re.DOTALL)
    for prompt in extracted_prompts:
        prompts.append(prompt)
    return prompts


while True:
    if len(uniq_prompts) >= new_prompts_to_generate:
        break

    task_prompts = get_random_prompts(ift_df)
    answer = do_sample(model, tokenizer, task_prompts)
    prompts = extract_prompt(answer)
    for prompt in prompts:
        if prompt not in uniq_prompts:
            uniq_prompts.add(prompt)
            prompt_id = str(uuid.uuid4())
            new_prompts.append({"id": prompt_id, "prompt": prompt, "source": "generated"})

    new_prompts_df = pd.DataFrame(new_prompts)
    write_jsonl_file(new_prompts_df, gen_prompts_path)
