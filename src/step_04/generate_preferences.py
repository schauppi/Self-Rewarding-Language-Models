import os
import pandas as pd
import json

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, '..', 'data/')
gen_score_path = os.path.normpath(data_path) + '/gen_scores.jsonl'
preference_path = os.path.normpath(data_path) + '/preference_pairs.jsonl'

prompts = {}

with open(gen_score_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        prompt_id = data['prompt_id']
        if prompt_id not in prompts:
            prompts[prompt_id] = []
        prompts[prompt_id].append(data)

pairs = []

for prompt_id, prompt_group in prompts.items():
    best_prompt = None
    worst_prompt = None
    max_score = float('-inf')  
    min_score = float('inf')  

    for prompt in prompt_group:
        if prompt['score'] > max_score:
            max_score = prompt['score']
            best_prompt = prompt
        if prompt['score'] < min_score:
            min_score = prompt['score']
            worst_prompt = prompt

    if best_prompt and worst_prompt:
        pairs.append({
            "prompt_id": best_prompt['prompt_id'],
            "prompt": best_prompt['prompt'],
            "chosen": best_prompt['completion'],
            "rejected": worst_prompt['completion'],
            "score_chosen": best_prompt['score'],
            "score_rejected": worst_prompt['score']
        })

df_pairs = pd.DataFrame(pairs)
df_pairs.to_json(preference_path, lines=True, orient='records')

