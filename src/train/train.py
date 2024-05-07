from pathlib import Path
import os
import logging
from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader
from src.utils.create_sft_dataset import create_sft_dataset
from src.utils.SFTTrainer import TrainerSFT
from src.utils.DPOTrainer import TrainerDPO
from src.utils.generate_prompts import generate_new_prompts
from src.utils.generate_responses import generate_responses
from src.utils.generate_scores import generate_scores
from src.utils.generate_preferences import generate_preferences
from src.utils.generate_dpo_dataset import generate_dpo_dataset

setup_logging()
logger = logging.getLogger()

config_loader = ConfigLoader()
config = config_loader.config

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
os.environ["WANDB_PROJECT"] = config["wandb_project"]

###STEP1###
"""loader = ModelLoader(config)
model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config

dataset = create_sft_dataset(
    dataset_path=(config["data_path"] / config["ift_dataset"]), tokenizer=tokenizer
)
sft_trainer = TrainerSFT(config=config)
sft_adapter_path = sft_trainer.output_dir
sft_trainer = sft_trainer.train(
    model=model, tokenizer=tokenizer, lora_config=lora_config, dataset=dataset
)"""
###STEP1###

###LOOP###

iteration = 0

"""
###STEP2###
sft_adapter_path = "/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-04-28_16-53-58/sft"

loader = ModelLoader(config, adapter=True, adapter_path=sft_adapter_path)
model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config
prompts_path = generate_new_prompts(model, tokenizer, config, iteration)
###STEP2###

###STEP3###
responses_path = generate_responses(model, tokenizer, config, iteration, prompts_path)
###STEP3###

###STEP4###
scores_path = generate_scores(model, tokenizer, config, iteration, responses_path)
###STEP4###

###STEP5###
preferences_path = generate_preferences(config, iteration, scores_path)
###STEP5###
"""

###STEP6###
preferences_path = "/home/ds/workspace/Self-Rewarding-Language-Models/src/data/0/preference_pairs.jsonl"
sft_adapter_path = "/home/ds/workspace/Self-Rewarding-Language-Models/results/results_2024-04-28_16-53-58/sft"

loader = ModelLoader(config, adapter=True, adapter_path=sft_adapter_path)
model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config

dpo_dataset = generate_dpo_dataset(preferences_path, tokenizer)

dpo_trainer = TrainerDPO(config=config)
dpo_adapter_path = dpo_trainer.output_dir
dpo_trainer = dpo_trainer.train(
    model=model, tokenizer=tokenizer, lora_config=lora_config, dataset=dpo_dataset
)

###STEP6###


"""
for iteration in range(num_iterations):
    # Step 1: Self-Instruction Creation
    new_prompts = generate_new_prompts(model, IFT_data, num_samples)
    candidate_responses = {}

    for prompt in new_prompts:
        # Generate multiple candidate responses per new prompt
        candidate_responses[prompt] = generate_candidate_responses(model, prompt, num_responses)
        
        # Self-evaluate responses to assign rewards
        scores = self_evaluate_responses(model, candidate_responses[prompt])
    
    # Step 2: Build preference pairs from the self-evaluated responses
    preference_pairs = build_preference_pairs(candidate_responses, scores)

    # Step 3: Instruction Following Training using Direct Preference Optimization (DPO)
    model = train_with_DPO(model, preference_pairs)

    # Optionally, update the IFT and EFT datasets with new generated data
    update_datasets(IFT_data, EFT_data, candidate_responses, scores)

# Save the final model
model.save('self_rewarding_model_final')
"""