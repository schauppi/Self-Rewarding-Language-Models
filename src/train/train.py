from pathlib import Path
import os
import logging
from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader

from datasets import load_dataset

setup_logging()
logger = logging.getLogger()


config_loader = ConfigLoader()
config = config_loader.config

# print the paths
logger.info(f"Data path: {config['data_path']}")
logger.info(f"Model directory path: {config['model_dir_path']}")

os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
logger.info(f"Visible CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

# initial fine tune
model = ModelLoader(config).model

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
