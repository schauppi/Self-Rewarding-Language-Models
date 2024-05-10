from pathlib import Path
import os
import logging
from src.utils.logging.logging_config import setup_logging
from src.utils.ModelLoader import ModelLoader
from src.utils.ConfigLoader import ConfigLoader
from src.utils.create_sft_dataset import create_sft_dataset
from src.utils.Trainer.SFTTrainer import TrainerSFT
from src.utils.Trainer.DPOTrainer import TrainerDPO
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
if config["wandb_enable"] == True:
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
else:
    os.environ["WANDB_MODE"] = "disabled"

logger.info(f"WandB Enabled: {config['wandb_enable']}")


###STEP1###
logger.info("Step 0: Training SFT model")
loader = ModelLoader(config)
model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config

dataset = create_sft_dataset(
    dataset_path=(config["ift_data_path"] / config["ift_dataset"]), tokenizer=tokenizer
)
sft_trainer = TrainerSFT(config=config, iteration=0)
sft_adapter_path = sft_trainer.output_dir
sft_trainer = sft_trainer.train(
    model=model, tokenizer=tokenizer, lora_config=lora_config, dataset=dataset
)
###STEP1###

###LOOP###

for iteration in range(config["iterations"]):
    logger.info(f"Starting iteration {iteration}")

    ###STEP2###
    logger.info(f"Step 1 | iteration {iteration}: Generating new prompts")
    if iteration == 0:
        loader = ModelLoader(config, adapter=True, adapter_path=sft_adapter_path)
    else:
        loader = ModelLoader(config, adapter=True, adapter_path=dpo_adapter_path)
    model, tokenizer, lora_config = loader.model, loader.tokenizer, loader.lora_config
    prompts_path = generate_new_prompts(model, tokenizer, config, iteration)
    ###STEP2###

    ###STEP3###
    logger.info(f"Step 2 | iteration {iteration}: Generating responses")
    responses_path = generate_responses(
        model, tokenizer, config, iteration, prompts_path
    )
    ###STEP3###

    ###STEP4###
    logger.info(f"Step 3 | iteration {iteration}: Generating scores")
    scores_path = generate_scores(model, tokenizer, config, iteration, responses_path)
    ###STEP4###

    ###STEP5###
    logger.info(f"Step 4 | iteration {iteration}: Generating preferences")
    preferences_path = generate_preferences(config, iteration, scores_path)
    ###STEP5###

    ###STEP6###
    logger.info(f"Step 5 | iteration {iteration}: Training DPO model")
    dpo_dataset = generate_dpo_dataset(preferences_path, tokenizer)

    dpo_trainer = TrainerDPO(config=config, iteration=iteration)
    dpo_adapter_path = dpo_trainer.output_dir
    dpo_trainer = dpo_trainer.train(
        model=model, tokenizer=tokenizer, lora_config=lora_config, dataset=dpo_dataset
    )
    ###STEP6###
