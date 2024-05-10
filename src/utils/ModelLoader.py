from typing import Tuple, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    load_peft_weights,
)

from src.utils.logging.logging_config import setup_logging
import logging


class ModelLoader:
    """
    Class to load the model and tokenizer.

    Methods:
        get_bnb_config: Loads the BitsAndBytesConfig.
        load_tokenizer: Loads the Tokenizer.
        load_model: Loads the Model.
        create_peft_config: Creates the PEFT Config.
        get_model_and_config: Returns the Model and Config.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        adapter: bool = False,
        adapter_path: Union[str, None] = None,
    ):
        """
        Initialize the ModelLoader with the given configuration.

        Args:
            config: The configuration dictionary.
            adapter: Whether to use an adapter.
            adapter_path: The path to the adapter, if one is being used.
        """
        self.model_name = config["model_name"]
        self.tokenizer_name = config["tokenizer_name"]
        self.peft_config = config["peft_config"]
        self.adapter = adapter
        self.adapter_path = adapter_path
        setup_logging()
        self.logger = logging.getLogger()
        self.bnb_config = self.get_bnb_config()
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.lora_config = self.create_peft_config()

    def get_bnb_config(self) -> BitsAndBytesConfig:
        """
        Load the BitsAndBytesConfig.

        Returns:
            The loaded BitsAndBytesConfig.
        """
        self.logger.info("Loading BitsAndBytesConfig")
        try:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception as e:
            self.logger.error(f"Error loading BitsAndBytesConfig: {e}")
            raise

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the Tokenizer.

        Returns:
            The loaded Tokenizer.
        """
        self.logger.info("Loading Tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            return tokenizer
        except Exception as e:
            self.logger.error(f"Error loading Tokenizer: {e}")
            raise

    def load_model(self) -> AutoModelForCausalLM:
        """
        Load the Model.

        Returns:
            The loaded Model.
        """
        self.logger.info("Loading Model")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=self.bnb_config
            )
            if self.adapter:
                self.logger.info("Loading Lora Weights")
                lora_weights = load_peft_weights(self.adapter_path)
                _ = model.load_state_dict(lora_weights, strict=False)
            model.config.pretraining_tp = 1
            return model
        except Exception as e:
            self.logger.error(f"Error loading Model: {e}")
            raise

    def create_peft_config(self) -> LoraConfig:
        """
        Create the PEFT Config.

        Returns:
            The created PEFT Config.
        """
        self.logger.info("Creating PEFT Config")
        try:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                lora_dropout=self.peft_config["lora_dropout"],
                lora_alpha=self.peft_config["lora_alpha"],
                r=self.peft_config["lora_r"],
                bias="none",
                target_modules=self.peft_config["target_modules"],
            )
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            return peft_config
        except Exception as e:
            self.logger.error(f"Error creating PEFT Config: {e}")
            raise

    def get_model_and_config(self) -> Tuple[AutoModelForCausalLM, LoraConfig]:
        """
        Return the Model and Config.

        Returns:
            A tuple containing the Model and Config.
        """
        self.logger.info("Returning Model and Config")
        try:
            return self.model, self.lora_config
        except Exception as e:
            self.logger.error(f"Error returning Model and Config: {e}")
            raise
