import logging
from enum import Enum
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
)

from .datasets_qa.easyvqa import EasyVQADataset
from .types import CustomDataset

logger = logging.getLogger(__name__)


class DatasetTypes(str, Enum):
    EASY_VQA = "easy-vqa"


class ModelTypes(str, Enum):
    BLIP2 = "blip2"


class HFRepos(str, Enum):
    BLIP2_OPT = "Salesforce/blip2-opt-2.7b"


# Mapping from model types to repo IDs
MODEL_REPO_MAPPING = {ModelTypes.BLIP2: HFRepos.BLIP2_OPT.value}

# Mapping from model types to model classes
MODEL_CLASS_MAPPING = {ModelTypes.BLIP2: Blip2ForConditionalGeneration}


class DatasetFactory:
    @staticmethod
    def create_dataset(
        dataset_type: DatasetTypes, train_args: dict, test_args: dict
    ) -> Tuple[CustomDataset, CustomDataset]:
        # Get corresponding datatbase class
        dataset_rep_to_generator_type = {DatasetTypes.EASY_VQA: EasyVQADataset}

        if dataset_type not in dataset_rep_to_generator_type:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        generator = dataset_rep_to_generator_type[dataset_type]

        logger.info(f"Initializing dataset generator {generator.__name__}")
        train_ds = generator(**train_args)
        val_ds = generator(**test_args)
        return train_ds, val_ds


class ModelFactory:
    @staticmethod
    def create_bnb_config() -> BitsAndBytesConfig:
        """Create a BitsAndBytesConfig for quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    @staticmethod
    def create_lora_config() -> LoraConfig:
        """Create a LoraConfig for PEFT."""
        return LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )

    def get_models(
        model_name: ModelTypes, apply_qlora: bool = True
    ) -> Tuple[AutoModel, AutoProcessor]:
        """
        Get the model and processor for a given model type.

        Args:
            model_name (ModelTypes): The type of model to retrieve.
            apply_qlora (bool): Whether to apply QLoRA quantization.

        Returns:
            Tuple[AutoModel, AutoProcessor]: The model and processor.

        Raises:
            ValueError: If the model type is invalid.
        """
        if model_name not in MODEL_REPO_MAPPING:
            raise ValueError(f"Invalid model type: {model_name}")

        repo_id = MODEL_REPO_MAPPING[model_name]
        model_class = MODEL_CLASS_MAPPING[model_name]

        # Load both the processor and model
        processor = AutoProcessor.from_pretrained(repo_id)

        if apply_qlora:
            bnb_config = ModelFactory.create_bnb_config()
            model = model_class.from_pretrained(
                repo_id, torch_dtype=torch.float16, quantization_config=bnb_config
            )

            lora_config = ModelFactory.create_lora_config()
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
        else:
            model = model_class.from_pretrained(repo_id, torch_dtype=torch.float16)

        return model, processor
