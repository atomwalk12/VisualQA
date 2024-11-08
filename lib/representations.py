import logging
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Conv1D,
)

from .types import MODEL_CLASS_MAPPING, MODEL_REPO_MAPPING, PROCESSOR_CLASS_MAPPING, ModelTypes, DatasetTypes

logger = logging.getLogger(__name__)


class ModelFactory:
    def get_specific_layer_names(model):
        # Create a list to store the layer names
        layer_names = []

        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                # model name parsing
                layer_names.append(name)

        return list(set(layer_names))

    @staticmethod
    def bnb_config() -> BitsAndBytesConfig:
        """Create a BitsAndBytesConfig for quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )

    @staticmethod
    def lora_config() -> LoraConfig:
        """Create a LoraConfig for PEFT."""
        return LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )

    def get_models(
        model_name: ModelTypes, apply_lora: bool, lora_config, bnb_config: BitsAndBytesConfig, torch_dtype
    ) -> Tuple[AutoModel, AutoProcessor]:
        """
        Get the model for a given model type.

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

        repo_id: str = MODEL_REPO_MAPPING[model_name]
        model_class = MODEL_CLASS_MAPPING[model_name]

        model = model_class.from_pretrained(repo_id, torch_dtype=torch_dtype, quantization_config=bnb_config)

        if apply_lora:
            model = ModelFactory.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model
    
    def get_processor(model_name):
        repo_id: str = MODEL_REPO_MAPPING[model_name]
        processor_class: Blip2Processor = PROCESSOR_CLASS_MAPPING[model_name]
        
        processor = processor_class.from_pretrained(repo_id)
        return processor
        
    def prepare_model_for_kbit_training(model, use_gradient_checkpointing):
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
