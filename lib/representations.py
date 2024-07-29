import logging
from enum import Enum
from typing import Tuple

from transformers import AutoModel, AutoProcessor, Blip2ForConditionalGeneration

from .datasets_qa.easyvqa import EasyVQADataset
from .types import CustomDataset

logger = logging.getLogger(__name__)


class DatasetTypes(str, Enum):
    EASY_VQA = "easy-vqa"


class ModelTypes(str, Enum):
    BLIP2 = "blip2"


class HFRepos(str, Enum):
    BLIP2_OPT = "salesforce/blip2-opt-2.7b"


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
    def get_models(model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        # Get corresponding datatbase class
        names_to_repo_ids = {ModelTypes.BLIP2: HFRepos.BLIP2_OPT}

        if model_name not in names_to_repo_ids:
            raise ValueError(f"Invalid model type: {model_name}")

        model_classes = {ModelTypes.BLIP2: Blip2ForConditionalGeneration}

        repo_id = names_to_repo_ids[model_name].value
        processor = AutoProcessor.from_pretrained(repo_id)
        model = model_classes[model_name].from_pretrained(repo_id)
        return model, processor
