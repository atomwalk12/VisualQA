import logging
from enum import Enum
from typing import Tuple

from .datasets_qa.easyvqa import EasyVQADataset
from .types import CustomDataset

logger = logging.getLogger(__name__)


class DatasetTypes(str, Enum):
    EASY_VQA = "easy-vqa"


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
