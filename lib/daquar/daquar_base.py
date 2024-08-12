import logging
import os
from abc import abstractmethod

from datasets import Dataset, load_dataset
from PIL import Image
from collections import Counter
from ..types import DatasetPath, DatasetTypes, Suffix, VQAParameters
from ..utils import EXPERIMENT, parse_split_slicer
from ..dataset_base import DatabaseBase

logger = logging.getLogger(__name__)

class DaquarDatasetBase(DatabaseBase):
    # NOTE Total number of classes = 582. After filtering: 53
    
    def __init__(self, params: VQAParameters):
        super().__init__(DatasetTypes.DAQUAR, params)

    def _get_answers(self):
        with open(os.path.join(DatasetPath.DAQUAR, "answer_space.txt")) as f:
            return f.read().splitlines()
    
    @classmethod
    def get_answers(cls):
        with open(os.path.join(DatasetPath.DAQUAR, "answer_space.txt")) as f:
            return f.read().splitlines()

    def get_train_questions(self):
        dataset = load_dataset("csv", data_files={"train": os.path.join(DatasetPath.DAQUAR, "data_train.csv")})
        return dataset

    def get_test_questions(self):
        dataset = load_dataset("csv", data_files={"val": os.path.join(DatasetPath.DAQUAR, "data_val.csv")})
        return dataset

    def initialize_stratified_raw(self):
        """Method to initialize the dataset."""

        if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Val):
            dataset = self.get_train_questions()['train']
        elif self.split.startswith(Suffix.Test):
            dataset = self.get_test_questions()['val']

        dict = {
            "question": dataset["question"],
            "answer": [answer.replace(" ", "").split(",") for answer in dataset['answer']],
            "image_id": dataset["image_id"],
            "image": [Image.open(f"{DatasetPath.DAQUAR}/images/{item['image_id']}.png") for item in dataset],
        }

        # Needs a shuffle, otherwise the stratification doesn't work since after
        # filtering there will be too few entries from some classes.
        raw_dataset = Dataset.from_dict(dict)

        # Now filter the dataset based on the number of items requested
        split, start, end = parse_split_slicer(self.split)

        if start is not None or end is not None:
            assert split in [choice for choice in Suffix]
            ds = raw_dataset.map(lambda example: {"stratify_column": example["answer"][0]}, batched=False)

            answer_counts = Counter(ds['stratify_column'])
            valid_classes = {key for key, count in answer_counts.items() if count >= self.min_class_size}
            
            filtered = ds.filter(lambda example: example["stratify_column"] in valid_classes)
        
            start = 0 if start is None else start
            end = len(filtered) if end is None else end

            if split == Suffix.Val or split == Suffix.Test:
                size = end - start
            else:
                size = len(filtered) - (end - start)

            filtered = filtered.class_encode_column("stratify_column").train_test_split(
                test_size=size,
                stratify_by_column="stratify_column",
                seed=EXPERIMENT.get_seed(),
            )
            
            raw_dataset = filtered[split if split == Suffix.Train else Suffix.Test]

            assert len(raw_dataset) == end - start

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        
        return raw_dataset

    def initialize_raw(self):
        raise NotImplementedError()

    @abstractmethod
    def _prepare_for_training(self, item: dict):
        pass

    def get_padding_max_length(self):
        return 40