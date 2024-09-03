import logging
import random
from collections import defaultdict

from datasets import Dataset
from easy_vqa import (
    get_answers,
    get_test_image_paths,
    get_test_questions,
    get_train_image_paths,
    get_train_questions,
)
from PIL import Image

from ..dataset_base import DatabaseBase
from ..types import DatasetTypes, Suffix, VQAParameters
from ..utils import EXPERIMENT, parse_split_slicer

logger = logging.getLogger(__name__)


class EasyVQADatasetBase(DatabaseBase):
    raw_dataset: Dataset = None
    _dataset: Dataset = None

    def __init__(self, params: VQAParameters):
        super().__init__(DatasetTypes.EASY_VQA, params)

    def _get_answers(self):
        return get_answers()

    @classmethod
    def get_answer(cls):
        return get_answers()

    def initialize_filtered_dataset(self):
        """Method to initialize the dataset."""

        if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Val):
            questions = get_train_questions()
            images = get_train_image_paths()
        elif self.split.startswith(Suffix.Test):
            questions = get_test_questions()
            images = get_test_image_paths()

        data_dict = {
            "question": questions[0],
            "answer": questions[1],
            "image_id": questions[2],
            "image_path": [images[image_id] for image_id in questions[2]],
            "image": [Image.open(images[image_id]).copy() for image_id in questions[2]],
        }

        # Needs a shuffle, otherwise the stratification doesn't work since after
        # filtering there will be too few entries from some classes.
        raw_dataset = Dataset.from_dict(data_dict)

        # Now filter the dataset based on the number of items requested
        split, start, end = parse_split_slicer(self.split)

        if start is not None or end is not None:
            assert split in [choice for choice in Suffix]
            ds = raw_dataset.map(lambda example: {"stratify_column": example["answer"]})
            start = 0 if start is None else start
            end = len(ds) if end is None else end

            if split == Suffix.Val or split == Suffix.Test:
                size = end - start
            else:
                size = len(ds) - (end - start)

            ds = ds.class_encode_column("stratify_column").train_test_split(
                test_size=size,
                stratify_by_column="stratify_column",
                seed=EXPERIMENT.get_seed(),
            )
            raw_dataset = ds[split if split == Suffix.Train else Suffix.Test]

            assert len(raw_dataset) == end - start
        elif self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Val):
            ds = raw_dataset.map(lambda example: {"stratify_column": example["answer"]})
            ds = ds.class_encode_column("stratify_column").train_test_split(
                test_size=0.2,
                stratify_by_column="stratify_column",
                seed=EXPERIMENT.get_seed(),
                shuffle=True,
            )
            raw_dataset = ds[split if split == Suffix.Train else Suffix.Test]

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset

    def initialize_raw(self):
        """Method to initialize the dataset."""

        # Combine train and test datasets
        train_questions = get_train_questions()
        train_images = get_train_image_paths()
        test_questions = get_test_questions()
        test_images = get_test_image_paths()

        # Merge the questions and images
        questions = [
            train_questions[0] + test_questions[0],
            train_questions[1] + test_questions[1],
            train_questions[2] + test_questions[2],
        ]
        images = {**train_images, **test_images}

        dict = {
            "question": questions[0],
            "answer": questions[1],
            "image_id": questions[2],
            "image_path": [images[image_id] for image_id in questions[2]],
            "image": [Image.open(images[image_id]).copy() for image_id in questions[2]],
        }

        raw_dataset = Dataset.from_dict(dict)

        logger.info(f"Read combined dataset, length: {len(raw_dataset)}")
        return raw_dataset

    def get_padding_max_length(self):
        return 25

    def initialize_proportional_raw(self):
        """Method to initialize the dataset with a proportional and balanced split."""
        raw_dataset = self.initialize_raw()

        def stratified_split(examples):
            # Group examples by answer
            grouped = defaultdict(list)
            for i, answer in enumerate(examples["answer"]):
                grouped[answer].append(i)

            train_indices, val_indices = [], []
            for answer, indices in grouped.items():
                total_samples = len(indices)
                if total_samples < 125:
                    # All samples go to training
                    train_indices.extend(indices)
                elif total_samples <= 3125:
                    # 80-20 split for train/val
                    split_point = int(total_samples * 0.8)
                    train_indices.extend(indices[:split_point])
                    val_indices.extend(indices[split_point:])
                else:
                    # Cap at 2500 for training and 625 for validation
                    train_samples = random.sample(indices, 1700)
                    remaining = [i for i in indices if i not in train_samples]
                    val_samples = random.sample(remaining, 400)
                    train_indices.extend(train_samples)
                    val_indices.extend(val_samples)

            return train_indices, val_indices

        # Apply the custom splitting function
        train_indices, val_indices = stratified_split(raw_dataset)

        indices = (
            train_indices
            if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Test)
            else val_indices
        )
        raw_dataset = raw_dataset.select(indices)

        # New code to further split the training dataset into train/test
        if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Test):
            raw_dataset = raw_dataset.map(
                lambda example: {"stratify_column": example["answer"][0]}, batched=False
            )

            train_dataset, test_dataset = (
                raw_dataset.class_encode_column("stratify_column")
                .train_test_split(
                    test_size=0.2,
                    stratify_by_column="stratify_column",
                    seed=EXPERIMENT.get_seed(),
                    shuffle=True
                )
                .values()
            )
            result = train_dataset if self.split == Suffix.Train else test_dataset
            
            logger.info(f"Read {self.split} dataset, length: {len(result)}")
            return result

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset
