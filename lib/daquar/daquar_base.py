import logging
import os
from abc import abstractmethod
from collections import Counter
from itertools import chain
from functools import lru_cache

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image
from skmultilearn.model_selection import iterative_train_test_split

from ..dataset_base import DatabaseBase
from ..types import DatasetPath, DatasetTypes, Suffix, VQAParameters
from ..utils import EXPERIMENT, parse_split_slicer
from tqdm import tqdm
logger = logging.getLogger(__name__)


class DaquarDatasetBase(DatabaseBase):
    # NOTE Total number of classes = 582. After filtering: 53
    train_dataset = None
    val_dataset = None
    answer_space = None

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
        dataset = load_dataset(
            "csv",
            data_files={"train": os.path.join(DatasetPath.DAQUAR, "data_train.csv")},
        )
        return dataset

    def get_test_questions(self):
        dataset = load_dataset(
            "csv", data_files={"val": os.path.join(DatasetPath.DAQUAR, "data_val.csv")}
        )
        return dataset

    def parse_dataset(self, dataset):
        images = []
        for item in dataset:
            img = Image.open(f"{DatasetPath.DAQUAR}/images/{item['image_id']}.png")
            images.append(img)
            img.close()

        parsed = {
            "question": dataset["question"],
            "answer": [
                answer.replace(" ", "").split(",") for answer in dataset["answer"]
            ],
            "image_id": dataset["image_id"],
            "image": images,
        }
        return Dataset.from_dict(parsed)

    def initialize_stratified_raw_old(self):
        """Method to initialize the dataset."""

        dataset_train = self.get_train_questions()["train"]
        dataset_val = self.get_test_questions()["val"]

        dataset_train = self.parse_dataset(dataset_train)
        dataset_val = self.parse_dataset(dataset_val)

        # Using combined items, otherwise the dataset is unbalanced.
        combined = concatenate_datasets([dataset_train, dataset_val])

        # Used for debugging and testing against a smaller dataset
        split, start, end = parse_split_slicer(self.split)

        if start is not None or end is not None:
            start = 0 if start is None else start
            end = len(combined) if end is None else end

            if split == Suffix.Val or split == Suffix.Test:
                size = end - start
            else:
                size = len(combined) - (end - start)

        # default train/test ratio
        size = 0.2
        if self.keep_infrequent:
            filtered = combined.train_test_split(
                test_size=size,
                seed=EXPERIMENT.get_seed(),
            )
            result = filtered[split if split == Suffix.Train else Suffix.Test]
        else:
            # Now create the stratified column using the answer as the key
            combined = combined.map(
                lambda example: {"stratify_column": example["answer"][0]}, batched=False
            )

            # Now prepare the dataset for the stratified split
            answer_counts = Counter(combined["stratify_column"])
            valid_classes = {
                key
                for key, count in answer_counts.items()
                if count >= self.min_class_size
            }

            filtered = combined.filter(
                lambda example: example["stratify_column"] in valid_classes
            )

            # Set the answer space
            self.answer_space = list(
                set([label for sublist in filtered["answer"] for label in sublist])
            )

            filtered = filtered.class_encode_column("stratify_column").train_test_split(
                test_size=size,
                stratify_by_column="stratify_column",
                seed=EXPERIMENT.get_seed(),
            )

            result = filtered[split if split == Suffix.Train else Suffix.Test]
            logger.info(f"Read {self.split} dataset, length: {len(result)}")

        return result

    def combine_datasets(self):
        # Combine train and validation datasets
        train_dataset = self.get_train_questions()["train"]
        val_dataset = self.get_test_questions()["val"]

        # Concatenate datasets
        combined_dataset = concatenate_datasets([train_dataset, val_dataset])

        images = []
        for item in combined_dataset["image_id"]:
            img = Image.open(f"{DatasetPath.DAQUAR}/images/{item}.png")
            images.append(img)
            img.close()

        data_dict = {
            "question": combined_dataset["question"],
            "answer": [
                answer.replace(" ", "").split(",")
                for answer in combined_dataset["answer"]
            ],
            "image_id": combined_dataset["image_id"],
            "image": images,
        }

        raw_dataset = Dataset.from_dict(data_dict)
        return raw_dataset

    def initialize_filtered_dataset(self):
        """Method to initialize the dataset with proper multi-label or multi-class handling."""

        raw_dataset = self.combine_datasets()

        # Count all labels across the entire dataset
        all_labels = list(chain.from_iterable(raw_dataset["answer"]))
        label_counts = Counter(all_labels)

        # Identify valid labels
        valid_labels = {
            label
            for label, count in label_counts.items()
            if count >= self.min_class_size
        }

        # Filter the dataset to keep only items with at least one valid label
        def filter_valid_labels(example):
            valid_answers = [
                answer for answer in example["answer"] if answer in valid_labels
            ]
            return len(valid_answers) > 0

        filtered_dataset = raw_dataset.filter(filter_valid_labels)

        # Update the answers based on the classification type
        def update_answers(example):
            valid_answers = [
                answer for answer in example["answer"] if answer in valid_labels
            ]
            if self.multi_class_classifier:
                # For multi-class, select the first valid answer
                example["answer"] = [valid_answers[0]] if valid_answers else []
            else:
                # For multi-label, keep all valid answers
                example["answer"] = valid_answers
            return example

        final_dataset = filtered_dataset.map(update_answers)

        logger.info(f"Original dataset size: {len(raw_dataset)}")
        logger.info(f"Filtered dataset size: {len(final_dataset)}")
        logger.info(f"Number of unique labels after filtering: {len(valid_labels)}")
        logger.info(f"Classification type: {'Multi-class' if self.multi_class_classifier else 'Multi-label'}")

        return final_dataset

    def initialize_raw(self):
        return self.combine_datasets()

    @abstractmethod
    def _prepare_for_training(self, item: dict):
        pass

    def get_padding_max_length(self):
        return 50

    def initialize_proportional_raw(self):
        """Compute and cache the stratified split."""
        
        # Used for debugging and testing against a smaller dataset
        split, _, _ = parse_split_slicer(self.split)
        self.answer_space = DaquarDatasetBase.answer_space
        
        if DaquarDatasetBase.train_dataset is not None and split.startswith(Suffix.Train):
            return DaquarDatasetBase.train_dataset
        
        if DaquarDatasetBase.val_dataset is not None and split.startswith(Suffix.Val):
            return DaquarDatasetBase.val_dataset
        
        # We need to create an instance to call instance methods
        raw_dataset = self.initialize_filtered_dataset()

        # Get all unique labels
        all_labels = sorted(set(label for example in raw_dataset['answer'] for label in example))

        # Create a binary matrix representation of the labels with a progress bar
        total_examples = len(raw_dataset)
        y = []
        with tqdm(total=total_examples, desc="Creating label matrix", unit="example") as pbar:
            for example in raw_dataset:
                if self.multi_class_classifier:
                    # For multi-class, use one-hot encoding
                    y.append([1 if label == example['answer'][0] else 0 for label in all_labels])
                else:
                    # For multi-label, use multi-hot encoding
                    y.append([1 if label in example['answer'] else 0 for label in all_labels])
                pbar.update(1)
                
        y = np.array(y)

        # Create a dummy feature matrix (just using indices)
        X = np.arange(total_examples).reshape(-1, 1)

        # Perform iterative stratification
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.2)

        # Get the indices for train and test sets
        train_indices = X_train.flatten()
        test_indices = X_test.flatten()

        # Split the dataset
        train_dataset = raw_dataset.select(train_indices)
        test_dataset = raw_dataset.select(test_indices)

        
        result = train_dataset if split == Suffix.Train else test_dataset

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info(f"Selected {split} dataset size: {len(result)}")

        DaquarDatasetBase.train_dataset = train_dataset
        DaquarDatasetBase.val_dataset = test_dataset
        DaquarDatasetBase.answer_space = all_labels
        self.answer_space = all_labels

        return result


