import logging
import os
import random
from abc import abstractmethod
from collections import Counter, defaultdict
from itertools import chain

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from PIL import Image
from skmultilearn.model_selection import iterative_train_test_split
from torchvision import transforms
from tqdm import tqdm

from ..dataset_base import DatabaseBase
from ..types import DatasetPath, DatasetTypes, Suffix, VQAParameters
from ..utils import EXPERIMENT, parse_split_slicer

logger = logging.getLogger(__name__)


class DaquarDatasetBase(DatabaseBase):
    # NOTE Total number of classes = 582. After filtering: 53
    train_dataset = None
    val_dataset = None
    answer_space = None
    max_examples_per_label = 750
    augmented_dataset = Dataset.from_dict(
        {"question": [], "answer": [], "image_id": [], "image": []}
    )
    # Percentile to determine the cutoff for low-frequency labels
    percentile = 70

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

        # Initialize a dictionary to keep track of examples per label
        label_examples = defaultdict(list)

        # First pass: collect all examples for each label
        for idx, example in enumerate(raw_dataset):
            valid_answers = [
                answer for answer in example["answer"] if answer in valid_labels
            ]
            if valid_answers:
                if self.multi_class_classifier:
                    # Sort valid answers by their frequency in the dataset (least frequent first)
                    sorted_answers = sorted(
                        valid_answers, key=lambda x: label_counts[x]
                    )
                    for label in sorted_answers:
                        if len(label_examples[label]) < self.max_examples_per_label:
                            label_examples[label].append(idx)
                            break  # Only add the example to the least frequent label that hasn't reached the limit
                else:
                    for label in valid_answers:
                        if len(label_examples[label]) < self.max_examples_per_label:
                            label_examples[label].append(idx)

        # Second pass: ensure minimum number of examples per label and maintain distribution
        final_examples = set()
        for label, examples in sorted(label_examples.items(), key=lambda x: len(x[1])):
            if len(examples) >= self.min_class_size:
                # Add examples up to the maximum limit
                final_examples.update(examples[: self.max_examples_per_label])
            else:
                # Remove labels that don't meet the minimum size
                del label_examples[label]

        # Convert to list and shuffle
        final_examples = list(final_examples)
        random.shuffle(final_examples)

        # Select the final examples from the dataset
        filtered_dataset = raw_dataset.select(final_examples)

        # Update the answers based on the classification type
        def update_answers(example):
            valid_answers = [
                answer for answer in example["answer"] if answer in label_examples
            ]
            if self.multi_class_classifier:
                # Choose the least frequent valid answer
                chosen_label = min(valid_answers, key=lambda x: len(label_examples[x]))
                example["answer"] = [chosen_label]
            else:
                example["answer"] = valid_answers
            return example

        final_dataset = filtered_dataset.map(update_answers)

        # Calculate label distribution
        all_labels = [label for example in final_dataset['answer'] for label in example]
        label_counts = Counter(all_labels)
        total_labels = sum(label_counts.values())
        label_distribution = {label: count / total_labels for label, count in label_counts.items()}

        # Identify labels in the lowest 30% of distribution
        threshold = np.percentile(list(label_distribution.values()), self.percentile)
        low_distribution_labels = set(label for label, dist in label_distribution.items() if dist <= threshold)

        def augment_image(image):
            augmentations = [
                transforms.RandomRotation(degrees=5),  # Reduced rotation
                transforms.RandomResizedCrop(size=image.size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),  # Minimal crop
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),  # Subtle color changes
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Subtle blur
            ]
            aug_transform = transforms.Compose(random.sample(augmentations, k=3))
            return aug_transform(image)

        augmented_examples = {
            'question': [],
            'answer': [],
            'image_id': [],
            'image': []
        }

        # Iterate through the original dataset
        for example in tqdm(final_dataset, desc="Augmenting dataset"):
            # Check if any of the example's labels are in the low distribution set
            if any(label in low_distribution_labels for label in example['answer']):
                # Generate two additional augmented images
                for _ in range(2):
                    augmented_examples['question'].append(example['question'])
                    augmented_examples['answer'].append(example['answer'])
                    augmented_examples['image_id'].append(example['image_id'])
                    augmented_examples['image'].append(augment_image(example['image']))
            else:
                # Generate one additional augmented image for other labels
                augmented_examples['question'].append(example['question'])
                augmented_examples['answer'].append(example['answer'])
                augmented_examples['image_id'].append(example['image_id'])
                augmented_examples['image'].append(augment_image(example['image']))

        # Create the augmented dataset
        self.augmented_dataset = Dataset.from_dict(augmented_examples)

        # Combine original and augmented datasets
        combined_dataset = concatenate_datasets([final_dataset, self.augmented_dataset])

        logger.info(f"Original dataset size: {len(raw_dataset)}")
        logger.info(f"Filtered dataset size: {len(final_dataset)}")
        logger.info(f"Augmented dataset size: {len(self.augmented_dataset)}")
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
        logger.info(f"Number of unique labels after filtering: {len(label_examples)}")
        logger.info(
            f"Classification type: {'Multi-class' if self.multi_class_classifier else 'Multi-label'}"
        )

        # Log distribution of examples per label
        label_distribution = {
            label: len(examples) for label, examples in label_examples.items()
        }
        logger.info(f"Label distribution: {label_distribution}")

        return combined_dataset

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

        if DaquarDatasetBase.train_dataset is not None and split.startswith(
            Suffix.Train
        ):
            return DaquarDatasetBase.train_dataset

        if DaquarDatasetBase.val_dataset is not None and split.startswith(Suffix.Val):
            return DaquarDatasetBase.val_dataset

        # We need to create an instance to call instance methods
        raw_dataset = self.initialize_filtered_dataset()

        # Get all unique labels
        all_labels = sorted(
            set(label for example in raw_dataset["answer"] for label in example)
        )

        # Create a binary matrix representation of the labels with a progress bar
        total_examples = len(raw_dataset)
        y = []
        with tqdm(
            total=total_examples, desc="Creating label matrix", unit="example"
        ) as pbar:
            for example in raw_dataset:
                if self.multi_class_classifier:
                    # For multi-class, use one-hot encoding
                    y.append(
                        [
                            1 if label == example["answer"][0] else 0
                            for label in all_labels
                        ]
                    )
                else:
                    # For multi-label, use multi-hot encoding
                    y.append(
                        [1 if label in example["answer"] else 0 for label in all_labels]
                    )
                pbar.update(1)

        y = np.array(y)

        # Create a dummy feature matrix (just using indices)
        X = np.arange(total_examples).reshape(-1, 1)

        # Perform iterative stratification
        X_train, y_train, X_test, y_test = iterative_train_test_split(
            X, y, test_size=0.2
        )

        # Get the indices for train and test sets
        train_indices = X_train.flatten()
        test_indices = X_test.flatten()

        # Split the dataset
        train_dataset = raw_dataset.select(train_indices)
        test_dataset = raw_dataset.select(test_indices)
        
        # Shuffle the train dataset
        train_dataset = train_dataset.shuffle(seed=EXPERIMENT.get_seed())
        test_dataset = test_dataset.shuffle(seed=EXPERIMENT.get_seed())
        
        result = train_dataset if split == Suffix.Train else test_dataset

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info(f"Selected {split} dataset size: {len(result)}")

        DaquarDatasetBase.train_dataset = train_dataset
        DaquarDatasetBase.val_dataset = test_dataset
        DaquarDatasetBase.answer_space = all_labels
        self.answer_space = all_labels

        return result
