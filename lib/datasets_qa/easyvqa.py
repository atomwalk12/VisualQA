import logging
import pickle

import pandas as pd
from datasets import Dataset
from easy_vqa import (
    get_answers,
    get_test_image_paths,
    get_test_questions,
    get_train_image_paths,
    get_train_questions,
)
from PIL import Image

from ..types import CustomDataset
from ..utils import get_complete_path, parse_split_slicer

logger = logging.getLogger(__name__)


class EasyVQADataset(CustomDataset):
    raw_dataset: Dataset = None
    _dataset: Dataset = None
    ready_for_training: bool = False

    def __init__(self, split: str, classify=False, load_raw=True):
        super().__init__()
        self.classify: bool = classify

        # can be train or val
        self.split: str = split

        # Store the answer space for commodity
        self.answer_space = get_answers()

        if load_raw:
            self.raw_dataset = self.initialize_raw()

    def _initialize_raw(self):
        """Method to initialize the dataset."""

        if self.split.startswith("train") or self.split.startswith("val"):
            questions = get_train_questions()
            images = get_train_image_paths()
        elif self.split.startswith("test"):
            questions = get_test_questions()
            images = get_test_image_paths()

        dict = {
            "question": questions[0],
            "answer": questions[1],
            "image_id": questions[2],
            "image_path": [images[image_id] for image_id in questions[2]],
            "image": [Image.open(images[image_id]) for image_id in questions[2]],
        }

        raw_dataset = Dataset.from_dict(dict)

        # Now filter the dataset based on the number of items requested
        split, start, end = parse_split_slicer(self.split)
        if start is not None or end is not None:
            raw_dataset = raw_dataset.select(range(start or 0, end or len(raw_dataset)))

        if start is not None or end is not None:
            if split in ["train", "val"]:
                ds = raw_dataset.map(lambda example: {"stratify_column": example["answer"]})
                ds = ds.class_encode_column("stratify_column").train_test_split(
                    test_size=0.1, stratify_by_column="stratify_column", seed=1220
                )
                raw_dataset = ds[split if split == "train" else "test"]

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset


    def initialize_raw(self):
        """Method to initialize the dataset."""

        if self.split.startswith("train"):
            questions = get_train_questions()
            images = get_train_image_paths()
        elif self.split.startswith("val"):
            questions = get_test_questions()
            images = get_test_image_paths()

        dict = {
            "question": questions[0],
            "answer": questions[1],
            "image_id": questions[2],
            "image_path": [images[image_id] for image_id in questions[2]],
            "image": [Image.open(images[image_id]) for image_id in questions[2]],
        }

        raw_dataset = Dataset.from_dict(dict)

        _, start, end = parse_split_slicer(self.split)
        if start is not None or end is not None:
            raw_dataset = raw_dataset.select(range(start or 0, end or len(raw_dataset)))

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset

    @property
    def dataset(self):
        if self._dataset is None:
            raise Exception(
                "Please call transform() before accessing the training dataset."
            )
        return self._dataset

    def __len__(self) -> int:
        if self.ready_for_training:
            return len(self._dataset)
        else:
            return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        if self.ready_for_training:
            return self.dataset[index]
        else:
            assert (
                self.raw_dataset is not None
            ), "Please initialize the raw dataset (pass load_raw=True in the constructor)"
            return self.raw_dataset[index]

    def initialize_for_training(self):
        """Prepare the dataset for training"""
        # if self.raw_dataset is None:
        #    self.initialize_raw()

        logger.info("Preparing data for training")
        columns_to_remove = self.raw_dataset.column_names
        columns_to_remove.remove("image")
        self._dataset = self.raw_dataset.map(
            lambda item: self._prepared_for_training(item),
            remove_columns=columns_to_remove,
        )
        self.ready_for_training = True

    def _prepared_for_training(self, item: dict):
        """
        Prepare a training example. When classify is true the retrieved label
        represents a number from the answer space instead of simple text.
        """
        if self.classify:
            return self._classify(item)
        else:
            return self._autoregression(item)

    def _classify(self, item):
        raise NotImplementedError()

    def _autoregression(self, item: dict):
        """Generate training example based on textual labels.

        Args:
            item (dict): Raw item from self.raw_dataset

        Returns:
            str: An element to be used during training
        """
        input_text = item["question"]
        label = item["answer"]

        if self.split.startswith("train"):
            prompt = f"Question: {input_text} Answer: {label}."
        elif self.split.startswith("val"):
            prompt = f"Question: {input_text} Answer:"
        else:
            raise Exception(f"Flag {self.split} not recognized.")

        return {"prompt": prompt, "label": label}

    def save(self, out: str):
        """Utility used for saving the dataset at the given output path.

        Args:
            out (str): If out is a directory, then the split name is used as
            the name of the file.
        """

        if not self._prepared_for_training:
            raise Exception(f"First, call {self._prepared_for_training.__name__}.")

        complete_path = get_complete_path(out, opt_name=self.split)

        logger.info("Saving dataset to %s", complete_path)

        try:
            with open(complete_path, "wb") as file:
                pickle.dump(self, file)

        except TypeError:
            logger.error(f"Was not able to save pickle file at {out}")
            raise

    def load(self, out: str):
        """Utility to load the pickle from a given path."""
        complete_path = get_complete_path(out, opt_name=self.split)

        try:
            loaded_data = pd.read_pickle(complete_path)

            for key, value in vars(loaded_data).items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Attribute {key} not found in class instance.")

            logger.info(f"Loaded {len(self._dataset)
                                  } items from {complete_path}")
            return self
        except FileNotFoundError:
            logger.error(f"Was not able to load pickle file at {
                         complete_path}")
            raise

    def equals(self, other: Dataset):
        dataset = self.dataset.to_pandas()
        return dataset.equals(other.to_pandas())
