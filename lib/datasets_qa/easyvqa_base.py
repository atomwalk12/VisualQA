import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

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

from ..types import CustomDataset, VQAParameters
from ..utils import ROOT_DATA_DIR, parse_split_slicer

logger = logging.getLogger(__name__)


class EasyVQADatasetBase(CustomDataset, ABC):
    raw_dataset: Dataset = None
    _dataset: Dataset = None
    ready_for_training: bool = False

    def __init__(self, params: VQAParameters):
        super().__init__()

        assert params.processor is not None

        # Store parameters
        self.padding_max_length = params.padding_max_length
        self.processor = params.processor
        self.use_stratified_split = params.use_stratified_split

        # can be train or val
        self.split: str = params.split

        # Store the answer space for commodity
        self.answer_space = get_answers()

        # Create mappings
        self.answers_to_id = {answer: idx for idx, answer in enumerate(self.answer_space)}
        self.id_to_answer = {idx: answer for idx, answer in enumerate(self.answer_space)}

        # Load data
        if params.load_from_disk:
            self.load()
        else:
            if self.use_stratified_split:
                self.raw_dataset = self.initialize_stratified_raw()
            else:
                self.raw_dataset = self.initialize_raw()

            self.initialize_for_training()

        self.is_testing = params.is_testing

    def shuffle(self, seed):
        self._dataset = self._dataset.shuffle(seed)

    def initialize_stratified_raw(self):
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

        # Needs a shuffle, otherwise the stratification doesn't work since after
        # filtering there will be too few entries from some classes.
        raw_dataset = Dataset.from_dict(dict)

        # Now filter the dataset based on the number of items requested
        split, start, end = parse_split_slicer(self.split)

        if start is not None or end is not None:
            assert split in ["train", "val", "test"]
            ds = raw_dataset.map(lambda example: {"stratify_column": example["answer"]})
            start = 0 if start is None else start
            end = len(ds) if end is None else end

            if split == "val" or split == "test":
                size = end - start
            else:
                size = len(ds) - (end - start)

            ds = ds.class_encode_column("stratify_column").train_test_split(
                test_size=size, stratify_by_column="stratify_column", seed=1220
            )
            raw_dataset = ds[split if split == "train" else "test"]

            assert len(raw_dataset) == end - start

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset

    def initialize_raw(self):
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

        split, start, end = parse_split_slicer(self.split)
        if self.split.startswith("train") or self.split.startswith("val"):
            target = "test" if split.startswith("val") else split
            raw_dataset = raw_dataset.train_test_split(test_size=0.25, seed=2024)[target]

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

    def __getitem__(self, idx):
        # Encode the dataset item
        item = self.dataset[idx]

        if self.split.startswith("test"):
            padding = {}
        else:
            padding = {"padding": "max_length", "max_length": self.padding_max_length}

        encoding = self.processor(
            images=item["image"],
            text=item["prompt"],
            return_tensors="pt",
            **padding,
        )

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return item, encoding

    def initialize_for_training(self):
        """Prepare the dataset for training"""

        logger.info("Preparing data for training")
        columns_to_remove = self.raw_dataset.column_names
        columns_to_remove.remove("image")

        # generate the dataset
        self._dataset = self.raw_dataset.map(
            lambda item: self._prepare_for_training(item),
            remove_columns=columns_to_remove,
        )
        self.ready_for_training = True

    def save(self):
        """Utility used for saving the dataset at the given output path.

        Args:
            out (str): If out is a directory, then the split name is used as
            the name of the file.
        """

        if not self._prepare_for_training:
            raise Exception(f"First, call {self._prepare_for_training.__name__}.")

        complete_path = self.get_make_complete_path()

        logger.info("Saving dataset to %s", complete_path)

        try:
            with open(complete_path, "wb") as file:
                pickle.dump(self, file)
                logger.info(f"Saved dataset configuration to {complete_path}")
        except TypeError:
            logger.warning(f"Pickle file not found at at {complete_path}.")
            raise

    def load(self):
        """Utility to load the pickle from a given path."""
        assert not self.ready_for_training
        complete_path = self.get_make_complete_path()

        try:
            loaded_data = pd.read_pickle(complete_path)

            for key, value in vars(loaded_data).items():
                if hasattr(self, key):
                    if (
                        key == "use_stratified_split"
                        and self.use_stratified_split != value
                    ):
                        raise ValueError(
                            f"Mismatch in stratified splitting. "
                            f"Required: {self.use_stratified_split}, "
                            f"Dataset created with: {value}"
                        )
                    setattr(self, key, value)
                else:
                    raise KeyError(f"Attribute {key} not found in class instance.")

            logger.info(f"Loaded {len(self._dataset)
                                  } items from {complete_path}")
            return self
        except FileNotFoundError:
            logger.error(f"Was not able to load pickle file at {complete_path}")
            if self.use_stratified_split:
                self.raw_dataset = self.initialize_stratified_raw()
            else:
                self.raw_dataset = self.initialize_raw()
            self.initialize_for_training()
            self.save()

    def equals(self, other: Dataset):
        dataset = self.dataset.to_pandas()
        return dataset.equals(other.to_pandas())

    @abstractmethod
    def _prepare_for_training(self, item: dict):
        pass

    def get_make_complete_path(self, type):
        # type will be instantiated in subclass
        assert type is not None

        Path(f"{ROOT_DATA_DIR}/easy_vqa/{type}").mkdir(parents=True, exist_ok=True)
        out = f"{ROOT_DATA_DIR}/easy_vqa/{type}/{self.split}.pkl"
        return os.path.abspath(out)
