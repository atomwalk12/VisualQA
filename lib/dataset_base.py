import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from datasets import Dataset

from .types import CustomDataset, DatasetTypes, Suffix, VQAParameters
from .utils import ROOT_DATA_DIR

logger = logging.getLogger(__name__)


class DatabaseBase(CustomDataset, ABC):
    raw_dataset: Dataset = None
    _dataset: Dataset = None
    padding_max_length: int

    def __init__(self, dataset_name: DatasetTypes, params: VQAParameters):
        super().__init__(params.split)
        self.dataset_name = dataset_name

        assert params.processor is not None
        
        self.padding_max_length = self.get_padding_max_length()
        
        self.min_class_size = 50

        # Store parameters
        self.processor = params.processor
        self.use_stratified_split = params.use_stratified_split

        # can be train or val
        self.split: str = params.split

        # Store the answer space for commodity
        self.answer_space = self._get_answers()

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
        self.use_raw_dataset = False

    @property
    def dataset(self):
        if self._dataset is None:
            raise Exception("Please call transform() before accessing the training dataset.")
        return self._dataset

    def __len__(self) -> int:
        if self.use_raw_dataset:
            return len(self.raw_dataset)
        else:
            return len(self._dataset)

    def __getitem__(self, idx):
        if self.use_raw_dataset:
            return self.raw_dataset[idx]

        # Encode the dataset item
        item = self.dataset[idx]

        if self.split.startswith(Suffix.Test):
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
        complete_path = self.get_make_complete_path()

        try:
            loaded_data = pd.read_pickle(complete_path)

            for key, value in vars(loaded_data).items():
                if hasattr(self, key):
                    if key == "use_stratified_split" and self.use_stratified_split != value:
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

    def get_make_complete_path(self, type):
        # type will be instantiated in subclass
        assert type is not None

        Path(f"{ROOT_DATA_DIR}/{self.dataset_name}/{type}").mkdir(parents=True, exist_ok=True)
        out = f"{ROOT_DATA_DIR}/{self.dataset_name}/{type}/{self.split}.pkl"
        return os.path.abspath(out)

    def equals(self, other: Dataset):
        dataset = self.dataset.to_pandas()
        return dataset.equals(other.to_pandas())

    @abstractmethod
    def initialize_stratified_raw(self):
        pass

    @abstractmethod
    def initialize_raw(self):
        pass

    @abstractmethod
    def _prepare_for_training(self, item: dict):
        pass

    @abstractmethod
    def _get_answers(self):
        pass

    @abstractmethod
    def get_padding_max_length(self):
        pass