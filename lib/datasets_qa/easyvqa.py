import logging

from easy_vqa import (
    get_test_image_paths,
    get_test_questions,
    get_train_image_paths,
    get_train_questions,
)
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset

from ..utils import parse_split_slicer

logger = logging.getLogger(__name__)


class EasyVQADataset(TorchDataset):
    raw_dataset: Dataset
    _dataset: Dataset
    ready_for_training: bool = False

    def __init__(self, split: str, classify=False):
        super().__init__()
        self.classify: bool = classify

        # can be train or val
        self.split: str = split

        self.raw_dataset = self.initialize_raw()

    def initialize_raw(self):
        """Method to initialize the dataset"""

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

        _, count = parse_split_slicer(self.split)
        if count is not None:
            raw_dataset = raw_dataset.select(range(count))

        return raw_dataset

    @property
    def dataset(self):
        if self._dataset is None:
            raise Exception(
                "Please call transform() before accessing the dataset property"
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
            return self.raw_dataset[index]

    def initialize_for_training(self):
        self._dataset = self.raw_dataset.map(
            lambda item: self._prepare_for_training(item)
        )
        self.ready_for_training = True

    def _prepare_for_training(self, item):
        if self.classify:
            return self._classify(item)
        else:
            return self._autoregression(item)

    def _classify(self, item):
        raise NotImplementedError()

    def _autoregression(self, item: dict):
        input_text = item["question"]
        label = item["answer"]

        if self.split.startswith("train"):
            prompt = f"Question: {input_text} Answer: {label}."
        elif self.split.startswith("val"):
            prompt = f"{input_text}. Answer:"
        else:
            raise Exception(f"Flag {self.split} not recognized.")

        return {"question": prompt, "label": label}

    def save(self, out_dir):
        self.dataset.to_pandas().to_pickle(out_dir)

    def save(self, out_dir):
        self.dataset
