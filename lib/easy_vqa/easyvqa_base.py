import logging

from datasets import Dataset
from easy_vqa import get_answers, get_test_image_paths, get_test_questions, get_train_image_paths, get_train_questions
from PIL import Image

from ..types import DatasetTypes, Suffix, VQAParameters
from ..utils import EXPERIMENT, parse_split_slicer
from ..dataset_base import DatabaseBase

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
    
    def initialize_stratified_raw(self):
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
            "image": [Image.open(images[image_id]) for image_id in questions[2]],
        }

        # Needs a shuffle, otherwise the stratification doesn't work since after
        # filtering there will be too few entries from some classes.
        raw_dataset = Dataset.from_dict(data_dict)
        raw_dataset.shuffle(seed=EXPERIMENT.get_seed())

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
                shuffle=True
            )
            raw_dataset = ds[split if split == Suffix.Train else Suffix.Test]
            

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset

    def initialize_raw(self):
        """Method to initialize the dataset."""

        if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Val):
            questions = get_train_questions()
            images = get_train_image_paths()
        elif self.split.startswith(Suffix.Test):
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
        if self.split.startswith(Suffix.Train) or self.split.startswith(Suffix.Val):
            target = Suffix.Test if split.startswith(Suffix.Val) else split
            raw_dataset = raw_dataset.train_test_split(test_size=0.25, seed=EXPERIMENT.get_seed())[target]

        if start is not None or end is not None:
            raw_dataset = raw_dataset.select(range(start or 0, end or len(raw_dataset)))

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        return raw_dataset

    def get_padding_max_length(self):
        return 25