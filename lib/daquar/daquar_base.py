import logging
import os
from abc import abstractmethod
from datasets import concatenate_datasets
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
    

    def parse_dataset(self,dataset):
        images = []
        for item in dataset:
            img = Image.open(f"{DatasetPath.DAQUAR}/images/{item['image_id']}.png")
            images.append(img)
            img.close()

        parsed = {
            "question": dataset["question"],
            "answer": [answer.replace(" ", "").split(",") for answer in dataset['answer']],
            "image_id": dataset["image_id"],
            "image": images,
        }
        return Dataset.from_dict(parsed)

    
    def initialize_stratified_raw(self):
        """Method to initialize the dataset."""

        dataset_train = self.get_train_questions()['train']
        dataset_val = self.get_test_questions()['val']

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
        size=0.2 
        if self.keep_infrequent:
            filtered = combined.train_test_split(
                test_size=size,
                seed=EXPERIMENT.get_seed(),
            )
            result = filtered[split if split == Suffix.Train else Suffix.Test]
        else:
            # Now create the stratified column using the answer as the key
            combined = combined.map(lambda example: {"stratify_column": example["answer"][0]}, batched=False)
            
            # Now prepare the dataset for the stratified split
            answer_counts = Counter(combined['stratify_column'])
            valid_classes = {key for key, count in answer_counts.items() if count >= self.min_class_size}
            
            filtered = combined.filter(lambda example: example["stratify_column"] in valid_classes)
            
            # Set the answer space
            self.answer_space = list(set([label for sublist in filtered['answer'] for label in sublist]))
            
            filtered = filtered.class_encode_column("stratify_column").train_test_split(
                test_size=size,
                stratify_by_column="stratify_column",
                seed=EXPERIMENT.get_seed(),
            )
            
            result = filtered[split if split == Suffix.Train else Suffix.Test]
            logger.info(f"Read {self.split} dataset, length: {len(result)}")
        
        return result

    def initialize_stratified_raw_old(self):
        """Method to initialize the dataset."""

        if self.split.startswith(Suffix.Train):
            dataset = self.get_train_questions()['train']
        elif self.split.startswith(Suffix.Val) or self.split.startswith(Suffix.Test):
            dataset = self.get_test_questions()['val']

        images = []
        for item in dataset:
            img = Image.open(f"{DatasetPath.DAQUAR}/images/{item['image_id']}.png")
            images.append(img)
            img.close()

        dict = {
            "question": dataset["question"],
            "answer": [answer.replace(" ", "").split(",") for answer in dataset['answer']],
            "image_id": dataset["image_id"],
            "image": images,
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
        
        if self.split.startswith(Suffix.Train):
            raw_dataset.shuffle(seed=EXPERIMENT.get_seed())

        logger.info(f"Read {self.split} dataset, length: {len(raw_dataset)}")
        
        return raw_dataset

    def initialize_raw(self):
        raise NotImplementedError()

    @abstractmethod
    def _prepare_for_training(self, item: dict):
        pass

    def get_padding_max_length(self):
        return 50