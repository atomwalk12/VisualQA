import logging

from easy_vqa import get_train_image_paths, get_train_questions, get_test_questions, get_test_image_paths
from PIL import Image
from torch.utils.data import Dataset

from ..types import EasyVQAElement, EasyVQARawElement

logger = logging.getLogger(__name__)


class EasyVQADataset(Dataset):
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

        if self.split.startswith('train'):
            questions = get_train_questions()
            images = get_train_image_paths()
        elif self.split.startswith('val'):
            questions = get_test_questions()
            images = get_test_image_paths()

        if '[' in self.split:
            split, count = self.split.split("[:")
            count = int(count[:-1])
        else:
            count = None

        # Combine and process elements
        raw_dataset = [
            self._translate(elements, images[elements[2]])
            for elements in zip(*questions)
        ]

        if count is not None:
            return raw_dataset[:count]
        else:
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

    def __getitem__(self, index: int) -> EasyVQARawElement:
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

    def _translate(self, item, image_path):
        """
        Translates the raw data retrieved from the data into a EasyVQAElement.

        Args:
            item (str): The retrieved raw data

        Returns:
            EasyVQAElement: An element of the dataset
        """
        return EasyVQARawElement(
            question=item[0],
            answer=item[1],
            image_id=item[2],
            image_path=image_path,
            image=Image.open(image_path),
        )

    def initialize_for_training(self):
        self._dataset = [self._prepare_for_training(
            item) for item in self.raw_dataset]
        self.ready_for_training = True

    def _prepare_for_training(self, item):
        if self.classify:
            return self._classify(item)
        else:
            return self._autoregression(item)

    def _classify(self, item):
        raise NotImplementedError()

    def _autoregression(self, item: EasyVQARawElement):
        input_text = item.question
        label = item.answer

        if self.split.startswith('train'):
            prompt = f"Question: {input_text} Answer: {label}."
        elif self.split.startswith('val'):
            prompt = f"{input_text}. Answer:"
        else:
            raise Exception(f"Flag {self.split} not recognized.")

        return EasyVQAElement(question=prompt, label=label)
