import logging

from easy_vqa import get_train_questions
from torch.utils.data import Dataset
from ..types import EasyVQAElement

logger = logging.getLogger(__name__)


class EasyVQADataset(Dataset):
    dataset: Dataset

    def __init__(self):
        super().__init__()

    def initialize(self):
        """Method to initialize the dataset"""

        questions = get_train_questions()

        # Use zip to combine elements at corresponding positions
        zipped_elements = zip(*questions)

        # Use map to apply the process_elements function to each tuple of elements
        processed_results = list(map(self._translate, zipped_elements))

        self.dataset = processed_results

    def __getitem__(self, index: int) -> dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        return self.dataset[index]

    def _translate(self, item: str):
        return EasyVQAElement()
