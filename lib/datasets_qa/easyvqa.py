import logging

from easy_vqa import get_train_image_paths, get_train_questions
from PIL import Image
from torch.utils.data import Dataset

from ...ckpt.lib.types import EasyVQAElement

logger = logging.getLogger(__name__)


class EasyVQADataset(Dataset):
    dataset: Dataset

    def __init__(self):
        super().__init__()

    def initialize(self):
        """Method to initialize the dataset"""

        questions = get_train_questions()
        images = get_train_image_paths()

        # Combine and process elements
        self.dataset: Dataset[EasyVQAElement] = [
            self._translate(elements, images[elements[2]])
            for elements in zip(*questions)
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        return self.dataset[index]

    def _translate(self, item, image_path):
        """
        Translates the raw data retrieved from the data into a EasyVQAElement.

        Args:
            item (str): The retrieved raw data

        Returns:
            EasyVQAElement: An element of the dataset
        """
        return EasyVQAElement(
            question=item[0], answer=item[1], image_id=item[2], image_path=image_path, image=Image.open(image_path)
        )
