import logging

from easy_vqa import get_train_image_paths, get_train_questions
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EasyVQADataset(Dataset):
    dataset: Dataset

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.dataset = get_train_questions()
