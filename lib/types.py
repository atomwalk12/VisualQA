from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor


@dataclass
class ModuleConfig:
    train_dataset: Dataset
    val_dataset: Dataset
    processor: AutoProcessor
    model: AutoModel
    shuffle_train: bool = True
    batch_size: int = 7
    max_length: int = 25


class CustomDataset(Dataset, ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save():
        pass
