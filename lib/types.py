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
    batch_size: int = 7
    max_length: int = 25
    shuffle_train: bool = True

    lr = 5e-4

    def __repr__(self):
        return (
            f"ModuleConfig(\n"
            f"  train_dataset_length={len(self.train_dataset)},\n"
            f"  val_dataset_length={len(self.val_dataset)},\n"
            f"  batch_size={self.batch_size},\n"
            f"  max_length={self.max_length},\n"
            f"  shuffle_train={self.shuffle_train}\n"
            f"  lr={self.lr}\n"
            f")"
        )


@dataclass
class LightningConfig:
    accumulate_grad_batches = 8
    gradient_clip_val = 1.0
    limit_val_batches = 1.0
    max_epochs = 200
    check_val_every_n_epochs = 5


class CustomDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save():
        pass

    @abstractmethod
    def initialize_for_training():
        pass

    @abstractmethod
    def load():
        pass
