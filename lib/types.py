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
    

@dataclass
class LightningConfig:
    accumulate_grad_batches: 8
    gradient_clip_val: 1.0
    limit_val_batches: 1.0
    max_epochs: 200
    check_val_every_n_epochs: 5


class CustomDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save():
        pass
