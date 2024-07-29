from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModel


@dataclass
class ModuleConfig:
    train_dataset: Dataset
    val_dataset: Dataset
    processor: AutoProcessor
    model: AutoModel
    shuffle_train: bool = True
    batch_size: int = 7
    max_length: int = 25
