from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class ModuleConfig():
    train_dataset: Dataset
    val_dataset: Dataset
    batch_size: int
