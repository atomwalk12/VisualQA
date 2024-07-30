import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import evaluate
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor


logger = logging.getLogger(__name__)


class Metric:
    name: str
    log_columns: []

    @abstractmethod
    def compute(self, pred, references):
        pass


@dataclass
class ModuleConfig:
    train_dataset: Dataset
    val_dataset: Dataset
    metrics: List[Metric]
    processor: AutoProcessor = None
    model: AutoModel = None
    batch_size: int = 12
    max_length: int = 25
    shuffle_train: bool = True

    lr = 5e-4

    def __repr__(self):
        metrics = " ".join(x.name for x in self.metrics)

        return (
            f"ModuleConfig(\n"
            f"  train_dataset_length={len(self.train_dataset)},\n"
            f"  val_dataset_length={len(self.val_dataset)},\n"
            f"  batch_size={self.batch_size},\n"
            f"  max_length={self.max_length},\n"
            f"  shuffle_train={self.shuffle_train}\n"
            f"  lr={self.lr}\n"
            f"  metrics={metrics}\n"
        )


@dataclass
class LightningConfig:
    accumulate_grad_batches = 8
    gradient_clip_val = 1.0
    limit_val_batches = 1.0
    limit_train_batches = 1.0
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


class BertScoreMetric(Metric):
    def __init__(self) -> None:
        self.metric = evaluate.load("bertscore")
        self.name = "bertscore"
        self.log_columns = ["recall", "precision", "f1"]

    def compute(self, pred, references):
        # Calculate the scores
        scores = self.metric.compute(predictions=pred, references=references, lang="en")

        # Log the results
        logger.info(f"val_bertscore_recall: {np.mean(scores["recall"])}")
        logger.info(f"val_bertscore_precision: {np.mean(scores["precision"])}")
        logger.info(f"val_bertscore_f1: {np.mean(scores["f1"])}")
        return scores
