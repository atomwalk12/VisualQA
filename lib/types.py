import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import evaluate
import numpy as np
from torch.optim import AdamW, lr_scheduler
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
class TorchTrainerConfig:
    num_epochs: int = 5
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    n_accumulate: int = 1
    train_batch_size: int = 64
    val_batch_size: int = 64
    optimizer: AdamW = None
    scheduler: lr_scheduler = None

    def set_optimizer_and_scheduler(self, model):
        self.optimizer = self.fetch_optimizer(model)
        self.scheduler = self.fetch_scheduler(self.optimizer)

    def fetch_optimizer(self, model, lr=1e-4, weight_decay=1e-6):
        assert self.optimizer_name == "AdamW", "Optimizer not implemented"

        if self.optimizer_name == "AdamW":
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def fetch_scheduler(self, optimizer, t_max=500, min_lr=1e-6, T_0=15):
        if self.scheduler_name == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=min_lr
            )
        elif self.scheduler_name == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, eta_min=min_lr
            )
        elif self.scheduler_name is None:
            return None

        return scheduler


@dataclass
class ModuleConfig:
    torch_hyperparameters: TorchTrainerConfig

    train_dataset: Dataset
    val_dataset: Dataset
    metrics: List[Metric]
    processor: AutoProcessor
    model: AutoModel
    model_name: str
    max_length: int = 25
    wandb_project: str = "ComputerVision"
    shuffle_train: bool = True

    def __repr__(self):
        metrics = " ".join(x.name for x in self.metrics)

        return (
            f"ModuleConfig(\n"
            f"  model_name={self.model_name},\n"
            f"  train_dataset_length={len(self.train_dataset)},\n"
            f"  val_dataset_length={len(self.val_dataset)},\n"
            f"  max_length={self.max_length},\n"
            f"  shuffle_train={self.shuffle_train}\n"
            f"  metrics={metrics})\n"
        )


@dataclass
class LightningConfig:
    accumulate_grad_batches = 8
    gradient_clip_val = 1.0
    max_epochs = 200
    check_val_every_n_epochs = 5

    def __init__(self, limit_train_batches=None, limit_val_batches=None):
        self.limit_train_batches = (
            limit_train_batches if limit_train_batches is None else 1.0
        )
        self.limit_val_batches = limit_val_batches if limit_val_batches is None else 1.0


class CustomDataset(Dataset, ABC):
    ready_for_training: bool
    
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

    @abstractmethod
    def shuffle(seed):
        pass


class BertScoreMetric(Metric):
    def __init__(self) -> None:
        self.metric = evaluate.load("bertscore")
        self.name = "bertscore"
        self.log_columns = ["recall", "precision", "f1"]

    def compute(self, pred, references, log=True):
        # Calculate the scores
        scores = self.metric.compute(predictions=pred, references=references, lang="en")

        # Log the results
        if log:
            logger.info(f"val_bertscore_recall: {np.mean(scores["recall"])}")
            logger.info(f"val_bertscore_precision: {np.mean(scores["precision"])}")
            logger.info(f"val_bertscore_f1: {np.mean(scores["f1"])}")
        return scores
