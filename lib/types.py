import logging
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from enum import StrEnum
import evaluate
import numpy as np
import pandas as pd
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset
from transformers import Blip2Processor
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Model,
)

logger = logging.getLogger(__name__)


class DatasetTypes(StrEnum):
    EASY_VQA = "easy-vqa"


class ModelTypes(StrEnum):
    BLIP2Generator = "blip2-generator"
    BLIP2Classifier = "blip2-classifier"
    BLIP2BaseClassifier = "blip2-base-classifier"


class HFRepos(StrEnum):
    BLIP2_OPT = "Salesforce/blip2-opt-2.7b"
    VILT = "dandelin/vilt-b32-mlm"


# Mapping from model types to repo IDs
MODEL_REPO_MAPPING = {
    ModelTypes.BLIP2Generator: HFRepos.BLIP2_OPT.value,
    ModelTypes.BLIP2Classifier: HFRepos.BLIP2_OPT.value,
    ModelTypes.BLIP2BaseClassifier: HFRepos.BLIP2_OPT.value,
    # TODO[F] ModelTypes.VILT: HFRepos.VILT.value,
}

# Mapping from model types to model classes
MODEL_CLASS_MAPPING = {
    ModelTypes.BLIP2BaseClassifier: Blip2Model,
    ModelTypes.BLIP2Generator: Blip2ForConditionalGeneration,
    ModelTypes.BLIP2Classifier: Blip2ForConditionalGeneration,
    # TODO[F] ModelTypes.VILT: ViltForQuestionAnswering,
}

PROCESSOR_CLASS_MAPPING = {
    ModelTypes.BLIP2Generator: Blip2Processor,
    ModelTypes.BLIP2Classifier: Blip2Processor,
    ModelTypes.BLIP2BaseClassifier: Blip2Processor,
}


class SAVE_PATHS(StrEnum):
    BLIP2_Generator = "data/models/generator"
    BLIP2_Classifier = "data/models/classifier"
    BLIP2_BaseClassifier = "data/models/base_classifier"

    def make_dirs():
        Path(SAVE_PATHS.BLIP2_Generator).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATHS.BLIP2_Classifier).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATHS.BLIP2_BaseClassifier).mkdir(parents=True, exist_ok=True)


class Metric:
    name: str
    log_columns: []

    @abstractmethod
    def compute(self, pred, references):
        pass


@dataclass
class State:
    best_epoch_loss: int = np.inf
    history: defaultdict[list] = field(default_factory=lambda: defaultdict(list))
    current_epoch: int = 1
    scheduler_state_dict = None
    optimizer_state_dict = None

    def save_state(
        self,
        best_path: str,
        epoch_loss: float,
        history: dict,
        epoch: int,
        scheduler: lr_scheduler.CosineAnnealingLR,
        optimizer: AdamW,
        file_name: str = "state_dict.pkl",
    ):
        self.best_epoch_loss = epoch_loss
        self.history = history
        self.current_epoch = epoch
        self.scheduler_state_dict = scheduler.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()

        with open(f"{best_path}/{file_name}", "wb") as file:
            pickle.dump(self, file)

        logger.info(f"Results were saved to {best_path}/{file_name}")

    @classmethod
    def load_state(self, path):
        try:
            return pd.read_pickle(f"{path}/state_dict.pkl")
        except FileNotFoundError:
            return State()


@dataclass
class VQAParameters:
    split: str
    is_testing: bool = False
    padding_max_length: int = 25
    processor: Blip2Processor = None
    load_from_disk: bool = True
    dataset_name: str = DatasetTypes.EASY_VQA
    use_stratified_split: bool = False


@dataclass
class TrainingParameters:
    resume: bool
    model_name: str
    is_trainable: bool
    train_args: VQAParameters
    val_args: VQAParameters
    test_args: VQAParameters
    wandb_project: str = "ComputerVision"
    shuffle_train: bool = True

    def __repr__(self):
        return (
            f"ModuleConfig(\n"
            f"  model_name={self.model_name},\n"
            f"  train_dataset_length={len(self.train_dataset)},\n"
            f"  val_dataset_length={len(self.val_dataset)},\n"
            f"  shuffle_train={self.shuffle_train}\n"
        )

    num_epochs: int = 2
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    n_accumulate: int = 1
    train_batch_size: int = 64
    val_batch_size: int = 64
    test_batch_size: int = 1
    optimizer: AdamW = None
    scheduler: lr_scheduler.CosineAnnealingLR = None

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
    answer_space: int

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

    def compute(self, pred, references, log=True):
        # Calculate the scores
        scores = self.metric.compute(predictions=pred, references=references, lang="en")

        # Log the results
        if log:
            logger.info(f"val_bertscore_recall: {np.mean(scores["recall"])}")
            logger.info(f"val_bertscore_precision: {np.mean(scores["precision"])}")
            logger.info(f"val_bertscore_f1: {np.mean(scores["f1"])}")
        return scores
