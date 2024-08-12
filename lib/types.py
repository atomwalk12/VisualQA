import logging
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset
from transformers import Blip2ForConditionalGeneration, Blip2Model, Blip2Processor

logger = logging.getLogger(__name__)


class DatasetTypes(StrEnum):
    EASY_VQA = "easy-vqa"
    DAQUAR = "daquar"

class DatasetPath(StrEnum):
    DAQUAR = "data/daquar/dataset"

class ModelTypes(StrEnum):
    BLIP2Generator = "blip2-generator"
    BLIP2Classifier = "blip2-classifier"
    BLIP2BaseClassifier = "blip2-base-classifier"


class HFRepos(StrEnum):
    BLIP2_OPT = "Salesforce/blip2-opt-2.7b"


class Suffix(StrEnum):
    Test = "test"
    Train = "train"
    Val = "val"


# Mapping from model types to repo IDs
MODEL_REPO_MAPPING = {
    ModelTypes.BLIP2Generator: HFRepos.BLIP2_OPT.value,
    ModelTypes.BLIP2Classifier: HFRepos.BLIP2_OPT.value,
    ModelTypes.BLIP2BaseClassifier: HFRepos.BLIP2_OPT.value,
}

# Mapping from model types to model classes
MODEL_CLASS_MAPPING = {
    ModelTypes.BLIP2BaseClassifier: Blip2Model,
    ModelTypes.BLIP2Generator: Blip2ForConditionalGeneration,
    ModelTypes.BLIP2Classifier: Blip2ForConditionalGeneration,
}

PROCESSOR_CLASS_MAPPING = {
    ModelTypes.BLIP2Generator: Blip2Processor,
    ModelTypes.BLIP2Classifier: Blip2Processor,
    ModelTypes.BLIP2BaseClassifier: Blip2Processor,
}


class SAVE_PATHS(StrEnum):
    BLIP2_Generator_EasyVQA = "data/models/easy_vqa/generator"
    BLIP2_Classifier_EasyVQA = "data/models/easy_vqa/classifier"
    BLIP2_Generator_DAQUAR = "data/models/daquar/generator"
    BLIP2_Classifier_DAQUAR = "data/models/daquar/classifier"

    def make_dirs():
        Path(SAVE_PATHS.BLIP2_Generator_EasyVQA).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATHS.BLIP2_Classifier_EasyVQA).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATHS.BLIP2_Generator_DAQUAR).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATHS.BLIP2_Classifier_DAQUAR).mkdir(parents=True, exist_ok=True)

class CustomDataset(Dataset, ABC):
    ready_for_training: bool
    answer_space: int
    split: str

    def __init__(self, split) -> None:
        super().__init__()
        self.split = split

    @abstractmethod
    def save():
        pass

    @abstractmethod
    def initialize_for_training():
        pass

    @abstractmethod
    def load():
        pass


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
        dataset: CustomDataset,
        file_name: str = "state_dict",
    ):
        self.best_epoch_loss = epoch_loss
        self.history = history
        self.current_epoch = epoch
        self.scheduler_state_dict = scheduler.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()

        path = f"{best_path}/{file_name}_{dataset.split}.pkl"
        with open(path, "wb") as file:
            pickle.dump(self, file)

        logger.info(f"Results were saved to {path}")
        
    def save_embeddings(
        self,
        best_path: str,
        file_name: str,
    ):
        path = f"{best_path}/{file_name}"
        with open(path, "wb") as file:
            pickle.dump(self, file)

        logger.info(f"Results were saved to {path}")

    @classmethod
    def load_state(self, path, dataset_name, filename: str):
        path = f"{path}/{dataset_name}/{filename}"
        try:
            return pd.read_pickle(path)
        except FileNotFoundError:
            raise


@dataclass
class VQAParameters:
    split: str
    is_testing: bool = False
    processor: Blip2Processor = None
    load_from_disk: bool = True
    dataset_name: str = DatasetTypes.EASY_VQA
    use_stratified_split: bool = False


@dataclass
class TrainingParameters:
    resume_checkpoint: bool
    model_name: str
    is_trainable: bool
    train_args: VQAParameters
    val_args: VQAParameters
    test_args: VQAParameters
    dataset_name: str
    wandb_project: str = "ComputerVision"
    shuffle_train: bool = True
    num_train_workers: int = 12
    num_val_workers: int = 8
    num_test_workers: int = 8
    use_wandb: bool = True
    split: str = None
    resume_state: bool = True
    
    def __post_init__(self):
        if self.test_args is not None:
            self.split = self.test_args.split
        if self.train_args is not None:
            self.split = self.train_args.split

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
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
        elif self.scheduler_name == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=min_lr)
        elif self.scheduler_name is None:
            return None

        return scheduler

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
