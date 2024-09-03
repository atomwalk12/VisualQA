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
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset
from transformers import Blip2ForConditionalGeneration, Blip2Processor, Blip2ForImageTextRetrieval, AutoProcessor

logger = logging.getLogger(__name__)


class FileNames(StrEnum):
    UMAPEmbedding = "embeddings_{0}.pkl"
    ConfusionMatrixPDF = "confusion_matrix_{0}.pdf"
    UMAPClustering = "umap_{0}.pdf"
    StateDictionary = "state_dict_{0}.pkl"
    ConfusionMatrix = "confusion_matrix_{0}.pkl"


class EvaluationMetrics(StrEnum):
    ConfusionMatrix = "confusion-matrix"
    UMAP = "umap"
    DATA_DISTRIBUTION = "distribution"


class DatasetTypes(StrEnum):
    EASY_VQA = "easy-vqa"
    DAQUAR = "daquar"


class DatasetPath(StrEnum):
    DAQUAR = "data/daquar/dataset"


class ModelTypes(StrEnum):
    BLIP2Generator = "blip2-generator"
    BLIP2Classifier = "blip2-classifier"
    BLIP2FinetunedClassifier = "blip2-finetuned-classifier"
    BLIP2FinetunedGenerator = "blip2-finetuned-generator"
    BLIP2FinetunedBaseClassifier = "blip2-finetuned-base-classifier"


class HFRepos(StrEnum):
    BLIP2_OPT = "Salesforce/blip2-opt-2.7b"
    BLIP2_COCO = "Salesforce/blip2-opt-2.7b-coco"
    BLIP2_ITM = "Salesforce/blip2-itm-vit-g"


class Suffix(StrEnum):
    Test = "test"
    Train = "train"
    Val = "val"
    All = "all"


# Mapping from model types to repo IDs
MODEL_REPO_MAPPING = {
    ModelTypes.BLIP2Generator: HFRepos.BLIP2_OPT.value,
    ModelTypes.BLIP2Classifier: HFRepos.BLIP2_ITM.value,
    ModelTypes.BLIP2FinetunedClassifier: HFRepos.BLIP2_COCO.value,
    ModelTypes.BLIP2FinetunedGenerator: HFRepos.BLIP2_COCO.value,
}

# Mapping from model types to model classes
MODEL_CLASS_MAPPING = {
    ModelTypes.BLIP2FinetunedGenerator: Blip2ForConditionalGeneration,
    ModelTypes.BLIP2FinetunedClassifier: Blip2ForConditionalGeneration,
    ModelTypes.BLIP2Generator: Blip2ForConditionalGeneration,
    ModelTypes.BLIP2Classifier: Blip2ForImageTextRetrieval,
}

PROCESSOR_CLASS_MAPPING = {
    ModelTypes.BLIP2Generator: Blip2Processor,
    ModelTypes.BLIP2Classifier: AutoProcessor,
    ModelTypes.BLIP2FinetunedGenerator: Blip2Processor,
    ModelTypes.BLIP2FinetunedClassifier: Blip2Processor,
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

    @staticmethod
    def make_dir(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)


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
    def prepare_labels():
        pass

    @abstractmethod
    def load():
        pass


class Metric:
    name: str
    log_columns = []

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

    def save_state_to_file(
        self,
        best_path: str,
        file_name: str,
    ):
        path = f"{best_path}/{file_name}"
        with open(path, "wb") as file:
            pickle.dump(self, file)

        logger.info(f"Results were saved to {path}")

    @classmethod
    def load_state(self, path, filename: str):
        path = f"{path}/{filename}"
        try:
            return pd.read_pickle(path)
        except FileNotFoundError:
            raise


@dataclass
class VQAParameters:
    split: str
    is_testing: bool = False
    processor: Blip2Processor = None
    recompute: bool = False
    use_filtered_split: bool = False
    use_proportional_split: bool = False
    keep_infrequent: bool = False

    def __hash__(self):
        return hash(
            (
                self.split,
                self.is_testing,
                self.recompute,
                self.use_filtered_split,
                self.use_proportional_split,
                self.keep_infrequent,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, VQAParameters):
            return False
        return (
            self.split == other.split
            and self.is_testing == other.is_testing
            and self.recompute == other.recompute
            and self.use_filtered_split == other.use_filtered_split
            and self.use_proportional_split == other.use_proportional_split
            and self.keep_infrequent == other.keep_infrequent
        )

@dataclass
class Blip2ClassificationModelOutput:
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: torch.FloatTensor = None
    vision_model_output: torch.FloatTensor = None


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
    lora_config = None
    bnb_config = None

    @property
    def _resume_state(self) -> int:
        """Getter for my_property."""
        if self.is_trainable and self.resume_checkpoint and not self.resume_state:
            return True
        return self.resume_state

    @_resume_state.setter
    def _resume_state(self, value):
        self.resume_state = value

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

    num_epochs: int = 200
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    n_accumulate: int = 1
    train_batch_size: int = 64
    val_batch_size: int = 64
    test_batch_size: int = 1
    optimizer: AdamW = None
    scheduler: lr_scheduler.CosineAnnealingLR = None

    def set_optimizer_and_scheduler(self, model):
        if self.optimizer_name == "AdamW" and self.scheduler_name == "CosineAnnealingLR":
            self.optimizer = self.fetch_optimizer(model)
            self.scheduler = self.fetch_scheduler(self.optimizer)
        elif self.optimizer_name == "AdamW" and self.scheduler_name == "CosineAnnealingWarmRestarts":
            self.optimizer = self.fetch_optimizer(model, lr=5e-5, weight_decay=0.001)
            self.scheduler = self.fetch_scheduler(self.optimizer)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} and scheduler {self.scheduler_name} not implemented")
        
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
            logger.info(f"val_bertscore_precision: {np.mean(scores["precision"])}")
            logger.info(f"val_bertscore_f1: {np.mean(scores["f1"])}")

        return scores
