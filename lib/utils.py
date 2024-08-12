import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from lib.experiments import Reproducible

from .types import Suffix

EXPERIMENT: Reproducible = Reproducible()
ROOT_DATA_DIR = "data/"
MODELS_DIR = f"{ROOT_DATA_DIR}/models/"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_split_slicer(split: str):
    if "[" in split:
        split, slicer = split.split("[")
        slicer = slicer[:-1]  # Remove the trailing ']'
        if ":" in slicer:
            start, end = slicer.split(":")
            start = int(start) if start else None
            end = int(end) if end else None
        else:
            start = int(slicer)
            end = None
    else:
        start = None
        end = None

    return split, start, end


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_time(seconds):
    """
    Format time in seconds to hours, minutes, and seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int((seconds % 3600) % 60)
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"


class MetricsAccumulator:
    update_frequency = 10

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_metrics(self, outputs, labels, phase: Suffix):
        loss = outputs.loss
        logits = outputs.logits
        targets = labels

        metrics = {
            f"{phase}_loss": loss.item(),
            f"{phase}_perplexity": self.calculate_perplexity(loss).item(),
            f"{phase}_bleu": self.adapted_bleu_score(logits, targets),
            f"{phase}_token_accuracy": self.token_level_accuracy(logits, targets).item(),
            f"{phase}_entropy": self.entropy_of_predictions(logits).item(),
            f"{phase}_top_5_accuracy": self.top_k_accuracy(logits.to("cpu"), targets, k=5).item(),
        }

        return metrics

    def reset(self):
        self.__init__(self.tokenizer)

    def calculate_perplexity(self, loss):
        return torch.exp(loss)

    def token_level_accuracy(self, logits, targets):
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions.to("cpu") == targets).float()
        return correct.mean()

    def adapted_bleu_score(self, logits, targets):
        smoothing = SmoothingFunction().method1

        predictions = torch.argmax(logits, dim=-1)
        pred_text = self.tokenizer.decode(predictions[0])
        target_text = self.tokenizer.decode(targets[0])
        return sentence_bleu([target_text.split()], pred_text.split(), smoothing_function=smoothing, weights=(0.5, 0.5))

    def entropy_of_predictions(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()

    def top_k_accuracy(self, logits, targets, k=5):
        _, top_k_indices = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_indices)
        correct = (top_k_indices == targets_expanded).sum(dim=-1)
        return (correct > 0).float().mean()
