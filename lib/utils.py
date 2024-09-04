from contextlib import contextmanager
import logging
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
from sentence_transformers import util
import wandb
from lib.experiments import Reproducible

from .types import DatasetTypes, Suffix
import numpy as np

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


class ClassificationMetricsAccumulator:

    def __init__(self, dataset_name, answer_space, name, update_frequency, plot_confusion_matrix=True):
        self.dataset_name = dataset_name
        self.accumulated_targets = []
        self.accumulated_predictions = []
        self.accumulated_predictions_epoch = []
        self.accumulated_targets_epoch = []
        self.iteration = 0
        self.answer_space = answer_space
        self.epoch = 1
        self.name = name
        self.update_frequency = update_frequency
        self.plot_confusion_matrix = plot_confusion_matrix

    def log_multi_class_statistics(self, y_pred, y_true):
        logits = y_pred.logits.clone().detach().cpu()
        targets = y_true.clone().detach().cpu()
        self.iteration += 1

        # For multi-class, apply softmax and select the class with the highest probability
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        targets = torch.argmax(targets, dim=1)

        self._accumulate_data_multi_class(targets, predictions)

        if self.iteration % self.update_frequency == 0:
            self.report_multi_class_statistics()
            
    def report_multi_class_statistics(self):
        predictions = self.accumulated_predictions
        targets = self.accumulated_targets
        if len(targets) > 0:
            report = self.generate_report(targets, predictions, multi_label=False)
            accuracy = accuracy_score(targets, predictions)

            report[f"{self.name}_accuracy"] = accuracy
            wandb.log(report)

        self.accumulated_targets = []
        self.accumulated_predictions = []

    def generate_report(self, targets, predictions, multi_label):
        if multi_label:
            report = classification_report(targets, predictions, target_names=self.answer_space, output_dict=True, zero_division=0.0)
        else:
            # Classification report
            report = classification_report(targets, predictions, labels=np.unique(predictions), output_dict=True, zero_division=0.0)

        # Extract relevant metrics from the classification report
        precision_macro = report["macro avg"]["precision"]
        recall_macro = report["macro avg"]["recall"]
        f1_macro = report["macro avg"]["f1-score"]

        precision_weighted = report["weighted avg"]["precision"]
        recall_weighted = report["weighted avg"]["recall"]
        f1_weighted = report["weighted avg"]["f1-score"]

        result = {
            f"{self.name}_precision_weighted": precision_weighted,
            f"{self.name}_recall_weighted": recall_weighted,
            f"{self.name}_f1_weighted": f1_weighted,
            f"{self.name}_precision_macro": precision_macro,
            f"{self.name}_recall_macro": recall_macro,
            f"{self.name}_f1_macro": f1_macro,
        }

        if multi_label:
            precision_micro = report["micro avg"]["precision"]
            recall_micro = report["micro avg"]["recall"]
            f1_micro = report["micro avg"]["f1-score"]
            precision_samples = report["samples avg"]["precision"]
            recall_samples = report["samples avg"]["recall"]
            f1_samples = report["samples avg"]["f1-score"]

            result[f"{self.name}_precision_samples"] = precision_samples
            result[f"{self.name}_recall_samples"] = recall_samples
            result[f"{self.name}_f1_samples"] = f1_samples
            result[f"{self.name}_precision_micro"] = precision_micro
            result[f"{self.name}_recall_micro"] = recall_micro
            result[f"{self.name}_f1_micro"] = f1_micro

        return result

    def log_multi_label_statistics(self, y_pred, y_true):
        logits = y_pred.logits.clone().detach().cpu()
        targets = y_true.clone().detach().cpu()
        self.iteration += 1

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Method 1: Adaptive Thresholding
        threshold = probs.mean()  # Use the mean probability as the threshold
        predictions_adaptive = (probs > threshold).int()

        # Method 2: Top-3 Selection
        k = 3 
        _, top_3_indices = torch.topk(probs, k, dim=1)
        predictions_top_3 = torch.zeros_like(probs)
        predictions_top_3.scatter_(1, top_3_indices, 1)
        
        # Method 2: Top-2 Selection
        k = 2
        _, top_2_indices = torch.topk(probs, k, dim=1)
        predictions_top_2 = torch.zeros_like(probs)
        predictions_top_2.scatter_(1, top_2_indices, 1)
        
        # Method 2: Top-1 Selection
        k = 1
        _, top_1_indices = torch.topk(probs, k, dim=1)
        predictions_top_1 = torch.zeros_like(probs)
        predictions_top_1.scatter_(1, top_1_indices, 1)

        # Method 3: Percentile Thresholding
        percentile_threshold = np.percentile(probs.numpy(), 70)  # 70th percentile
        predictions_percentile = (probs > percentile_threshold).int()

        # Store all types of predictions
        self._accumulate_data_multi_label(targets, {
            'adaptive': predictions_adaptive,
            'top_3': predictions_top_3,
            'top_2': predictions_top_2,
            'top_1': predictions_top_1,
            'percentile': predictions_percentile
        })

        if self.iteration % self.update_frequency == 0:
            self.report_multi_label_statistics()

    def report_multi_label_statistics(self):
        if len(self.accumulated_predictions) > 0:
            targets = torch.stack([t for t, _ in self.accumulated_predictions])
            predictions = {
                method: torch.stack([p[method] for _, p in self.accumulated_predictions])
                for method in ['adaptive', 'top_3', 'top_2', 'top_1', 'percentile']
            }

            # Squeeze the extra dimension
            targets = targets.squeeze(0).numpy().astype(int)
            predictions = {method: preds.squeeze(0).numpy() for method, preds in predictions.items()}

            for method, preds in predictions.items():
                report = self.generate_report(targets, preds, multi_label=True)
                
                # Add method-specific prefix to metric names
                report = {f"{method}_{k}": v for k, v in report.items()}
                
                # Multi-label statistics
                report[f"{self.name}_{method}_hamming_loss"] = hamming_loss(targets, preds)
                report[f"{self.name}_{method}_jaccard"] = jaccard_score(targets, preds, average="samples", zero_division=0.0)
                report[f"{self.name}_{method}_accuracy"] = accuracy_score(targets, preds, normalize=True, sample_weight=None)
                report[f"{self.name}_{method}_exact_match_ratio"] = (preds == targets).all(axis=1).mean()

                wandb.log(report)

        self.accumulated_predictions = []

    def log_confusion_matrix(self):
        if self.plot_confusion_matrix:
            title = f"{self.epoch}_{self.name}_Confusion_Matrix"
            wandb.log(
                {
                    title: wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=[t.numpy().item() for t in self.accumulated_targets_epoch],
                        preds=[p.numpy().item() for p in self.accumulated_predictions_epoch],
                        class_names=self.answer_space,
                        title=title,
                    )
                }
            )
        self.accumulated_predictions_epoch = []
        self.accumulated_targets_epoch = []
        self.epoch += 1
    
    def _accumulate_data_multi_class(self, targets, predictions):
        self.accumulated_predictions.extend(predictions)
        self.accumulated_targets.extend(targets)
        if self.plot_confusion_matrix:
            self.accumulated_predictions_epoch.extend(predictions)
            self.accumulated_targets_epoch.extend(targets)

    def _accumulate_data_multi_label(self, targets, predictions):
        self.accumulated_predictions.append((targets, predictions))


class GeneratorMetricsAccumulator:

    def __init__(self, tokenizer, sbert, name, update_frequency):
        self.tokenizer = tokenizer
        self.iteration = 0
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.accumulated_loss = []
        self.name = name
        self.sbert = sbert
        self.update_frequency = update_frequency

    def sbert_similarity(self, logits, _targets):

        # Convert logits to predicted token indices: shape: (batch_size, sequence_length)
        predictions = torch.argmax(logits, dim=-1) 
        
        # Decode predictions and targets into sentences
        sentences1 = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        sentences2 = self.tokenizer.batch_decode(_targets, skip_special_tokens=True)
        
        # Encode sentences into embeddings
        embeddings1 = self.sbert.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.sbert.encode(sentences2, convert_to_tensor=True)
        
        # Calculate cosine similarity between embeddings
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        
        # Average the cosine scores across the batch dimension
        return cosine_scores.mean().item()
    
    def log_metrics(self, outputs, labels):
        loss = outputs.loss.clone().detach().cpu()
        predictions = outputs.logits.clone().detach().cpu()
        targets = labels.clone().detach().cpu()
        self.iteration += 1
        self._accumulate_data(targets, predictions, loss)
        
        if self.iteration % self.update_frequency == 0:
            self.report()
    
    def report(self):
        predictions = self.accumulated_predictions
        targets = self.accumulated_targets
        accumulated_loss = self.accumulated_loss

        # Initialize a dictionary to accumulate the metrics
        accumulated_metrics = {
            f"{self.name}_loss": 0.0,
            f"{self.name}_perplexity": 0.0,
            f"{self.name}_bleu": 0.0,
            f"{self.name}_sbert_similarity": 0.0,
            f"{self.name}_token_accuracy": 0.0,
            f"{self.name}_entropy": 0.0,
            f"{self.name}_top_5_accuracy": 0.0,
        }

        # Iterate over the pairs of target and prediction
        for loss, target, prediction in zip(accumulated_loss, targets, predictions):
            metrics = {
                f"{self.name}_loss": loss.item(),
                f"{self.name}_sbert_similarity": self.sbert_similarity(prediction, target),
                f"{self.name}_perplexity": self.calculate_perplexity(loss).item(),
                f"{self.name}_bleu": self.adapted_bleu_score(prediction, target),
                f"{self.name}_token_accuracy": self.token_level_accuracy(prediction, target).item(),
                f"{self.name}_entropy": self.entropy_of_predictions(prediction).item(),
                f"{self.name}_top_5_accuracy": self.top_k_accuracy(prediction.to("cpu"), target, k=5).item(),
            }
            
            # Accumulate the metrics
            for key, value in metrics.items():
                accumulated_metrics[key] += value

        # Compute the average of each metric
        num_samples = len(accumulated_loss)
        if num_samples > 0:
            averaged_metrics = {key: value / num_samples for key, value in accumulated_metrics.items()}
            wandb.log(averaged_metrics)
            
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.accumulated_loss = []

    def _accumulate_data(self, targets, predictions, loss):
        self.accumulated_targets.append(targets)
        self.accumulated_predictions.append(predictions)
        self.accumulated_loss.append(loss)

    def calculate_perplexity(self, loss):
        loss = loss.clone().detach()
        return torch.exp(loss)

    def token_level_accuracy(self, logits, targets):
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions.to("cpu") == targets).float()
        return correct.mean()

    def adapted_bleu_score(self, logits, targets):
        smoothing = SmoothingFunction().method1
        batch_size = logits.size(0)
        
        bleu_scores = []
        
        # Convert logits to predicted token indices
        predictions = torch.argmax(logits, dim=-1)  # shape: (batch_size, sequence_length)
        
        for i in range(batch_size):
            pred_text = self.tokenizer.decode(predictions[i], skip_special_tokens=True)
            target_text = self.tokenizer.decode(targets[i], skip_special_tokens=True)
            
            # Calculate BLEU score
            bleu_score = sentence_bleu(
                [target_text.split()], 
                pred_text.split(), 
                smoothing_function=smoothing, 
                weights=(0.5, 0.5)
            )
            bleu_scores.append(bleu_score)
        
        # Return the average BLEU score across the batch
        return sum(bleu_scores) / len(bleu_scores)

    def entropy_of_predictions(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1).mean()

    def top_k_accuracy(self, logits, targets, k=5):
        _, top_k_indices = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_indices)
        correct = (top_k_indices == targets_expanded).any(dim=-1).float()
        return correct.mean()


def read_wandb_id(file):
    with open(file, 'r') as file:
        # Read the content of the file
        content = file.read().strip()

        # Convert the content to an integer
        number = int(content.split(' ')[-1])
        return number


@contextmanager
def without_seed():
    # Save the current state
    state = random.getstate()
    
    try:
        # Temporarily ignore the seed (use system randomness)
        random.seed(None)
        yield
    finally:
        # Restore the original state
        random.setstate(state)

def write_wandb_id(file):
    with without_seed():
        id = random.randint(0, np.iinfo(np.int32).max)
    
    with open(file, 'a') as file:
        # Read the content of the file
        file.write(str(f"{id} "))
        file.close()
    return id
