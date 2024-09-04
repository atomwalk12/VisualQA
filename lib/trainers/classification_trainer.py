import gc
import logging

import torch
import torch.utils.checkpoint
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig

import wandb
from config import Repositories

from ..daquar.daquar_classification import DaquarClassification
from ..easy_vqa.easyvqa_classification import EasyVQAClassification
from ..models.base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig
from ..models.blip2_classifier_experiment1 import Blip2ClassifierExperiment1
from ..models.blip2_classifier_experiment5 import Blip2ClassifierExperiment
from ..representations import ModelFactory, ModelTypes
from ..types import SAVE_PATHS, DatasetTypes, FileNames, Suffix, TrainingParameters
from ..utils import ClassificationMetricsAccumulator
from .base_trainer import TorchBase

logger = logging.getLogger(__name__)


class ClassificationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)
        plot_confusion_matrix = self.dataset_name == DatasetTypes.EASY_VQA

        self.train_accumulator = ClassificationMetricsAccumulator(
            self.dataset_name, self.answer_space, Suffix.Train, update_frequency=1, plot_confusion_matrix=plot_confusion_matrix
        )
        self.val_accumulator = ClassificationMetricsAccumulator(
            self.dataset_name, self.answer_space, Suffix.Val, update_frequency=1, plot_confusion_matrix=plot_confusion_matrix
        )
        self.state_update_frequency = 1
        self.multi_class_classifier = config.train_args.multi_class_classifier


    def get_repository(self):
        if self.dataset_name == DatasetTypes.EASY_VQA:
            return Repositories.VQAClassificationEasyVQA
        elif self.dataset_name == DatasetTypes.DAQUAR:
            return Repositories.VQAClassificationDaquar

    def get_save_path(self):
        if self.dataset_name == DatasetTypes.DAQUAR:
            return SAVE_PATHS.BLIP2_Classifier_DAQUAR
        elif self.dataset_name == DatasetTypes.EASY_VQA:
            return SAVE_PATHS.BLIP2_Classifier_EasyVQA

    def load_from_checkpoint(self, is_trainable):
        base_model = self.get_models(apply_lora=False)

        base_model = ModelFactory.prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

        local_model_path = self.best_path

        if (
            self.model_name == ModelTypes.BLIP2Classifier
            or self.model_name == ModelTypes.BLIP2FinetunedClassifier
        ):
            model = Blip2ClassifierExperiment.from_pretrained(
                base_model,
                local_model_path,
                adapter_name="vqa_classification",
                is_trainable=is_trainable,
            )
        elif self.model_name == ModelTypes.BLIP2FinetunedBaseClassifier:
            model = Blip2BaseClassifier.from_pretrained(
                base_model,
                local_model_path,
                adapter_name="vqa_base_classification",
                is_trainable=is_trainable,
            )
        else:
            raise KeyError()

        return model

    def bootstrap_model(self, answer_space):
        model_name = self.model_name
        # Load the model and processor
        model = self.get_models(apply_lora=True)

        if model_name == ModelTypes.BLIP2FinetunedBaseClassifier:
            config = Blip2ClassifierConfig(
                classification_input_dim=5120,
                answer_space=answer_space,
                dataset_name=self.dataset_name,
                multi_class_classifier=self.config.train_args.multi_class_classifier
            )
            model = Blip2BaseClassifier(config, model)
        elif (
            model_name == ModelTypes.BLIP2Classifier
            or model_name == ModelTypes.BLIP2FinetunedClassifier
        ):
            config = Blip2ClassifierConfig(
                classification_input_dim=768,
                answer_space=answer_space,
                dataset_name=self.dataset_name,
                multi_class_classifier=self.config.train_args.multi_class_classifier
            )
            model = Blip2ClassifierExperiment(config, model)

        return model

    def test(self):
        self.model.eval()
        answer_space = self.test_dataloader.dataset.answer_space
        history = self.state.history

        dataset_size = 0
        running_similarity = 0.0
        with torch.no_grad():
            bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
            for step, data in bar:
                # Unpack the collator values
                pixel_values = data["pixel_values"].to(self.device)
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                _, preds = torch.max(outputs.logits, 1)
                _, target_pred = torch.max(labels, 1)

                for sample, label in zip(preds, target_pred):
                    predicted = answer_space[sample.item()]
                    target = answer_space[label.item()]

                    similarity = self.sbert_similarity(predicted, target)
                    running_similarity += similarity
                    dataset_size += 1

                best_epoch_loss = running_similarity / dataset_size

                history["SBert Loss"].append(best_epoch_loss)
                wandb.log({"SBert Similarity": best_epoch_loss})

                bar.set_postfix(Batch=step, Test_Loss=best_epoch_loss)

                # Save the state
                if step % self.state_update_frequency == 0:
                    self.save_trainer_state(
                        best_epoch_loss,
                        history,
                        step + 1,
                        self.test_dataloader,
                    )

        # Save the entire results
        self.save_trainer_state(
            best_epoch_loss, history, step + 1, self.test_dataloader
        )

        # Release resources
        gc.collect()
        del self.test_dataloader
        self.run.finish()
        wandb.finish()

        return history

    def send_to_device_if_needed(self, data):
        input_ids = data["input_ids"].to(self.device)
        pixel_values = data["pixel_values"].to(self.device)
        labels = data["labels"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        return input_ids, pixel_values, attention_mask, labels

    def get_dataset(self, args):
        if self.dataset_name == DatasetTypes.EASY_VQA:
            return EasyVQAClassification(args)
        elif self.dataset_name == DatasetTypes.DAQUAR:
            return DaquarClassification(args)

    def update_state_with_embeddings(self, embeddings):
        # This is done automatically within the model's state. Nothing to do.
        pass

    def save_trainer_state(
        self, best_epoch_loss, history, epoch, dataloader: DataLoader
    ):  # noqa: F821
        # This should always be true, but checking for intellisense completion.
        assert isinstance(self.model, Blip2BaseClassifier)

        # retrieve model's state and save to file
        self.state.save_state(
            self.best_path,
            best_epoch_loss,
            history,
            epoch,
            self.scheduler,
            self.optimizer,
            dataloader.dataset,
        )

    def get_models(self, apply_lora):
        return ModelFactory.get_models(
            self.model_name,
            apply_lora=apply_lora,
            lora_config=self.lora,
            bnb_config=self.bnb,
            torch_dtype=torch.float32,
        )

    def bnb_config(self) -> BitsAndBytesConfig:
        """Create a BitsAndBytesConfig for quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )

    def lora_config(self) -> LoraConfig:
        """Create a LoraConfig for PEFT."""
        if self.dataset_name == DatasetTypes.DAQUAR:
            return LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
        elif self.dataset_name == DatasetTypes.EASY_VQA:
            return LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )

    def on_batch_processed(self, y_pred, y_true):
        if self.multi_class_classifier:
            if self.model.training:
                self.train_accumulator.log_multi_class_statistics(y_pred, y_true)
            else:
                self.val_accumulator.log_multi_class_statistics(y_pred, y_true)
        else:
            if self.model.training:
                self.train_accumulator.log_multi_label_statistics(y_pred, y_true)
            else:
                self.val_accumulator.log_multi_label_statistics(y_pred, y_true)

    def generate_local_confusion_matrix(self, y_pred, y_true):
        # Get the predicted class with the highest probability
        prediction = torch.argmax(y_pred.logits, dim=1)

        for pred, label in zip(prediction, y_true):
            # Check if the predicted class is one of the correct classes
            if label[pred] == 1:
                self.state.history["confusion_predictions"].append(pred.item())
                self.state.history["confusion_labels"].append(pred.item())
            else:
                self.state.history["confusion_predictions"].append(pred.item())

                # Get all indices where label is 1
                true_label = torch.nonzero(label).flatten().tolist()

                # Default to -1 if no label is found
                chosen_label = next(iter(true_label), -1)
                self.state.history["confusion_labels"].append(chosen_label)

    def on_best_epoch(self):
        pass

    def save_confusion_matrix_state(self):
        self.state.save_state_to_file(
            self.best_path,
            file_name=FileNames.ConfusionMatrix.format(self.config.split),
        )

        self.state.history["confusion_predictions"] = []
        self.state.history["confusion_labels"] = []

    def on_epoch_end(self):
        if self.multi_class_classifier:
            self.train_accumulator.log_confusion_matrix()
            self.val_accumulator.log_confusion_matrix()
            self.train_accumulator.report_multi_class_statistics()
            self.val_accumulator.report_multi_class_statistics()
        else:
            self.train_accumulator.report_multi_label_statistics()
            self.val_accumulator.report_multi_label_statistics()
