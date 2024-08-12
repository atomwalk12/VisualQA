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

from ..datasets_qa.daquar.daquar_classification import DaquarClassification
from ..datasets_qa.easyvqa_classification import EasyVQAClassification
from ..models.base_classifier import Blip2, Blip2ClassifierConfig
from ..models.blip2_base_classifier import Blip2BaseClassifier
from ..models.blip2_classifier import Blip2Classifier
from ..representations import ModelFactory, ModelTypes
from ..types import SAVE_PATHS, DatasetTypes, TrainingParameters
from .base_trainer import TorchBase

logger = logging.getLogger(__name__)


class ClassificationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)

        self.update_frequency = 64

    def get_repository(self):
        if self.model_name == ModelTypes.BLIP2BaseClassifier:
            return Repositories.VQAClassificationBase
        else:
            return Repositories.VQAClassification

    def get_save_path(self):
        if self.dataset_name == DatasetTypes.DAQUAR:
            return SAVE_PATHS.BLIP2_Classifier_DAQUAR
        elif self.dataset_name == DatasetTypes.EASY_VQA:
            return SAVE_PATHS.BLIP2_Classifier_EasyVQA

    def load_from_checkpoint(self, is_trainable):
        base_model, processor = self.get_models(apply_lora=False)

        base_model = ModelFactory.prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

        local_model_path = self.best_path
        if self.model_name == ModelTypes.BLIP2Classifier:
            model = Blip2Classifier.from_pretrained(
                base_model,
                local_model_path,
                adapter_name="vqa_classification",
                is_trainable=is_trainable,
            )
        elif self.model_name == ModelTypes.BLIP2BaseClassifier:
            model = Blip2BaseClassifier.from_pretrained(
                base_model,
                local_model_path,
                adapter_name="vqa_base_classification",
                is_trainable=is_trainable,
            )
        else:
            raise KeyError()

        return model, processor


    def bootstrap_model(self):
        model_name = self.model_name
        # Load the model and processor
        model, processor = self.get_models(apply_lora=True)
        answer_space_dim = 13 if self.config.dataset_name == DatasetTypes.EASY_VQA else 582

        if model_name == ModelTypes.BLIP2BaseClassifier:
            config = Blip2ClassifierConfig(
                classification_input_dim=5120, save_embeddings=self.save_embeddings, answer_space_dim=answer_space_dim
            )
            model = Blip2BaseClassifier(config, model)
        elif model_name == ModelTypes.BLIP2Classifier:
            config = Blip2ClassifierConfig(
                classification_input_dim=768, save_embeddings=self.save_embeddings, answer_space_dim=answer_space_dim
            )
            model = Blip2Classifier(config, model)

        return model, processor

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

                history["Test Loss"].append(best_epoch_loss)
                wandb.log({"Test Loss": best_epoch_loss})

                bar.set_postfix(Batch=step, Test_Loss=best_epoch_loss)

                # Save the state
                if step % self.update_frequency == 0:
                    self.save_state(
                        best_epoch_loss,
                        history,
                        step + 1,
                        self.test_dataloader,
                    )

        # Save the entire results
        self.save_state(best_epoch_loss, history, step + 1, self.test_dataloader)

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

    def save_state(self, best_epoch_loss, history, epoch, dataloader: DataLoader):  # noqa: F821
        # This should always be true, but checking for intellisense completion.
        assert isinstance(self.model, Blip2)
        if self.save_embeddings:
            embeddings = self.model.get_state()
            history["embeddings"] = embeddings

        # retrieve model's state and save to file
        self.state.save_state(
            self.best_path, best_epoch_loss, history, epoch, self.scheduler, self.optimizer, dataloader.dataset
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
            bnb_4bit_compute_dtype=torch.float16,
        )

    def lora_config(self) -> LoraConfig:
        """Create a LoraConfig for PEFT."""
        return LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
