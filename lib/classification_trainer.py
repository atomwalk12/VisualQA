import gc
import logging

import torch
import torch.utils.checkpoint
from peft import prepare_model_for_kbit_training

# PyTorch TensorBoard support
from tqdm import tqdm

import wandb
from config import Repositories
from lib.base_trainer import TorchBase
from lib.representations import SAVE_PATHS, ModelFactory, ModelTypes

from .datasets_qa.easyvqa_classification import EasyVQAClassification
from .models.base_classifier import Blip2ClassifierConfig
from .models.blip2_base_classifier import Blip2BaseClassifier
from .models.blip2_classifier import Blip2Classifier
from .types import TrainingParameters

logger = logging.getLogger(__name__)


class ClassificationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)

    def get_repository(self):
        if self.model_name == ModelTypes.BLIP2BaseClassifier:
            return Repositories.VQAClassificationBase
        else:
            return Repositories.VQAClassification

    def get_save_path(self):
        model_name = self.model_name
        if model_name == ModelTypes.BLIP2BaseClassifier:
            return SAVE_PATHS.BLIP2_BaseClassifier
        elif model_name == ModelTypes.BLIP2Classifier:
            return SAVE_PATHS.BLIP2_Classifier

    def load_from_checkpoint(self, is_trainable):
        base_model, processor = ModelFactory.get_models(self.model_name, apply_lora=False)

        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

        local_model_path = self.get_model_save_path()
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

    def get_model_save_path(self):
        if self.model_name == ModelTypes.BLIP2BaseClassifier:
            return SAVE_PATHS.BLIP2_BaseClassifier
        elif self.model_name == ModelTypes.BLIP2Classifier:
            return SAVE_PATHS.BLIP2_Classifier

    def bootstrap_model(self):
        model_name = self.model_name
        # Load the model and processor
        model, processor = ModelFactory.get_models(model_name, apply_lora=True)

        if model_name == ModelTypes.BLIP2BaseClassifier:
            config = Blip2ClassifierConfig(classification_input_dim=5120)
            model = Blip2BaseClassifier(config, model)
        elif model_name == ModelTypes.BLIP2Classifier:
            config = Blip2ClassifierConfig(classification_input_dim=24576)
            model = Blip2Classifier(config, model)

        return model, processor

    def test(self):
        self.model.eval()
        answer_space = self.model.answer_space
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

                batch_sim = running_similarity / dataset_size

                history["Test Loss"].append(batch_sim)
                wandb.log({"Test Loss": batch_sim})

                bar.set_postfix(Batch=step, Test_Loss=batch_sim)

        # save the state
        self.state.save_state(
            self.best_path,
            None,
            history,
            len(self.test_dataloader),
            self.scheduler,
            self.optimizer,
            f"{self.model_name}_test_results.pkl",
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
        return EasyVQAClassification(args)