import logging
import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel
from torch.nn import Module
from transformers import Blip2Config, PreTrainedModel

import wandb

from .feature_visualizer import FeatureVisualizer

logger = logging.getLogger(__name__)


class Blip2ClassifierConfig(Blip2Config):
    model_type = "blip-2"

    def __init__(
        self,
        dataset_name=None,
        answer_space=None,
        classification_input_dim=0,
        base_model_name="blip2",
        interm_dim=1024,
        multi_class_classifier=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.interm_dim = interm_dim
        self.classification_input_dim = classification_input_dim
        self.answer_space = answer_space
        self.dataset_name = dataset_name
        self.multi_class_classifier = multi_class_classifier


class Blip2BaseClassifier(PreTrainedModel):
    model: PreTrainedModel
    config_class: Blip2ClassifierConfig = Blip2ClassifierConfig()
    classifier: Module

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        self.embeddings: defaultdict[str, list] = defaultdict(list)

        self.answer_space = config.answer_space
        self.id_to_answer = {
            idx: answer for idx, answer in enumerate(self.answer_space)
        }

        self.feature_visualizer = FeatureVisualizer(
            self.id_to_answer, config.dataset_name
        )
        self.config = config
        self.model: PeftModel = peft_model
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space_dim = config.answer_space
        self.set_criterion()

    def save_pretrained(self, save_directory, **kwargs):
        output_path = f"{save_directory}"
        # Classification class config
        self.config.save_pretrained(output_path)

        # Save the PEFT model
        self.model.save_pretrained(output_path)

        # Save the additional layer
        additional_layer_path = os.path.join(output_path, "classification_layer.pt")
        torch.save(self.classifier.state_dict(), additional_layer_path)

    def save_statistics(self, output_path):
        features_path_train = os.path.join(output_path, "features_train.pkl")
        features_path_valid = os.path.join(output_path, "features_val.pkl")
        pickle.dump(
            self.feature_visualizer.get_features("train"),
            open(features_path_train, "wb"),
        )
        pickle.dump(
            self.feature_visualizer.get_features("val"), open(features_path_valid, "wb")
        )

    def reset_state(self, epoch, is_better):
        self.feature_visualizer.reset(epoch, is_better)

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        pretrained_model_name_or_path,
        adapter_name,
        is_trainable,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        peft_model = PeftModel.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
        )
        peft_model.print_trainable_parameters()

        model = cls(config, peft_model)

        # Load the additional layer
        additional_layer_path = os.path.join(
            pretrained_model_name_or_path, "classification_layer.pt"
        )
        if os.path.exists(additional_layer_path):
            model.classifier.load_state_dict(torch.load(additional_layer_path))

        return model

    def set_criterion(self):
        if self.config.multi_class_classifier:
            self.criterion = nn.CrossEntropyLoss()
        else:
            # self.criterion = FocalLoss()
            self.criterion = nn.BCEWithLogitsLoss()

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def log_losses(self, outputs, classifier_loss):
        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        wandb.log({"Classifier Batch Loss": classifier_loss.item()})
        logger.debug(
            f"Base model loss {outputs.loss} and classifier loss {classifier_loss}"
        )

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        raise NotImplementedError("Subclasses must implement forward method")

    def get_state(self):
        return self.embeddings


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        BCE_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss
