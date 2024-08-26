import logging
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel
from torch.nn import Module
from transformers import Blip2Config, PreTrainedModel

import wandb
from lib.types import DatasetTypes


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.interm_dim = interm_dim
        self.classification_input_dim = classification_input_dim
        self.answer_space = answer_space
        self.dataset_name = dataset_name


class Blip2BaseClassifier(PreTrainedModel):
    model: PreTrainedModel
    config_class: Blip2ClassifierConfig
    classifier: Module

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        self.embeddings: defaultdict[str, list] = defaultdict(list)

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
            pretrained_model_name_or_path, "classifier_layer.pt"
    )
        if os.path.exists(additional_layer_path):
            model.classifier.load_state_dict(torch.load(additional_layer_path))

        return model

    def set_criterion(self):
        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError(f"Unsupported dataset: {self.config.dataset_name}")

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
