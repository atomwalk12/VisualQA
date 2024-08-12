import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel

from transformers import Blip2Config

import wandb

from .base_classifier import Blip2, Blip2ClassifierConfig

logger = logging.getLogger(__name__)


class Blip2Classifier(Blip2):
    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space_dim = config.answer_space_dim

        self.interm_layer = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(config.interm_dim, self.answer_space_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Extract image features
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        features = outputs.qformer_outputs["pooler_output"]

        # Classification
        interm_output = self.interm_layer(features)
        logits = self.classifier(interm_output)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs.loss = loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {loss}")
            wandb.log({"Classifier Batch Loss": loss.item()})

        outputs.logits = logits

        return outputs
