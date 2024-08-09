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
        super(Blip2Classifier, self).__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space = config.answer_space

        self.interm_layer = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(config.interm_dim, len(self.answer_space))
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
            output_hidden_states=True,
        )

        # q-former_outputs 32 x 768
        batch_size = input_ids.size(0)

        # the total number of features is 24576
        features = outputs.qformer_outputs.last_hidden_state.view(batch_size, -1)

        # Classification
        interm_output = self.interm_layer(features)
        logits = self.classifier(interm_output)

        wandb.log({"Base Model Loss": outputs.loss.item()})
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs.loss = loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {loss}")
            wandb.log({"Classifier Loss": loss.item()})

        outputs.logits = logits

        return outputs
