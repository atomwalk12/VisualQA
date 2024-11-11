import logging
from typing import Optional

import torch
import torch.nn as nn
import wandb
from peft.peft_model import PeftModel
from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig

logger = logging.getLogger(__name__)


class Blip2ClassifierExperiment3(Blip2BaseClassifier):
    """This is the third classifier I used. It is a MLP classifier however the pooler_output
    of the qformer are being used instead of the last hidden state.
    """

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config, peft_model)
        logger.info(f"{config.interm_dim=}")

        self.classifier = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

        self.classifier.apply(self.initialize_weights)

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
        logits = self.classifier(features)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(
                f"Base model loss {outputs.loss} and classifier loss {classifier_loss}"
            )
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs
