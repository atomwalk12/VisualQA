import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel

import wandb
from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig
from lib.types import DatasetTypes


logger = logging.getLogger(__name__)


class Blip2ClassifierExperiment1(Blip2BaseClassifier):
    """
    This is the first classifier I used. It is a simple MLP classifier on-top of the base model.
    Uses the last hidden state of the Q-Former and the last_hidden_state of the vision encoder.
    Proceeds to use the mean across the first dimension of the q-former to make the features'
    dimensions homogenous.
    """

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config, peft_model)
        logger.info(f"{config.interm_dim=}")

        # Define dimensions for the combined features
        combined_dim = (
            config.vision_config.hidden_size + config.qformer_config.hidden_size
        )

        # TODO[RF] different networks
        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 1536),  # 2176 -> 1536
                nn.ReLU(),
                nn.BatchNorm1d(1536),
                nn.Dropout(0.4),
                nn.Linear(1536, 1024),  # 1536 -> 1024
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),  # 1024 -> 512
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, len(self.answer_space)),  # 512 -> 12
            )
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.classifier = nn.Sequential(
                nn.Linear(
                    combined_dim, 1024
                ),  # Reduced layer count: 2176 -> 1024 directly
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, len(self.answer_space)),
            )

        self.classifier.apply(self.initialize_weights)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        extract_features: bool = False,
    ):
        # Extract image features
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        # Get vision features
        vision_pooled = outputs.vision_outputs.last_hidden_state.mean(dim=1)

        # Get Q-Former features and pool them
        qformer_pooled = outputs.qformer_outputs["last_hidden_state"].mean(dim=1)

        # Combine vision and Q-Former features
        combined_features = torch.cat((vision_pooled, qformer_pooled), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        if extract_features:
            # Accumulate features and labels
            self.feature_visualizer.accumulate_features(combined_features, labels)
        else:
            self.feature_visualizer.visualize_features_with_umap()

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

