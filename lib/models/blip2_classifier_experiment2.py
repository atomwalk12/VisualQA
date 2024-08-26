import logging
from typing import Optional

import torch
import torch.nn as nn
import wandb
from peft.peft_model import PeftModel
from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig


logger = logging.getLogger(__name__)

class Blip2ClassifierExperiment2(Blip2BaseClassifier):
    """This is the second classifier I used. It is still a simple MLP classifier, however the 
    output is projected using two intermediate linear to make the input homogenous. Also, the
    Blip2ForConditionalGeneration model is used to get the features. See:
    https://huggingface.co/docs/transformers/model_doc/blip-2#transformers.Blip2ForConditionalGeneration

    Args:
        BaseBlip2Classifier: Configuration storing information about intermediary
        dimensions and answer space.
    """

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config, peft_model)
        logger.info(f"{config.interm_dim=}")

        # 1408 + 2560
        # Fusion and final classification
        self.qformer_norm = nn.LayerNorm(config.qformer_config.hidden_size)  # 768
        self.vit_norm = nn.LayerNorm(config.vision_config.hidden_size)  # 1408

        self.qformer_proj = nn.Linear(config.qformer_config.hidden_size, 512)
        self.vit_proj = nn.Linear(config.vision_config.hidden_size, 1024)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024, config.interm_dim),  # 32 x 768
            # nn.BatchNorm1d(config.interm_dim),  # BatchNorm before ReLU
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

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
        vit_features = outputs.vision_outputs.last_hidden_state.mean(dim=1)
        qformer_features = outputs.qformer_outputs["last_hidden_state"].mean(dim=1)
        # Normalize features
        qformer_features = self.qformer_norm(qformer_features)
        vit_features = self.vit_norm(vit_features)

        # Project features to same dimension
        qformer_features = self.qformer_proj(qformer_features)
        vit_features = self.vit_proj(vit_features)

        # concatenate features
        combined_features = torch.cat((qformer_features, vit_features), dim=-1)

        # Classification
        logits = self.classifier(combined_features)

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


