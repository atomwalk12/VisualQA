import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel

from transformers import Blip2Config

import wandb

from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig

logger = logging.getLogger(__name__)


class Result:
    loss = None
    logits = None


class Blip2ClassifierExperiment4(Blip2BaseClassifier):
    """This is the fourth classifier I used. The main idea is that I did not use the
    Blip2ForConditionalGeneration model to get the features. Instead, I use the Blip2
    model's features, a model not conceived for VQA (it does not contain the LLM).
    """

    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space = config.answer_space

        self.classifier = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, len(self.answer_space)),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Extract image features
        language_features, qformer_features = self.get_img_embedding(pixel_values)

        # the total number of features is 24576
        text_features = self.get_text_embedding(input_ids)
        features = torch.cat((language_features, text_features), dim=1)

        # Classification
        logits = self.classifier(features)

        outputs = Result()
        if labels is not None:
            loss = self.criterion(logits, labels)
            outputs.loss = loss

            logger.debug(f"Base model loss {outputs.loss}")
            wandb.log({"Classifier Loss": loss.item()})

        outputs.logits = logits

        return outputs

    def get_img_embedding(self, images):
        # pass images through the vision model and then the qformer to get query-conditional image features
        # tuple (last_hidden_state, pooler_output)
        qformer_features = self.model.get_qformer_features(images)
        query_output = qformer_features["pooler_output"]  # (batch_size, hidden_size)

        # project query-conditional image features into language space
        # shape (batch_size, hidden_size)
        language_projections = self.model.language_projection(query_output)
        # TODO[RV] image_features /= image_features.norm(dim=-1, keepdim=True)

        return language_projections, qformer_features

    def get_text_embedding(self, texts):
        text_outputs = self.model.get_text_features(texts, output_hidden_states=True)

        # extract [CLS] embedding from last hidden state, shape (batch_size, hidden_size)
        text_features = text_outputs["hidden_states"][-1][:, 0, :]

        # TODO[RV] text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
