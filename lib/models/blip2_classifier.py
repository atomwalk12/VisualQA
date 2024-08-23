import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel

from transformers import Blip2Config

import wandb

from lib.types import DatasetTypes

from .base_classifier import Blip2, Blip2ClassifierConfig

logger = logging.getLogger(__name__)

class Blip2Classifier3(Blip2):
    config_class = Blip2ClassifierConfig

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config)
        logger.info(f"{config.interm_dim=}")

        self.config = config
        self.model: PeftModel = peft_model

        # Define dimensions for the combined features
        combined_dim = config.vision_config.hidden_size + config.qformer_config.hidden_size
        # 1408 + 2560
        # Fusion and final classification
        self.peft_config: Blip2Config = peft_model.peft_config
        self.answer_space_dim = config.answer_space_dim
        
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
                nn.Linear(512, self.answer_space_dim),  # 512 -> 12
            )
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 1536),  # 2176 -> 1536
                nn.ReLU(),
                nn.BatchNorm1d(1536),
                nn.Dropout(0.4),
                nn.Linear(1536, 1024),  # 1536 -> 1024
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.4),
                nn.Linear(1024, self.answer_space_dim),  # 1024 -> 582
            )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()

        self.classifier.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


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
        # Get vision features
        vision_pooled = outputs.vision_outputs.last_hidden_state.mean(dim=1) # Adjust as needed
        # Get Q-Former features and pool them
        last_hidden_state = outputs.qformer_outputs["last_hidden_state"]
        qformer_pooled = last_hidden_state.mean(dim=1)  # Example using mean pooling

        # Combine vision and Q-Former features
        combined_features = torch.cat((vision_pooled, qformer_pooled), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs

class Blip2Classifier2(Blip2):
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
        
        # Layer Normalization
        self.qformer_norm = nn.LayerNorm(config.qformer_config.hidden_size) # 768 
        self.vit_norm = nn.LayerNorm(config.vision_config.hidden_size) # 1408
        
        self.qformer_proj = nn.Linear(config.qformer_config.hidden_size, 512)
        self.vit_proj = nn.Linear(config.vision_config.hidden_size, 1024)

        self.classifier = nn.Sequential(
            nn.Linear(512+1024, config.interm_dim),  # 32 x 768
            # nn.BatchNorm1d(config.interm_dim),  # BatchNorm before ReLU
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()



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
        
        # Concatenate features
        combined_features = torch.cat((qformer_features, vit_features), dim=-1)

        # Classification
        logits = self.classifier(combined_features)

        wandb.log({"Base Model Batch Loss": outputs.loss.item()})
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            combined_loss = outputs.loss + classifier_loss
            outputs.loss = combined_loss

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs

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

        self.classifier = nn.Sequential(
            nn.Linear(config.classification_input_dim, config.interm_dim),  # 32 x 768
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.interm_dim, self.answer_space_dim),
        )

        if self.config.dataset_name == DatasetTypes.EASY_VQA:
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.dataset_name == DatasetTypes.DAQUAR:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise KeyError()

        # self.classifier.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)



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

            logger.debug(f"Base model loss {outputs.loss} and classifier loss {classifier_loss}")
            wandb.log({"Classifier Batch Loss": classifier_loss.item()})

        outputs.logits = logits

        return outputs
