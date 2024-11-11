import logging
from typing import Optional

import torch
import torch.nn as nn
import wandb
from peft.peft_model import PeftModel
from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig


logger = logging.getLogger(__name__)


class Blip2ClassifierExperiment2(Blip2BaseClassifier):
    """
    This is the second classifier I used. It is still a MLP classifier, however the
    output is projected using two intermediate linear layers to make the input homogenous.
    This contrasts with the Experiment 1 which uses a mean average to make the features
    compatible.
    """

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config, peft_model)
        logger.info(f"{config.interm_dim=}")

        # 1408 + 2560
        # Fusion and final classification
        self.qformer_norm = nn.LayerNorm(config.qformer_config.hidden_size)  # 768
        self.vit_norm = nn.LayerNorm(config.vision_config.hidden_size)  # 1408

        self.qformer_proj = nn.Linear(config.qformer_config.hidden_size, 256)
        self.vit_proj = nn.Linear(config.vision_config.hidden_size, 256)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )

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

        # Process QFormer features
        qformer_features = outputs.qformer_outputs.last_hidden_state
        text_embeds = self.text_projection(qformer_features)
        text_embeds = nn.functional.normalize(text_embeds, dim=-1)

        # Process Vision features
        pooled_output = outputs.vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            pooled_output.size()[:-1], dtype=torch.long, device=pooled_output.device
        )

        query_tokens = self.query_tokens.expand(pooled_output.shape[0], -1, -1)

        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=pooled_output,
            encoder_attention_mask=image_attention_mask,
        )

        embeds = query_outputs.last_hidden_state
        image_embeds = self.vision_projection(embeds)
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)

        # Normalize features
        combined_features = torch.cat((text_embeds, image_embeds), dim=1)

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
