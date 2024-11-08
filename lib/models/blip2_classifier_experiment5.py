import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.peft_model import PeftModel
from transformers import Blip2Model, Blip2QFormerConfig

import wandb

from ..types import Blip2ClassificationModelOutput
from .base_classifier import Blip2BaseClassifier, Blip2ClassifierConfig

logger = logging.getLogger(__name__)


class Blip2TextEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config: Blip2QFormerConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if input_ids is not None:
            input_ids = input_ids.to(self.word_embeddings.weight.device)
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        return embeddings


class Blip2ClassifierExperiment(Blip2BaseClassifier):
    """This is the last classifier I used. It is a MLP classifier, however the logic
    behind the retrieval of the embeddings from the model is slightly more involved.
    Briefly, the code follows these steps:
    1. Pass the pixel values through the vision model to get the image embeddings.
    2. Pass the image embeddings through the q-former to get the query-conditioned image embeddings.
    3. Pass the input ids through the text model to get the text embeddings.
    4. Concatenate the text embeddings and the query-conditioned image embeddings.
    5. Pass the concatenated embeddings through the classifier to get the logits.
    The code was adapted from:
    https://github.com/huggingface/transformers/blob/0d86727354ad8b1ff23ab66380e949fe0d842590/src/transformers/models/blip_2/modeling_blip_2.py#L2392
    """

    def __init__(self, config: Blip2ClassifierConfig, peft_model: PeftModel):
        super().__init__(config, peft_model)
        logger.info(f"{config.interm_dim=}")

        self.model: Blip2Model = peft_model
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.vision_projection = nn.Linear(config.qformer_config.hidden_size, 256)
        self.text_projection = nn.Linear(config.qformer_config.hidden_size, 256)
        self.classifier = nn.Sequential(
            nn.Linear(8448, 5012),
            nn.ReLU(),
            nn.BatchNorm1d(5012),
            nn.Dropout(0.5),
            nn.Linear(5012, 2056),
            nn.ReLU(),
            nn.BatchNorm1d(2056),
            nn.Dropout(0.5),
            nn.Linear(2056, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, len(self.config.answer_space)),  # 32 x 256
        )

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.text_embeddings = Blip2TextEmbeddings(config.qformer_config)

        self.id_to_answer = {i: answer for i, answer in enumerate(self.config.answer_space)}
        self.post_init()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        extract_features: str = "train",
        log: bool = True,
    ):
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        image_embeds = query_outputs.last_hidden_state

        query_embeds = self.text_embeddings(
            input_ids=input_ids,
        )

        text_outputs = self.model.qformer(
            query_embeds=query_embeds, query_length=0, attention_mask=attention_mask
        )
        question_embeds = text_outputs.last_hidden_state

        # normalized features
        text_repr = F.normalize(
            self.text_projection(question_embeds[:, 0, :]), dim=-1
        )  # [batch_size, 256]
        image_repr = F.normalize(
            self.vision_projection(image_embeds), dim=-1
        )  # [batch_size, 32, 256]

        # Flatten image representation
        image_repr_flat = image_repr.view(image_repr.size(0), -1)  # [batch_size, 8192]

        # Combine features for classification
        combined_features = torch.cat(
            (text_repr, image_repr_flat), dim=-1
        )  # [batch_size, 8448]

        # Classification
        logits = self.classifier(combined_features)

        if extract_features == "train":
            self.feature_visualizer.accumulate_features(logits, labels, "train")
        elif extract_features == "val":
            self.feature_visualizer.accumulate_features(logits, labels, "val")
            
        if labels is not None:
            classifier_loss = self.criterion(logits, labels)
            total_loss = classifier_loss

            if log:
                wandb.log(
                    {
                    "Classifier Loss": classifier_loss.item(),
                }
            )
        else:
            total_loss = None

        return Blip2ClassificationModelOutput(
            loss=total_loss,
            logits=logits,
            text_embeds=text_repr,
            image_embeds=image_repr,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
