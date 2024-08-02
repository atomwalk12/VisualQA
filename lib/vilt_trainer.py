import logging
from datetime import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from easy_vqa import get_answers
from torch.utils.data import DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

from lib.lightning_trainer import BLIP2PLModule
from lib.types import ModuleConfig

from .representations import (
    DatasetTypes,
    ModelTypes,
    ModuleConfigGenerator,
    load_evaluation_metrics,
)

import torch.utils.checkpoint
import numpy as np

logger = logging.getLogger(__name__)




class ClassificationTrainer:
    def __init__(self, config):
        # Set up configuration parameters
        self.config = config
        self.base_model = config.model
        self.model = Blip2Classifier(self.base_model, len(get_answers()))

        self.train_dataset = config.train_dataset
        self.batch_size = config.batch_size
        self.train_dataloader = DataLoader(
            self.train_dataset,
            collate_fn=lambda batch: BLIP2PLModule.train_collate_fn(
                batch, config.processor, self.config
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            self.config.val_dataset,
            collate_fn=lambda batch: BLIP2PLModule.eval_collate_fn(
                batch, config.processor, self.config
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.metric = load_evaluation_metrics(
            model=ModelTypes.BLIP2, dataset=DatasetTypes.EASY_VQA
        )
        self.processor = config.processor
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )

        # Cross entropy loss
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        self.model.train(True)
        for i, data in enumerate(self.train_dataloader):
            # Every data instance is an input + label pair
            inputs = data
            labels = data["labels"]

            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(**inputs)

            # Compute the loss and its gradients

            loss = self.loss_fn(outputs, labels)

            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 5 == 4:
                last_loss = running_loss / 5  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_dataloader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def evaluate(self):
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        self.model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(self.train_dataloader):
                vinputs = vdata
                vlabels = vdata["labels"]
                voutputs = self.model(**vinputs)

                # Calculate cross entropy loss
                vloss = voutputs.loss

                # vloss = self.metric.compute(voutputs, vlabels, lang="en")
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        return avg_vloss



class Blip2Classifier(nn.Module):
    def __init__(self, base_model, num_classes=13):
        super().__init__()
        # ... (your existing model layers)
        self.base_model = base_model

        # Add the classification head
        self.classifier = nn.Linear(50272, num_classes)

    def forward(self, **inputs):
        outputs = self.base_model(**inputs)
        x = self.classifier(outputs.logits)
        return x
