import logging

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .representations import ModelFactory
from .types import LightningConfig, ModuleConfig

logger = logging.getLogger(__name__)


class BLIP2PLModule(L.LightningModule):
    def __init__(self, config: ModuleConfig):
        super().__init__()
        logger.info(config)

        self.config = config
        self.processor = config.processor
        self.model = config.model

    def train_dataloader(self):
        assert self.config.train_dataset is not None
        return DataLoader(
            self.config.train_dataset,
            collate_fn=lambda batch: self.train_collate_fn(
                batch, self.processor, self.config
            ),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=0,
        )

    def val_dataloader(self):
        assert self.config.val_dataset is not None
        return DataLoader(
            self.config.val_dataset,
            collate_fn=lambda batch: self.eval_collate_fn(
                batch, self.processor, self.config
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @staticmethod
    def train_collate_fn(batch: dict, processor: AutoProcessor, config: ModuleConfig):
        images = []
        texts = []
        for item in batch:
            texts.append(item.get("prompt"))
            images.append(item.get("image"))

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

        inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    @staticmethod
    def eval_collate_fn(batch: dict, processor: AutoProcessor, config: ModuleConfig):
        images = []
        texts = []
        labels = []
        for item in batch:
            texts.append(item.get("prompt"))
            images.append(item.get("image"))
            labels.append(item.get("label"))

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

        # TODO(Razvan) ATTENTION! Here I am not tokenizing the labels!
        result = {}
        result["inputs"] = inputs
        result["labels"] = labels
        return result

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        error = loss.item()
        logger.info(f"Epoch {self.current_epoch}, loss: {error}")

        self.log("train_loss", error, batch_size=self.config.batch_size)

        return loss


class LightningFineTune:
    def __init__(self, config: LightningConfig):
        # Set up configuration parameters
        self.config = config

    @staticmethod
    def create_module(model_name, train_ds, val_ds):
        model, processor = ModelFactory.get_models(model_name)

        config = ModuleConfig(
            train_dataset=train_ds,
            val_dataset=val_ds,
            processor=processor,
            model=model,
            shuffle_train=True,
        )

        module = BLIP2PLModule(config)
        return module

    def finetune(self, module: L.LightningModule):
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=self.config.max_epochs,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            check_val_every_n_epoch=self.config.check_val_every_n_epochs,
            gradient_clip_val=self.config.gradient_clip_val,
            precision="16-mixed",
            limit_val_batches=self.config.limit_val_batches,
            num_sanity_val_steps=0,
        )

        trainer.fit(module)
