import logging

import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .types import ModuleConfig

logger = logging.getLogger(__name__)


class BLIP2PLModule(L.LightningModule):
    def __init__(self, config: ModuleConfig):
        super().__init__()
        logger.info(config)

        self.config = config
        self.processor = self.config.processor

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

        result = {}
        result["inputs"] = inputs
        result["labels"] = labels
        return result
