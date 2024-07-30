import logging

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from .representations import DatasetFactory, ModelFactory, load_evaluation_metrics
from .types import LightningConfig, ModuleConfig
from .utils import likely_pickle_dir

logger = logging.getLogger(__name__)


class BLIP2PLModule(L.LightningModule):
    def __init__(self, config: ModuleConfig):
        super().__init__()
        logger.info(config)

        self.config = config
        self.processor = config.processor
        self.model = config.model
        self.batch_size = config.batch_size
        self.metrics = config.metrics

    def train_dataloader(self):
        assert self.config.train_dataset is not None
        # TODO Possible bottleneck if num_workers=0, use num_workers=31
        return DataLoader(
            self.config.train_dataset,
            collate_fn=lambda batch: self.train_collate_fn(
                batch, self.processor, self.config
            ),
            batch_size=self.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=0,
        )

    def val_dataloader(self):
        assert self.config.val_dataset is not None
        # TODO Possible bottleneck if num_workers=0, use num_workers=31
        return DataLoader(
            self.config.val_dataset,
            collate_fn=lambda batch: self.eval_collate_fn(
                batch, self.processor, self.config
            ),
            batch_size=self.batch_size,
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
        return inputs, labels

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        error = loss.item()
        logger.info(f"Epoch {self.current_epoch}, loss: {error}")

        self.log("train_loss", error, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        inputs, labels = batch

        # Generate predictions
        generated_ids = self.model.generate(**inputs)
        predictions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Log batch information
        self._log_batch_info(batch_idx, predictions, labels)

        # Compute evaluation metrics
        for metric in self.metrics:
            scores = metric.compute(pred=predictions, references=labels)
            for k in metric.log_columns:
                self.log(f"{metric.name}_{k}", np.mean(scores[k]), batch_size=self.batch_size)

        return scores

    def _log_batch_info(self, batch_idx, predictions, labels):
        total_batches = len(self.config.val_dataset) // self.batch_size
        logging.info(f"Batch: {batch_idx}/{total_batches}")

        for i, (pred, answer) in enumerate(zip(predictions, labels)):
            element = self.config.val_dataset[batch_idx * self.batch_size + i]
            logging.info(f"Question: {element['prompt']}")
            logging.info(f"Answer: {answer}")
            logging.info(f"Prediction: {pred}")


class LightningFineTune:
    def __init__(self, config: LightningConfig):
        # Set up configuration parameters
        self.config = config

    @staticmethod
    def create_module(
        model_name: str,
        ds_name: str,
        train_args: dict,
        val_args: dict,
        pickle_dir: str = None,
        apply_qlora=True,
    ):
        model, processor = ModelFactory.get_models(model_name, apply_qlora=apply_qlora)

        train_ds, val_ds = DatasetFactory.create_dataset(ds_name, train_args, val_args)
        pickle_dir = pickle_dir or likely_pickle_dir(ds_name)

        train_ds = train_ds.load(pickle_dir)
        val_ds = val_ds.load(pickle_dir)

        config = ModuleConfig(
            train_dataset=train_ds,
            val_dataset=val_ds,
            processor=processor,
            model=model,
            shuffle_train=True,
            metrics=load_evaluation_metrics(model=model_name, dataset=ds_name),
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
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            num_sanity_val_steps=0,
        )

        trainer.fit(module)
