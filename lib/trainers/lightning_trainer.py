import logging

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..types import LightningConfig

logger = logging.getLogger(__name__)


class BLIP2PLModule(L.LightningModule):
    def __init__(self, config: None):
        super().__init__()
        logger.info(config)

        self.config = config
        self.processor = config.processor
        self.model = config.model
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.torch_hyperparameters.val_batch_size
        self.metrics = config.metrics

    def train_dataloader(self):
        assert self.config.train_dataset is not None

        return DataLoader(
            self.config.train_dataset,
            collate_fn=lambda batch: self.train_collate_fn(batch, self.config),
            batch_size=self.train_batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=0,
        )

    def val_dataloader(self):
        assert self.config.val_dataset is not None

        return DataLoader(
            self.config.val_dataset,
            collate_fn=lambda batch: self.eval_collate_fn(batch, self.config),
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs.loss
        error = loss.item()
        logger.info(f"Epoch {self.current_epoch}, loss: {error}")

        self.log("train_loss", error, batch_size=self.train_batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        self.model.eval()
        inputs, labels = batch

        # Generate predictions
        generated_ids = self.model.generate(**inputs)
        predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Log batch information
        self._log_batch_info(batch_idx, predictions, labels)

        # Compute evaluation metrics
        for metric in self.metrics:
            scores = metric.compute(pred=predictions, references=labels)
            for k in metric.log_columns:
                self.log(
                    f"{metric.name}_{k}",
                    np.mean(scores[k]),
                    batch_size=self.train_batch_size,
                )

        return scores

    def _log_batch_info(self, batch_idx, predictions, labels):
        total_batches = len(self.config.val_dataset) // self.train_batch_size
        logging.info(f"Batch: {batch_idx}/{total_batches}")

        for i, (pred, answer) in enumerate(zip(predictions, labels)):
            element = self.config.val_dataset[batch_idx * self.train_batch_size + i]
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
        **args,
    ):
        config = None
        # ModuleConfigGenerator.create_from(
        #     model_name=model_name,
        #     ds_name=ds_name,
        #     train_args=train_args,
        #     val_args=val_args,
        #     apply_qlora=apply_qlora,
        #     pickle_dir=pickle_dir,
        #     **args,
        # )
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