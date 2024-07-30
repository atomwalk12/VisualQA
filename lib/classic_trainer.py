import logging
from datetime import datetime

import evaluate
import torch
from torch.utils.data import DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from transformers import Blip2ForConditionalGeneration

from lib.lightning_trainer import BLIP2PLModule
from lib.types import ModuleConfig
from lib.utils import likely_pickle_dir

from .representations import DatasetFactory, ModelFactory, load_evaluation_metrics

logger = logging.getLogger(__name__)


class SimpleFinetuneLoop:
    def __init__(self, config: ModuleConfig):
        # Set up configuration parameters
        self.config = config
        self.model: Blip2ForConditionalGeneration = config.model
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metric = evaluate.load("bertscore")
        self.processor = config.processor
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )

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

        module = SimpleFinetuneLoop(config)
        return module

    def finetune(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
        epoch_number = 0

        EPOCHS = 25

        best_vloss = 1_000_000.0

        for epoch in range(EPOCHS):
            print("EPOCH {}:".format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

            # Perform validation every validation_interval epochs
            avg_vloss = self.evaluate()

            print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number + 1,
            )
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = "model_{}_{}".format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_dataloader):
            # Every data instance is an input + label pair
            inputs = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(**inputs)

            # Compute the loss and its gradients
            # loss = loss_fn(outputs, labels)
            loss = outputs.loss

            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
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
            for i, vdata in enumerate(self.val_dataloader):
                vinputs, vlabels = vdata
                voutputs = self.model(**vinputs)
                voutputs = self.processor.batch_decode(
                    voutputs, skip_special_tokens=True
                )
                # vloss = loss_fn(voutputs, vlabels)
                vloss = self.metric.compute(voutputs, vlabels, lang="en")
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        return avg_vloss
