import copy
import gc
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from colorama import Fore, Style
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
import datetime

# PyTorch TensorBoard support
from tqdm import tqdm
import wandb

from .lightning_trainer import BLIP2PLModule
from .representations import DatasetFactory, ModelFactory
from .types import CustomDataset, ModuleConfig, TorchTrainerConfig
from .utils import format_time

logger = logging.getLogger(__name__)


class TorchBase:
    def __init__(self, config: ModuleConfig):
        # This forces CUDA operations to be executed sequentially, which can
        # help with debugging by providing more precise error messages
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # This disables tokenizer parallelism, which can help avoid potential
        # issues with multithreading in tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        # Use cuda
        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Check whether a CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup wandb and log model properties
        self.run = wandb.init(
            project=config.wandb_project,
            config=config,
            job_type="Train",
            tags=[config.model_name],
            name=f"{config.model_name}-baseline",
            anonymous="must",
        )

        # Initialize the model
        self.model: nn.Module = config.model
        self.model.to(self.device)

        wandb.watch(self.model, log_freq=100)

        # The path to store the best model
        date_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        self.best_path = f"{date_time}.{config.model_name}.best_loss.bin"

        # Load Sentence-BERT model
        self.sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        self.processor = config.processor
        self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def prepare_module_for_training(
        model_name: str,
        ds_name: str,
        train_args: dict,
        val_args: dict,
        torch_config: TorchTrainerConfig,
        apply_qlora=True,
    ):
        # Load the model and processor
        model, processor = ModelFactory.get_models(model_name, apply_qlora=apply_qlora)

        # Load the training and validation sets
        train_ds = DatasetFactory.create_dataset(ds_name, train_args)
        val_ds = DatasetFactory.create_dataset(ds_name, val_args)

        # Adjust config parameters
        config = ModuleConfig(
            model_name=model_name,
            processor=processor,
            model=model,
        )

        module = FineTuner(config, torch_config, train_ds, val_ds)
        return module

    @staticmethod
    def prepare_module_for_testing(
        model_name: str,
        ds_name: str,
        test_args: dict,
        apply_qlora=True,
    ):
        # Load the model and processor
        model, processor = ModelFactory.get_models(model_name, apply_qlora=apply_qlora)

        # Load the training and validation sets
        test_dataset = DatasetFactory.create_dataset(ds_name, test_args)

        # Adjust config parameters
        config = ModuleConfig(
            model_name=model_name,
            processor=processor,
            model=model,
        )

        module = TorchTest(config, test_dataset)
        return module

    # Function to calculate similarity using Sentence-BERT
    def sbert_similarity(self, sentence1, sentence2):
        embeddings1 = self.sbert.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.sbert.encode(sentence2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()


class FineTuner(TorchBase):
    def __init__(
        self,
        config: ModuleConfig,
        hyperparameters: TorchTrainerConfig,
        train_dataset: CustomDataset,
        val_dataset: CustomDataset,
    ):
        super().__init__(config)

        # Set up the run hyperparameters
        self.hyperparameters: TorchTrainerConfig = hyperparameters
        self.hyperparameters.set_optimizer_and_scheduler(self.model)
        self.optimizer = self.hyperparameters.optimizer
        self.scheduler = self.hyperparameters.scheduler

        assert self.optimizer is not None

        # Setup train and evaluation data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=lambda batch: BLIP2PLModule.train_collate_fn(batch, config),
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            collate_fn=lambda batch: BLIP2PLModule.eval_collate_fn(batch, config),
            batch_size=self.hyperparameters.val_batch_size,
            shuffle=False,
        )

    def finetune(self):
        # Initialize the statistics
        start = time.time()
        best_epoch_loss = np.inf
        history = defaultdict(list)

        # Use colors for the print statements
        blue = Fore.BLUE
        reset = Style.RESET_ALL

        # Store best weights
        best_weights = copy.deepcopy(self.model.state_dict())

        # Start training

        for epoch in range(1, self.hyperparameters.num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch=epoch)

            # Now perform validation
            val_loss = self.valid_one_epoch(epoch=epoch)

            # Log the metrics
            history["Train Loss"].append(train_loss)
            history["Valid Loss"].append(val_loss)

            wandb.log({"Train Loss": train_loss})
            wandb.log({"Valid Loss": val_loss})

            # Save the best result
            if val_loss <= best_epoch_loss:
                print(
                    f"{blue}Validation Loss Improved ({best_epoch_loss} ---> {val_loss})"
                )
                # Update best loss
                best_epoch_loss = val_loss

                # Log the statistics
                self.run.summary["Best Loss"] = best_epoch_loss

                # Store best weights
                model_state_dict = get_peft_model_state_dict(self.model)

                # Save the best model
                best_weights = copy.deepcopy(model_state_dict)
                torch.save(model_state_dict, self.best_path)

                # Save a model file from the current directory
                print(f"Model Saved{reset} --> {self.best_path}")

            print()

        end = time.time()

        # Now report the results
        time_elapsed = end - start
        print(f"Training complete in {format_time(time_elapsed)}")
        print(f"Best Loss: {best_epoch_loss:.4f}")

        # Load the best model
        best_weights = torch.load(self.best_path)
        set_peft_model_state_dict(self.model, best_weights)
        print()

        # Release resources
        del self.train_dataloader, self.val_dataloader
        self.run.finish()
        wandb.finish()

        return self.model, history

        # To load the best model from file use:
        # best_weights = torch.load(self.best_path)
        # model.load_state_dict(best_weights)

    def train_one_epoch(self, epoch):
        # Set in training mode
        self.model.train()

        dataset_size = 0
        epoch_loss = 0.0
        n_accumulate = self.hyperparameters.n_accumulate

        bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for step, data in bar:
            # Unpack the batch
            input_ids = data["input_ids"].to(self.device)
            pixel_values = data["pixel_values"].to(self.device)

            # Current batch size, can be less than self.batch_size for the last batch
            batch_size = input_ids.size(0)

            # Generate the output
            outputs = self.model(
                input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
            )

            # Now update the loss
            loss = outputs.loss
            loss = loss / n_accumulate
            loss.backward()

            # Now perform the optimizer step only when step is a multiple of n_accumulate
            if (step + 1) % n_accumulate == 0:
                self.optimizer.step()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            epoch_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = epoch_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch,
                Train_Loss=epoch_loss,
                LR=self.optimizer.param_groups[0]["lr"],
            )

        # Now perform garbage collection
        gc.collect()

        return epoch_loss

    def valid_one_epoch(self, epoch):
        # Set the model in evaluation mode
        self.model.eval()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
        for step, data in bar:
            # Unpack the collator values
            input_ids = data["input_ids"].to(self.device)
            pixel_values = data["pixel_values"].to(self.device)

            # Get the batch size
            batch_size = input_ids.size(0)

            # Perform generation using the model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
                )
            loss = outputs.loss

            # Accumulate the loss across multiple batches
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Now compute the final loss value
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch,
                Valid_Loss=epoch_loss,
                LR=self.optimizer.param_groups[0]["lr"],
            )

        gc.collect()

        return epoch_loss


class TorchTest(TorchBase):
    def __init__(self, config: ModuleConfig, test_dataset: CustomDataset):
        super().__init__(config)

        self.test_dataloader = DataLoader(
            test_dataset,
            collate_fn=lambda batch: BLIP2PLModule.test_collate_fn(batch, config),
            batch_size=64,
            shuffle=False,
        )

    def test(self):
        self.model.eval()

        dataset_size = 0
        running_similarity = 0.0
        with torch.no_grad():
            bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
            for step, (pixel_values, input_ids, attention_mask, labels) in bar:
                # Unpack the collator values
                pixel_values = pixel_values.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                for sample, label in zip(generated_text, labels):
                    similarity = self.sbert_similarity(sample, label)
                    running_similarity += similarity
                    dataset_size += 1

                batch_sim = running_similarity / dataset_size

                wandb.log({"Train Loss": batch_sim})

                bar.set_postfix(Batch=step, Test_Loss=batch_sim)

        gc.collect()

        return batch_sim
