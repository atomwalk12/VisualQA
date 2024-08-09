import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.checkpoint
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

import wandb

from ..types import State, TrainingParameters, VQAParameters
from ..utils import ROOT_DATA_DIR, format_time, get_generator, seed_worker

logger = logging.getLogger(__name__)


class TorchBase(ABC):
    model: PreTrainedModel

    def __init__(self, config: TrainingParameters):
        # This forces CUDA operations to be executed sequentially, which can
        # help with debugging by providing more precise error messages
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # This disables tokenizer parallelism, which can help avoid potential
        # issues with multithreading in tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        # Use cuda
        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Save the configuration
        self.config = config

        # Check whether a CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare the module for training
        self.resume = config.resume
        self.model_name = config.model_name
        self.model, self.processor = self.prepare_module()
        self.model.to(self.device)

        # Log information to file
        self.add_logger()

        # Load Sentence-BERT model
        self.sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # Setup wandb and log model properties
        self.run = wandb.init(
            project=config.wandb_project,
            config=config,
            job_type="Train",
            tags=[config.model_name],
            name=f"{config.model_name}-baseline",
            anonymous="must",
        )

        wandb.watch(self.model, log_freq=100)

        # The path to store the best model
        self.best_path = self.get_save_path()

        # Repository where to upload the results
        self.repo = self.get_repository()

        self.hyperparameters: TrainingParameters = config
        self.hyperparameters.set_optimizer_and_scheduler(self.model)
        self.optimizer = self.hyperparameters.optimizer
        self.scheduler = self.hyperparameters.scheduler

        if not self.resume:
            self.state = State()
        else:
            self.state = State.load_state(self.best_path)
            self.optimizer.load_state_dict(self.state.optimizer_state_dict)
            self.scheduler.load_state_dict(self.state.scheduler_state_dict)

    def prepare_module(self):
        params = self.config
        if self.resume:
            model, processor = self.load_from_checkpoint(is_trainable=params.is_trainable)
        else:
            model, processor = self.bootstrap_model()

        if params.train_args is not None:
            params.train_args.processor = processor
            train_dataset = self.get_dataset(params.train_args)
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=params.train_batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=get_generator(),
                num_workers=12,
            )

        if params.val_args is not None:
            params.val_args.processor = processor
            val_dataset = self.get_dataset(params.val_args)
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=params.val_batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=get_generator(),
                num_workers=8,
            )

        if params.test_args is not None:
            params.test_args.processor = processor
            test_dataset = self.get_dataset(params.test_args)
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=params.test_batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=get_generator(),
                num_workers=8,
            )

        return model, processor

    def finetune(self):
        # Initialize the statistics
        start = time.time()
        best_epoch_loss = self.state.best_epoch_loss
        history = self.state.history

        # Use colors for the print statements
        blue = Fore.BLUE
        reset = Style.RESET_ALL

        # Start training
        for epoch in range(self.state.current_epoch, self.hyperparameters.num_epochs + 1):
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
                self.model.save_pretrained(self.best_path)

                # Save the best model
                logger.info(f"Saved model {self.best_path} --> {best_epoch_loss:.4f}")

                # Save a model file from the current directory
                print(f"Model Saved{reset} --> {self.best_path}")

            self.state.save_state(
                self.best_path,
                best_epoch_loss,
                history,
                epoch + 1,
                self.scheduler,
                self.optimizer,
            )

            print()

        end = time.time()

        # Now report the results
        time_elapsed = end - start
        logger.info(f"Training complete in {format_time(time_elapsed)}")
        logger.info(f"Best Loss: {best_epoch_loss:.4f}")

        # Load the best model

        model, processor = self.load_from_checkpoint(is_trainable=True)
        self.push_to_hub(model, processor)
        print()

        # Release resources
        del self.train_dataloader, self.val_dataloader
        self.run.finish()
        wandb.finish()

        return model, history

    def push_to_hub(self, model, processor):
        loss = self.state.best_epoch_loss
        model.push_to_hub(self.repo, commit_message=f"Training done {loss=}")
        processor.push_to_hub(self.repo, commit_message="Training done")

    def train_one_epoch(self, epoch):
        # Set in training mode
        self.model.train()

        dataset_size = 0
        running_loss = 0.0
        n_accumulate = self.hyperparameters.n_accumulate

        bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for step, data in bar:
            # Unpack the batch
            input_ids, pixel_values, attention_mask, labels = (
                self.send_to_device_if_needed(data)
            )

            # Current batch size, can be less than self.batch_size for the last batch
            batch_size = input_ids.size(0)

            # Generate the output
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=labels,
                attention_mask=attention_mask,
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

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                epoch_loss = running_loss / dataset_size

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
        with torch.no_grad():
            bar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))
            for step, data in bar:
                # Unpack the collator values
                input_ids, pixel_values, attention_mask, labels = (
                    self.send_to_device_if_needed(data)
                )

                # Get the batch size
                batch_size = input_ids.size(0)

                # Perform generation using the model

                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                    attention_mask=attention_mask,
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

    def sbert_similarity(self, sentence1, sentence2):
        embeddings1 = self.sbert.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.sbert.encode(sentence2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_scores.item()

    def add_logger(self):
        Path(f"{ROOT_DATA_DIR}/logs/{self.model_name}").mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(
            filename=f"{ROOT_DATA_DIR}/logs/{self.model_name}/{datetime.now().strftime('_%H_%M_%d_%m_%Y.log')}"
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @abstractmethod
    def send_to_device_if_needed(self, data):
        pass

    @abstractmethod
    def bootstrap_model(self):
        pass

    @abstractmethod
    def load_from_checkpoint(self, is_trainable):
        pass

    @abstractmethod
    def get_repository(self):
        pass

    @abstractmethod
    def get_save_path(self):
        pass

    @abstractmethod
    def get_model_save_path(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_dataset(self, args: VQAParameters):
        pass
