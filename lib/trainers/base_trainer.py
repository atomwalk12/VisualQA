import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.checkpoint
from colorama import Fore, Style
from peft import LoraConfig
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from lib.models.base_classifier import Blip2BaseClassifier
from lib.models.blip2_generator_experiment1 import Blip2GeneratorExperiment1
from lib.utils import get_id
import wandb
from lib.representations import ModelFactory

from ..types import (
    SAVE_PATHS,
    CustomDataset,
    FileNames,
    State,
    TrainingParameters,
    VQAParameters,
)
from ..utils import (
    EXPERIMENT,
    ROOT_DATA_DIR,
    format_time,
    read_wandb_id,
    write_wandb_id,
)

logger = logging.getLogger(__name__)


class TorchBase(ABC):
    model: Blip2BaseClassifier | Blip2GeneratorExperiment1

    def __init__(self, config: TrainingParameters):
        # Log information to file
        self.model_name = config.model_name
        self.add_logger()
        self.dataset_name = config.dataset_name
        self.resume_checkpoint = config.resume_checkpoint
        self.best_path = self.get_save_path()

        if self.resume_checkpoint:
            wandb_id = read_wandb_id(f"{self.best_path}/wandb_run.txt")
        else:
            wandb_id = write_wandb_id(f"{self.best_path}/wandb_run.txt")

        # The path to store the best model
        self.best_path = self.best_path + f"/{str(wandb_id)}"
        SAVE_PATHS.make_dir(self.best_path)

        self.update_frequency = 64
        self.lora: LoraConfig = self.lora_config()
        self.bnb = self.bnb_config()

        # Log configurations
        logger.info(self.lora)
        logger.info(self.bnb)

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
        self.config.lora_config = self.lora_config()
        self.config.bnb_config = self.bnb_config()

        # Check whether a CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare the module for training

        self.model, self.processor = self.prepare_module()
        self.model.to(self.device)

        # Load Sentence-BERT model
        self.sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        if config.use_wandb:
            resume_wandb = "allow" if self.resume_checkpoint else "never"
            if config.is_testing:
                wandb_id = get_id()

            # Setup wandb and log model properties
            self.run = wandb.init(
                project=config.wandb_project,
                config=config,
                job_type="Train",
                tags=[
                    config.model_name,
                    f"rank_{self.lora.r}",
                    f"lora_alpha_{self.lora.lora_alpha}",
                ],
                id=f"{config.model_name}-{self.dataset_name}-baseline-{wandb_id}",
                resume=resume_wandb,
                name=f"{config.model_name}-{self.dataset_name}-baseline",
                anonymous="must",
            )

            wandb.watch(self.model, log_freq=100)

        # Repository where to upload the results
        self.repo = self.get_repository()
        self.hyperparameters: TrainingParameters = config
        self.hyperparameters.set_optimizer_and_scheduler(self.model)
        self.optimizer = self.hyperparameters.optimizer
        self.scheduler = self.hyperparameters.scheduler

        if not config._resume_state:
            self.state = State()
        else:
            self.state = State.load_state(
                self.best_path, FileNames.StateDictionary.format(config.split)
            )
            self.optimizer.load_state_dict(self.state.optimizer_state_dict)
            self.scheduler.load_state_dict(self.state.scheduler_state_dict)

    def prepare_module(self) -> Tuple[Blip2ForConditionalGeneration, Blip2Processor]:
        params = self.config

        processor = self.load_processor(use_checkpoint=self.resume_checkpoint)

        if params.train_args is not None:
            params.train_args.processor = processor
            train_dataset: CustomDataset = self.get_dataset(params.train_args)
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=params.train_batch_size,
                shuffle=True,
                worker_init_fn=EXPERIMENT.seed_worker,
                generator=EXPERIMENT.get_generator(),
                num_workers=params.num_train_workers,
            )
            self.answer_space = train_dataset.answer_space

        if params.val_args is not None:
            params.val_args.processor = processor
            val_dataset: CustomDataset = self.get_dataset(params.val_args)
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=params.val_batch_size,
                shuffle=False,
                worker_init_fn=EXPERIMENT.seed_worker,
                generator=EXPERIMENT.get_generator(),
                num_workers=params.num_val_workers,
            )
            self.answer_space = val_dataset.answer_space
            assert len(val_dataset.answer_space) == len(train_dataset.answer_space)

        if params.test_args is not None:
            params.test_args.processor = processor
            test_dataset: CustomDataset = self.get_dataset(params.test_args)
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=params.test_batch_size,
                shuffle=False,
                worker_init_fn=EXPERIMENT.seed_worker,
                generator=EXPERIMENT.get_generator(),
                num_workers=params.num_test_workers,
            )
            self.answer_space = test_dataset.answer_space

        if self.resume_checkpoint:
            model = self.load_from_checkpoint(is_trainable=params.is_trainable)
        else:
            model = self.bootstrap_model(answer_space=self.answer_space)

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
        for epoch in range(
            self.state.current_epoch, self.hyperparameters.num_epochs + 1
        ):
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch=epoch)

            # Now perform validation
            val_loss = self.valid_one_epoch(epoch=epoch)

            # Log the metrics
            history["Epoch Train Loss"].append(train_loss)
            history["Epoch Valid Loss"].append(val_loss)

            wandb.log({"Epoch Train Loss": train_loss})
            wandb.log({"Epoch Valid Loss": val_loss})
            self.on_epoch_end()

            # Save the best result
            if val_loss <= best_epoch_loss:
                print(
                    f"{blue}Validation Loss Improved ({best_epoch_loss} ---> {val_loss})"
                )
                # Update best loss
                best_epoch_loss = val_loss
                # Saving the epoch which is supposed to be the next
                self.save_trainer_state(
                    best_epoch_loss,
                    history,
                    epoch + 1,
                    self.train_dataloader,
                )

                # Log the statistics
                self.run.summary["Best Loss"] = best_epoch_loss

                # Store best weights
                self.model.save_pretrained(self.best_path)
                self.processor.save_pretrained(self.best_path)
                self.state.best_epoch_loss = best_epoch_loss

                # Save the best model
                logger.info(f"Saved model {self.best_path} --> {best_epoch_loss:.4f}")

                # Save a model file from the current directory
                print(f"Model Saved{reset} --> {self.best_path}")

                # Push to hub every time a better model is found
                try:
                    self.push_to_hub(self.model, self.processor, epoch)
                except Exception as e:
                    logger.error(f"Failed to push to hub: {e}")
                self.model.save_statistics(self.best_path)
                self.model.reset_state(epoch, is_better=True)
            else:
                logger.info(f"{val_loss=}")
                self.model.reset_state(epoch, is_better=False)

                # Saving the epoch which is supposed to be the next
                self.save_trainer_state(
                    best_epoch_loss,
                    history,
                    epoch + 1,
                    self.train_dataloader,
                )

            print()

        end = time.time()

        # Now report the results
        time_elapsed = end - start
        logger.info(f"Training complete in {format_time(time_elapsed)}")
        logger.info(f"Best Loss: {best_epoch_loss:.4f}")

        # Load the best model
        model = self.load_from_checkpoint(is_trainable=True)
        processor = self.load_processor(use_checkpoint=True)

        # Release resources
        del self.train_dataloader, self.val_dataloader
        self.run.finish()
        wandb.finish()

        return model, history

    def push_to_hub(self, model, processor, epoch):
        loss = self.state.best_epoch_loss
        model.push_to_hub(self.repo, commit_message=f"New best {loss=}, {epoch=}")
        processor.push_to_hub(self.repo, commit_message=f"New best {loss=}, {epoch=}")

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

            # Prepare arguments for the model's forward method
            model_kwargs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "labels": labels,
                "attention_mask": attention_mask,
            }

            # Add the extract_features parameter only during classification
            if hasattr(self.model, "classifier"):
                model_kwargs["extract_features"] = "train"

            # Generate the output trainable params:
            outputs = self.model(**model_kwargs)

            # Now update the loss
            loss = outputs.loss
            loss = loss / n_accumulate
            loss.backward()

            self.on_batch_processed(outputs, labels)
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

                wandb.log({"Train Epoch Loss": epoch_loss})
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

                # Prepare arguments for the model's forward method
                model_kwargs = {
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

                # Add the extract_features parameter only during classification
                if hasattr(self.model, "classifier"):
                    model_kwargs["extract_features"] = "val"

                outputs = self.model(**model_kwargs)

                loss = outputs.loss

                self.on_batch_processed(outputs, labels)

                # Accumulate the loss across multiple batches
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Now compute the final loss value
                epoch_loss = running_loss / dataset_size

                wandb.log({"Validation Epoch Loss": epoch_loss})
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
        Path(f"{ROOT_DATA_DIR}/logs/{self.model_name}").mkdir(
            parents=True, exist_ok=True
        )

        handler = logging.FileHandler(
            filename=f"{ROOT_DATA_DIR}/logs/{self.model_name}/{datetime.now().strftime('%m_%d_%Y_%H_%M.log')}"
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def load_processor(self, use_checkpoint):
        if use_checkpoint:
            processor = Blip2Processor.from_pretrained(self.best_path)
        else:
            processor = ModelFactory.get_processor(self.model_name)
        return processor

    @abstractmethod
    def send_to_device_if_needed(self, data):
        pass

    @abstractmethod
    def bootstrap_model(self, answer_space):
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
    def test(self):
        pass

    @abstractmethod
    def get_dataset(self, args: VQAParameters):
        pass

    @abstractmethod
    def update_state_with_embeddings(self, embeddings=None):
        # When training the model for generation, there are no embeddings to save
        pass

    @abstractmethod
    def save_trainer_state(
        self, best_epoch_loss, history, epoch, dataloader: DataLoader
    ):
        pass

    @abstractmethod
    def bnb_config(self):
        pass

    @abstractmethod
    def lora_config(self):
        pass

    @abstractmethod
    def on_batch_processed(self, preds, targets):
        pass

    @abstractmethod
    def on_best_epoch(self):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass
