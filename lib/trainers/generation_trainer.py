import gc
import logging

import torch
import torch.utils.checkpoint
from peft import prepare_model_for_kbit_training
from peft.peft_model import PeftModel

# PyTorch TensorBoard support
from tqdm import tqdm

import wandb
from config import Repositories

from ..datasets_qa.easyvqa_generation import EasyVQAGeneration
from ..representations import SAVE_PATHS, ModelFactory
from ..types import Suffix, TrainingParameters
from .base_trainer import TorchBase

logger = logging.getLogger(__name__)


class GenerationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)

        self.update_frequency = 64

    def get_repository(self):
        return Repositories.VQAGeneration

    def get_save_path(self):
        return SAVE_PATHS.BLIP2_Generator

    def load_from_checkpoint(self, is_trainable):
        base_model, processor = ModelFactory.get_models(self.model_name, apply_lora=False)

        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

        local_model_path = self.get_model_save_path()
        model = PeftModel.from_pretrained(
            base_model,
            local_model_path,
            adapter_name="vqa_generation",
            is_trainable=is_trainable,
        )
        model.print_trainable_parameters()

        return model, processor

    def get_model_save_path(self):
        return SAVE_PATHS.BLIP2_Generator

    def bootstrap_model(self):
        model, processor = ModelFactory.get_models(self.model_name, apply_lora=True)
        return model, processor

    def test(self):
        self.model.eval()
        history = self.state.history

        dataset_size = 0
        running_similarity = 0.0
        with torch.no_grad():
            bar = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))
            for step, data in bar:
                # Unpack the collator values
                pixel_values = data["pixel_values"].to(self.device)
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"]

                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for sample, label in zip(generated_text, labels):
                    similarity = self.sbert_similarity(sample, label)
                    running_similarity += similarity
                    dataset_size += 1

                best_epoch_loss = running_similarity / dataset_size

                history["Test Loss"].append(best_epoch_loss)
                wandb.log({"Test Loss": best_epoch_loss})

                bar.set_postfix(Batch=step, Test_Loss=best_epoch_loss)

                # Save the state
                if step % self.update_frequency == 0:
                    self.save_state(
                        best_epoch_loss,
                        history,
                        step + 1,
                        Suffix.Test,
                        self.test_dataloader.dataset,
                    )

        # Finally save the entire run results
        self.save_state(
            best_epoch_loss,
            history,
            step + 1,
            Suffix.Test,
            self.test_dataloader.dataset,
        )

        # Release resources
        gc.collect()
        del self.test_dataloader
        self.run.finish()
        wandb.finish()

        return history

    def send_to_device_if_needed(self, data):
        input_ids = data["input_ids"].to(self.device)
        pixel_values = data["pixel_values"].to(self.device)
        labels = data["labels"]
        attention_mask = data["attention_mask"].to(self.device)
        return input_ids, pixel_values, attention_mask, labels

    def get_dataset(self, args):
        return EasyVQAGeneration(args)

    def update_state_with_embeddings(self, embeddings):
        # this is done for every iteration
        self.state.history["embeddings"].append(embeddings)

    def save_state(self, best_epoch_loss, history, epoch, suffix, dataset):
        self.state.save_state(
            self.best_path,
            best_epoch_loss,
            history,
            epoch,
            self.scheduler,
            self.optimizer,
            dataset,
            file_name=f"{suffix}_state_dict.pkl",
        )
