import gc
import itertools
import logging

import torch
import torch.utils.checkpoint
from peft import LoraConfig
from peft.peft_model import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput

import wandb
from config import Repositories

from ..daquar.daquar_generation import DaquarGeneration
from ..easy_vqa.easyvqa_generation import EasyVQAGeneration
from ..representations import ModelFactory
from ..types import SAVE_PATHS, DatasetTypes, TrainingParameters
from .base_trainer import TorchBase

logger = logging.getLogger(__name__)


class GenerationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)
        self.update_frequency = 64

    def get_repository(self):
        return Repositories.VQAGeneration

    def get_save_path(self):
        if self.dataset_name == DatasetTypes.DAQUAR:
            return SAVE_PATHS.BLIP2_Generator_DAQUAR
        elif self.dataset_name == DatasetTypes.EASY_VQA:
            return SAVE_PATHS.BLIP2_Generator_EasyVQA

    def load_from_checkpoint(self, is_trainable):
        base_model, processor = self.get_models(apply_lora=False)

        base_model = ModelFactory.prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

        local_model_path = self.best_path
        model = PeftModel.from_pretrained(
            base_model,
            local_model_path,
            adapter_name="vqa_generation",
            is_trainable=is_trainable,
        )
        model.print_trainable_parameters()

        return model, processor

    def bootstrap_model(self):
        model, processor = self.get_models(apply_lora=True)
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

                max_similarity = 0
                for sample, label in zip(itertools.repeat(generated_text), labels):
                    similarity = self.sbert_similarity(sample, label)
                    if similarity > max_similarity:
                        max_similarity = similarity

                running_similarity += max_similarity
                dataset_size += 1

                best_epoch_loss = running_similarity / dataset_size

                history["Test Loss"].append(best_epoch_loss)
                wandb.log({"Test Loss": best_epoch_loss})

                bar.set_postfix(Batch=step, Test_Loss=best_epoch_loss)

                # Save the state
                if step % self.update_frequency == 0:
                    self.save_state(best_epoch_loss, history, step + 1, self.test_dataloader)

        # Finally save the entire run results
        self.save_state(best_epoch_loss, history, step + 1, self.test_dataloader)

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
        if self.dataset_name == DatasetTypes.EASY_VQA:
            return EasyVQAGeneration(args)
        elif self.dataset_name == DatasetTypes.DAQUAR:
            return DaquarGeneration(args)

    def update_state_with_embeddings(self, embeddings: Blip2ForConditionalGenerationModelOutput):
        # this is done for every iteration
        embeddings = {"pooler_output": embeddings.qformer_outputs.pooler_output.to("cpu")}
        self.state.history["embeddings"].append(embeddings)

    def save_state(self, best_epoch_loss, history, epoch, dataloader: DataLoader):
        self.state.save_state(
            self.best_path, best_epoch_loss, history, epoch, self.scheduler, self.optimizer, dataloader.dataset
        )

    def get_models(self, apply_lora):
        return ModelFactory.get_models(
            self.model_name,
            apply_lora=apply_lora,
            lora_config=self.lora,
            bnb_config=self.bnb,
            torch_dtype=torch.float16,
        )

    def bnb_config(self) -> BitsAndBytesConfig:
        """Create a BitsAndBytesConfig for quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    def lora_config(self) -> LoraConfig:
        """Create a LoraConfig for PEFT."""
        return LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
