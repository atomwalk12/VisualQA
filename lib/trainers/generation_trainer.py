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
from .base_trainer import TorchBase
from ..representations import SAVE_PATHS, ModelFactory

from ..datasets_qa.easyvqa_generation import EasyVQAGeneration
from ..types import State, TrainingParameters

logger = logging.getLogger(__name__)


class GenerationTrainer(TorchBase):
    def __init__(self, config: TrainingParameters):
        super().__init__(config)

        self.state = State()

    def get_repository(self):
        return Repositories.VQAGeneration

    def get_save_path(self):
        return SAVE_PATHS.BLIP2_Generator

    def load_from_checkpoint(self, is_trainable):
        base_model, processor = ModelFactory.get_models(self.model_name, apply_lora=False)

        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True
        )

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
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                for sample, label in zip(generated_text, labels):
                    similarity = self.sbert_similarity(sample, label)
                    running_similarity += similarity
                    dataset_size += 1

                batch_sim = running_similarity / dataset_size

                history["Test Loss"].append(batch_sim)
                wandb.log({"Test Loss": batch_sim})

                bar.set_postfix(Batch=step, Test_Loss=batch_sim)

        # Save the state
        self.state.save_state(
            self.best_path,
            None,
            history,
            len(self.test_dataloader),
            self.scheduler,
            self.optimizer,
            f"{self.model_name}_test_results.pkl",
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

    def update_state_with_embeddings(self):
        return None
