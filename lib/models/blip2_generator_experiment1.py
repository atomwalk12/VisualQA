import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from peft.peft_model import PeftModel

logger = logging.getLogger(__name__)


class Blip2GeneratorExperiment1(nn.Module):
    """
    This generator class is an attempt to reduce the strength of the generative
    capabilities since it overfits on the first training epoch. I use L1
    regularization to reduce the strength of the LoRA layers. L2 regularization
    is added through the AdamW optimizer.
    """

    def __init__(self, peft_model: PeftModel):
        super().__init__()
        self.base_model = peft_model

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        lambda_l1: float = 0.01,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )

        # Add L1 regularization
        l1_reg = 0.0
        num_params = 0
        for name, param in self.base_model.named_parameters():
            if "lora" in name and param.requires_grad:  # Only apply to LoRA parameters
                l1_reg += param.abs().sum()
                num_params += param.numel()

        if num_params > 0:
            l1_reg = l1_reg / num_params  # Normalize by number of parameters
            total_loss = outputs.loss + lambda_l1 * l1_reg
        else:
            total_loss = outputs.loss

        # Create a new output object with the updated loss
        new_outputs = type(outputs)(
            loss=total_loss, **{k: v for k, v in outputs.items() if k != "loss"}
        )

        return new_outputs

    def save_pretrained(self, path: str):
        self.base_model.save_pretrained(path)

    def push_to_hub(self, repo, commit_message):
        self.base_model.push_to_hub(repo, commit_message)

    def reset_state(self):
        pass

    def save_statistics(self, path: str):
        pass
