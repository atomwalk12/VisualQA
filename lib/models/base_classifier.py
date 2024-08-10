from collections import defaultdict
from dataclasses import field
import os
from collections import defaultdict
import torch
import torch.utils.checkpoint
from easy_vqa import get_answers
from peft.peft_model import PeftModel
from torch.nn import Module
from transformers import Blip2Config, PreTrainedModel


class Blip2ClassifierConfig(Blip2Config):
    model_type = "blip-2"

    def __init__(
        self,
        classification_input_dim=0,
        base_model_name="blip2",
        interm_dim=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.interm_dim = interm_dim
        self.classification_input_dim = classification_input_dim
        self.answer_space = get_answers()


class Blip2(PreTrainedModel):
    model: PreTrainedModel
    config_class: Blip2Config
    interm_layer: Module

    def __init__(self, config: Blip2ClassifierConfig):
        super().__init__(config)
        self.embeddings: defaultdict[str, list] = defaultdict(list)

    def save_pretrained(self, save_directory, **kwargs):
        output_path = f"{save_directory}"
        # Classification class config
        self.config.save_pretrained(output_path)

        # Save the PEFT model
        self.model.save_pretrained(output_path)

        # Save the additional layer
        additional_layer_path = os.path.join(output_path, "classification_layer.pt")
        torch.save(self.interm_layer.state_dict(), additional_layer_path)

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        pretrained_model_name_or_path,
        adapter_name,
        is_trainable,
        *model_args,
        **kwargs,
    ):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)

        peft_model = PeftModel.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
        )
        peft_model.print_trainable_parameters()

        model = cls(config, peft_model)

        # Load the additional layer
        additional_layer_path = os.path.join(
            pretrained_model_name_or_path, "classification_layer.pt"
        )
        if os.path.exists(additional_layer_path):
            model.interm_layer.load_state_dict(torch.load(additional_layer_path))

        return model

    def get_state(self):
        return self.embeddings
