import torch
import torch.nn.functional as F

from lib.daquar.daquar_base import DaquarDatasetBase

from ..types import VQAParameters


class DaquarClassification(DaquarDatasetBase):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        item, encoding = super().__getitem__(idx)

        encoding["labels"] = torch.tensor(item["label"]).to(dtype=torch.float16)
        return encoding

    def _prepare_for_training(self, item: dict):
        prompt = item["question"]
        labels = item["answer"]  # Assuming this is now a list of labels

        # Initialize a multi-hot vector with zeros
        multi_hot_label = torch.zeros(len(self.answer_space))

        # Encode each label as a one-hot vector and add it to the multi-hot vector
        for label in labels:
            digit = self.answers_to_id[label]
            one_hot_label = F.one_hot(torch.tensor(digit), len(self.answer_space))
            multi_hot_label += one_hot_label
        multi_hot_label = multi_hot_label.clamp(0, 1)

        return {"prompt": prompt, "label": multi_hot_label}

    def get_make_complete_path(self):
        return super().get_make_complete_path("classification")
