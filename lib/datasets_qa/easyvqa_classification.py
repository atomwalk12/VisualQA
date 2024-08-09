from ..datasets_qa.easyvqa_base import EasyVQADatasetBase
from ..types import VQAParameters
import torch.nn.functional as F
import torch


class EasyVQAClassification(EasyVQADatasetBase):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        item, encoding = super().__getitem__(idx)

        encoding["labels"] = torch.tensor(item["label"]).to(dtype=torch.float16)
        return encoding

    def _prepare_for_training(self, item: dict):
        prompt = item["question"]
        label = item["answer"]

        digit = self.answers_to_id[label]
        label = F.one_hot(torch.tensor(digit), len(self.answer_space))

        return {"prompt": prompt, "label": label}

    def get_make_complete_path(self):
        return super().get_make_complete_path("classification")
