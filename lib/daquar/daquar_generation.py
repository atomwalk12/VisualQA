from lib.daquar.daquar_base import DaquarDatasetBase
from typing import List
from ..types import Suffix, VQAParameters


class DaquarGeneration(DaquarDatasetBase):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        item, encoding = super().__getitem__(idx)
        if self.is_testing:
            encoding["labels"] = item["label"]
        else:
            encoding["labels"] = encoding["input_ids"]
        return encoding

    def _prepare_for_training(self, item: dict):
        input_text = item["question"]
        label: List[str] = item["answer"]

        if self.split.startswith(Suffix.Train):
            prompt = f"Question: {input_text}? Answer: {", ".join(label)}."
        elif self.split.startswith(Suffix.Val) or self.split.startswith(Suffix.Test):
            prompt = f"Question: {input_text}? Answer:"
        else:
            raise Exception(f"Flag {self.split} not recognized.")

        return {"prompt": prompt, "label": label}

    def get_make_complete_path(self):
        return super().get_make_complete_path("generation")