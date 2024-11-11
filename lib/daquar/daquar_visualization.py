from ..types import VQAParameters
from .daquar_generation import DaquarGeneration


class VisualizationDaquarGeneration(DaquarGeneration):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        # Encode the dataset item
        item = self.dataset[idx]

        padding = {"padding": "max_length", "max_length": self.padding_max_length}

        encoding = self.processor(
            images=item["image"],
            text=item["prompt"],
            return_tensors="pt",
            **padding,
        )

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        encoding["labels"] = encoding["input_ids"]

        return encoding, idx
