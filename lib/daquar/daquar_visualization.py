from ..types import VQAParameters
from .daquar_generation import DaquarGeneration


class VisualizationDaquarGeneration(DaquarGeneration):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        encoding = super().__getitem__(idx)
        return encoding, idx
