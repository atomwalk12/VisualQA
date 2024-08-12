from .easyvqa_generation import EasyVQAGeneration

from ..types import VQAParameters


class VisualizationEasyVQAGeneration(EasyVQAGeneration):
    def __init__(self, params: VQAParameters):
        super().__init__(params)

    def __getitem__(self, idx):
        encoding = super().__getitem__(idx)
        return encoding, idx