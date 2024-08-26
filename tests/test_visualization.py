from unittest.mock import MagicMock
import pytest

from ..lib.types import DatasetTypes, VQAParameters

from ..lib.datasets_qa.easyvqa_generation import EasyVQAGeneration
from ..lib.visualization import show_images_with_captions
from transformers import Blip2Processor


@pytest.fixture(scope="module")
def train_ds():
    # Get cached dataset is available, otherwise generate a new one
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    args = VQAParameters(split="train[16:25]", processor=processor, recompute=False)
    train_ds = EasyVQAGeneration(args)

    return train_ds


def test_show_train_samples(train_ds):
    elements = train_ds.raw_dataset[2:5]

    captions = [
        f"Question: {question} Answer: {answer}"
        for question, answer in zip(elements["question"], elements["answer"])
    ]

    show_images_with_captions(images_or_paths=elements["image"], captions=captions)
