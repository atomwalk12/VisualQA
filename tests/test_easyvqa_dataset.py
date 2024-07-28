import pytest

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQAElement


@pytest.fixture(scope="module")
def initialize_train_dataset():
    train_ds = EasyVQADataset(split="train")
    train_ds.initialize_for_training()
    return train_ds


def test_get_train_item(initialize_train_dataset: EasyVQADataset):
    element: EasyVQAElement = initialize_train_dataset[0]

    assert element.label is not None
    assert element.question is not None


@pytest.fixture(scope="module")
def initialize_val_dataset():
    train_ds = EasyVQADataset(split="val")
    train_ds.initialize_for_training()
    return train_ds


def test_get_val_item(initialize_val_dataset: EasyVQADataset):
    element: EasyVQAElement = initialize_val_dataset[0]

    assert element.label is not None
    assert element.question is not None


def test_get_10_val_items():
    val_ds = EasyVQADataset(split="val[:10]")
    val_ds.initialize_for_training()
    assert len(val_ds) == 10