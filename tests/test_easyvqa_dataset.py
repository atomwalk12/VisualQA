import pytest

from ..lib.datasets_qa.easyvqa import EasyVQADataset


@pytest.fixture(scope="module")
def initialize_train_dataset():
    train_ds = EasyVQADataset(split="train")
    train_ds.initialize_for_training()
    return train_ds


@pytest.fixture(scope="module")
def initialize_val_dataset():
    train_ds = EasyVQADataset(split="val")
    train_ds.initialize_for_training()
    return train_ds


def test_get_train_item(initialize_train_dataset: EasyVQADataset):
    element = initialize_train_dataset[0]

    assert element['label'] is not None
    assert element['question'] is not None


def test_get_val_item(initialize_val_dataset: EasyVQADataset):
    element = initialize_val_dataset[0]

    assert element['label'] is not None
    assert element['question'] is not None


def test_get_10_val_items():
    val_ds = EasyVQADataset(split="val[:10]")
    val_ds.initialize_for_training()
    assert len(val_ds) == 10


def test_save_dataset(initialize_train_dataset):
    out_dir = './data/easyvqa'
    success = initialize_train_dataset.save(out_dir)

    assert success

    dataset = initialize_train_dataset.load(out_dir)
    assert initialize_train_dataset.equals(dataset)
