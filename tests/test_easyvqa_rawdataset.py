import pytest
from PIL.Image import Image

from ..lib.datasets_qa.easyvqa import EasyVQADataset


@pytest.fixture(scope="module")
def val_raw_dataset():
    wrapper = EasyVQADataset(split="val")
    return wrapper


@pytest.fixture(scope="module")
def train_raw_dataset():
    wrapper = EasyVQADataset(split="train")
    return wrapper


def test_load_train_dataset(train_raw_dataset: EasyVQADataset):
    assert len(train_raw_dataset.raw_dataset) > 0


def test_get_raw_item(train_raw_dataset: EasyVQADataset):
    assert isinstance(train_raw_dataset[0], dict)


def test_check_training_item_properties(train_raw_dataset: EasyVQADataset):
    element: dict = train_raw_dataset[0]
    assert len(element["answer"]) > 0
    assert len(element["question"]) > 0
    assert len(element["image_path"]) > 0
    assert element["image_id"] is not None
    assert isinstance(element["image"], Image)


def test_check_individual_value_from_batch(train_raw_dataset: EasyVQADataset):
    elements = train_raw_dataset[:10]

    for question, answer, image_id, image_path, image in zip(
        elements["question"],
        elements["answer"],
        elements["image_id"],
        elements["image_path"],
        elements["image"],
    ):
        assert len(answer) > 0
        assert len(question) > 0
        assert len(image_path) > 0
        assert image_id is not None
        assert isinstance(image, Image)


def test_load_val_dataset(
    train_raw_dataset: EasyVQADataset, val_raw_dataset: EasyVQADataset
):
    assert len(train_raw_dataset) > len(val_raw_dataset)


def test_check_slicing_operator():
    dataset = EasyVQADataset(split="val[:10]")
    assert len(dataset) == 10
    dataset = EasyVQADataset(split="val[5:10]")
    assert len(dataset) == 5
    dataset = EasyVQADataset(split="val[1000:]")
    assert len(dataset) > 1000
