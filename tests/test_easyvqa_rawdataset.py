import pytest
from PIL.Image import Image

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQARawElement


@pytest.fixture(scope="module")
def train_raw_dataset():
    wrapper = EasyVQADataset(split='train')
    return wrapper


def test_load_dataset(train_raw_dataset: EasyVQADataset):
    """
    Testing Dataset initialization
    """
    assert len(train_raw_dataset.raw_dataset) > 0


def test_get_item(train_raw_dataset: EasyVQADataset):
    """
    Testing retrieval of dataset item
    """
    assert isinstance(train_raw_dataset[0], EasyVQARawElement)


def test_check_item(train_raw_dataset: EasyVQADataset):
    """
    Testing retrieval of an image with associated information
    """
    element: EasyVQARawElement = train_raw_dataset[0]
    check_element(element)


def test_iterate(train_raw_dataset: EasyVQADataset):
    """
    Iterate over the dataset by retrieving a number of elements
    """
    elements = train_raw_dataset[:10]

    for element in elements:
        check_element(element)


def check_element(element: EasyVQARawElement):
    assert len(element.answer) > 0
    assert len(element.question) > 0
    assert len(element.image_path) > 0
    assert element.image_id is not None
    assert isinstance(element.image, Image)


@pytest.fixture(scope="module")
def val_raw_dataset():
    wrapper = EasyVQADataset(split='val')
    return wrapper


def test_load_val_dataset(train_raw_dataset: EasyVQADataset, val_raw_dataset: EasyVQADataset):
    assert len(train_raw_dataset) > len(val_raw_dataset)
