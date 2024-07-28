import pytest
from PIL.Image import Image

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQARawElement


@pytest.fixture(scope="module")
def initialized_dataset():
    wrapper = EasyVQADataset(split='train', initialize_raw=True)
    return wrapper


def test_load_dataset(initialized_dataset: EasyVQADataset):
    """
    Testing Dataset initialization
    """
    assert len(initialized_dataset.raw_dataset) > 0


def test_get_item(initialized_dataset: EasyVQADataset):
    """
    Testing retrieval of dataset item
    """
    assert isinstance(initialized_dataset[0], EasyVQARawElement)


def test_check_item(initialized_dataset: EasyVQADataset):
    """
    Testing retrieval of an image with associated information
    """
    element: EasyVQARawElement = initialized_dataset[0]
    check_element(element)


def test_iterate(initialized_dataset: EasyVQADataset):
    """
    Iterate over the dataset by retrieving a number of elements
    """
    elements = initialized_dataset[:10]

    for element in elements:
        check_element(element)


def check_element(element: EasyVQARawElement):
    assert len(element.answer) > 0
    assert len(element.question) > 0
    assert len(element.image_path) > 0
    assert element.image_id is not None
    assert isinstance(element.image, Image)
