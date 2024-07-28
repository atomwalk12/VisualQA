import pytest
from PIL.Image import Image

from ..lib.datasets_vqa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQAElement


@pytest.fixture(scope="module")
def initialized_dataset():
    wrapper = EasyVQADataset()
    wrapper.initialize()
    return wrapper


def test_load_dataset(initialized_dataset):
    """
    Testing Dataset initialization
    """
    assert len(initialized_dataset.dataset) > 0


def test_get_item(initialized_dataset):
    """
    Testing retrieval of dataset item
    """
    assert isinstance(initialized_dataset[0], EasyVQAElement)


def test_check_item(initialized_dataset):
    """
    Testing retrieval of an image with associated information
    """
    element: EasyVQAElement = initialized_dataset[0]
    check_element(element)


def test_iterate(initialized_dataset):
    elements = initialized_dataset[:10]

    for element in elements:
        check_element(element)


def check_element(element):
    assert len(element.answer) > 0
    assert len(element.question) > 0
    assert len(element.image_path) > 0
    assert element.image_id is not None
    assert isinstance(element.image, Image)
