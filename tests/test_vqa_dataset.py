from PIL.Image import Image
import pytest
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

def test_dataset_fields(initialized_dataset):
    element: EasyVQAElement = initialized_dataset[0]
    assert isinstance(element.image, Image)