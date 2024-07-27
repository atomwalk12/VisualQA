from PIL.Image import Image
from ..lib.datasets_vqa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQAElement


def test_load_dataset():
    """
    Testing Dataset initialization
    """
    wrapper = EasyVQADataset()
    wrapper.initialize()

    assert len(wrapper.dataset) > 0


def test_get_item():
    """
    Testing retrieval of dataset item
    """
    wrapper = EasyVQADataset()
    wrapper.initialize()
    assert isinstance(wrapper[0], EasyVQAElement)


def test_dataset_fields():
    wrapper = EasyVQADataset()
    wrapper.initialize()
    element: EasyVQAElement = wrapper[0]
    assert isinstance(element.image, Image)
