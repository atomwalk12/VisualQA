from ..lib.datasets_vqa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQAElement


def test_easyvqa_load_dataset():
    """
    Testing Dataset initialization
    """
    wrapper = EasyVQADataset()
    wrapper.initialize()

    assert len(wrapper.dataset) > 0


def test_easyvqa_get_item():
    """
    Testing retrieval of dataset item
    """
    wrapper = EasyVQADataset()
    wrapper.initialize()
    assert isinstance(wrapper[0], EasyVQAElement)
