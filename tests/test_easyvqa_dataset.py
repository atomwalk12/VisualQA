import pytest

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.types import EasyVQAElement


@pytest.fixture(scope="module")
def initialized_dataset():
    wrapper = EasyVQADataset(split="train", initialize_raw=True)
    wrapper.initialize_for_training()
    return wrapper


def test_get_item(initialized_dataset: EasyVQADataset):
    element: EasyVQAElement = initialized_dataset[0]

    assert element.label is not None
    assert element.question is not None
