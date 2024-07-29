import pytest

from ..lib.representations import DatasetTypes

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.visualization import show_images_with_captions


@pytest.fixture(scope="module")
def train_ds():
    # Get cached dataset is available, otherwise generate a new one
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    train_ds = EasyVQADataset(split="train[16:25]", load_raw=False)

    try:
        train_ds = train_ds.load(dir)
    except FileNotFoundError:
        train_ds = EasyVQADataset(split="train[16:25]", load_raw=True)
        train_ds.initialize_for_training()
        train_ds.save(dir)

    return train_ds


@pytest.fixture(scope="module")
def train_raw_ds(request):
    # Get cached dataset is available, otherwise generate a new one
    train_ds = EasyVQADataset(split="val[16:25]", load_raw=True)

    return train_ds


def test_show_train_samples(train_ds):
    elements = train_ds[:9]

    show_images_with_captions(
        images_or_paths=elements["image"], captions=elements["prompt"]
    )


def test_show_test_samples(train_raw_ds):
    elements = train_raw_ds[2:5]

    captions = [
        f"Question: {question} Answer: {answer}"
        for question, answer in zip(elements["question"], elements["answer"])
    ]

    show_images_with_captions(
        images_or_paths=elements["image"], captions=captions)
