import pytest

from ..lib.representations import DatasetTypes

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


def test_check_training_item_properties(initialize_train_dataset: EasyVQADataset):
    element = initialize_train_dataset[0]
    check_properties(element)


def test_check_validation_item_properties(initialize_val_dataset: EasyVQADataset):
    element = initialize_val_dataset[0]
    check_properties(element)


def check_properties(element):
    assert element["label"] is not None
    assert element["prompt"] is not None
    assert element["image"] is not None


def test_get_10_val_items():
    val_ds = EasyVQADataset(split="val[:10]")
    val_ds.initialize_for_training()
    assert len(val_ds) == 10


def test_save_dataset(initialize_train_dataset):
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    initialize_train_dataset.save(dir)

    vqa_dataset = initialize_train_dataset.load(dir)
    assert initialize_train_dataset.equals(vqa_dataset.dataset)


def test_load_dataset():
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    train_ds = EasyVQADataset(split="train")

    vqa_dataset = train_ds.load(dir)
    assert len(vqa_dataset) > 0


def test_error_thrown_for_nonexistent_file_load():
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    train_ds = EasyVQADataset(split="train[:300]", load_raw=False)

    with pytest.raises(FileNotFoundError) as exc_info:
        train_ds = train_ds.load(dir)

    # Since train[:300].pkl does not exist a FileNotFoundError is raised
    assert str(exc_info.value).startswith(
        "[Errno 2] No such file or directory:")
