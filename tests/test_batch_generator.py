import pytest
from transformers import AutoProcessor

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.lightning_trainer import BLIP2PLModule
from ..lib.types import ModuleConfig


@pytest.fixture(scope="module")
def train_dataset():
    # Get cached dataset is available, otherwise generate a new one
    dir = "./data/easyvqa"
    train_ds = EasyVQADataset(split="train[:30]", load_raw=False)
    try:
        train_ds = train_ds.load(dir)
    except FileNotFoundError:
        train_ds = EasyVQADataset(split="train[:30]", load_raw=True)
        train_ds.initialize_for_training()
        train_ds.save(dir)

    return train_ds


@pytest.fixture(scope="module")
def processor():
    MODEL_ID = "Salesforce/blip2-opt-2.7b"

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return processor


def test_generate_batches(train_dataset, processor):
    batch_size = 4
    config = ModuleConfig(
        train_dataset=train_dataset.dataset,
        processor=processor,
        model=None,
        val_dataset=None,
        batch_size=batch_size,
    )
    module = BLIP2PLModule(config)
    for batch in module.train_dataloader():
        assert isinstance(batch.data, dict)


def test_train_collate_batches(train_dataset, processor):
    config = ModuleConfig(
        train_dataset=train_dataset.dataset,
        processor=processor,
        val_dataset=None,
        model=None,
    )
    module = BLIP2PLModule(config)

    for batch in module.train_dataloader():
        assert all(
            [key in ["input_ids", "attention_mask", "pixel_values"]
                for key in batch]
        )
