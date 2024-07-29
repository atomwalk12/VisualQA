import pytest
from transformers import AutoProcessor, Blip2Processor

from tests.utils import get_index

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
def val_dataset():
    # Get cached dataset is available, otherwise generate a new one
    dir = "./data/easyvqa"
    val_ds = EasyVQADataset(split="val[:30]", load_raw=False)
    try:
        val_ds = val_ds.load(dir)
    except FileNotFoundError:
        val_ds = EasyVQADataset(split="val[:30]", load_raw=True)
        val_ds.initialize_for_training()
        val_ds.save(dir)

    return val_ds


@pytest.fixture(scope="module")
def processor():
    MODEL_ID = "Salesforce/blip2-opt-2.7b"

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return processor


@pytest.fixture(scope="module")
def blip2_module(train_dataset, val_dataset, processor):
    config = ModuleConfig(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        model=None,
        shuffle_train=False,
    )

    module = BLIP2PLModule(config)
    return module


def test_batch_generation(blip2_module):
    for batch in blip2_module.train_dataloader():
        assert isinstance(batch.data, dict)


def test_load_training_dataset_and_check_output_values(blip2_module):
    for batch in blip2_module.train_dataloader():
        assert all(
            [key in ["input_ids", "attention_mask", "pixel_values"] for key in batch]
        )


def test_load_validation_dataset_and_check_output_values(
    blip2_module: BLIP2PLModule, val_dataset: EasyVQADataset
):
    for batch in blip2_module.val_dataloader():
        inputs = batch["inputs"]
        labels = batch["labels"]
        assert all(
            [key in ["input_ids", "attention_mask", "pixel_values"] for key in inputs]
        )
        assert all([key in val_dataset.answer_space for key in labels])


def test_decode_batch_and_check_against_training_examples(
    blip2_module: BLIP2PLModule,
    train_dataset: EasyVQADataset,
    processor: Blip2Processor,
):
    batch_size = blip2_module.config.batch_size
    idx = 0
    for batch in blip2_module.train_dataloader():
        # Decoded processed data
        text = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

        # retrieve correct index
        end = get_index(idx, batch_size, train_dataset)

        # Check decoded text
        assert text == train_dataset[idx:end]["prompt"]
        idx += batch_size
