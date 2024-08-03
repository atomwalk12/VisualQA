import pytest
from transformers import AutoProcessor, Blip2Processor

from ..lib.types import TorchTrainerConfig

from .utils import get_index

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.lightning_trainer import BLIP2PLModule
from ..lib.types import ModuleConfig
from ..lib.representations import (
    DatasetTypes,
    HFRepos,
    ModelTypes,
    load_evaluation_metrics,
)


@pytest.fixture(scope="module")
def train_dataset():
    # Get cached dataset is available, otherwise generate a new one
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    train_ds = EasyVQADataset(split="train[:30]", load=False)
    try:
        train_ds = train_ds.load()
    except FileNotFoundError:
        train_ds = EasyVQADataset(split="train[:30]", load=True)
        train_ds.save()

    return train_ds


@pytest.fixture(scope="module")
def val_dataset():
    # Get cached dataset is available, otherwise generate a new one
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
    val_ds = EasyVQADataset(split="val[:30]", load=False)
    try:
        val_ds = val_ds.load()
    except FileNotFoundError:
        val_ds = EasyVQADataset(split="val[:30]", load=True)
        val_ds.initialize_for_training()
        val_ds.save()

    return val_ds


@pytest.fixture(scope="module")
def processor():
    MODEL_ID = HFRepos.BLIP2_OPT.value

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return processor


@pytest.fixture(scope="module")
def blip2_module(train_dataset, val_dataset, processor):
    model = ModelTypes.BLIP2.value
    ds = DatasetTypes.EASY_VQA.value
    config = ModuleConfig(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        model=None,
        shuffle_train=False,
        torch_hyperparameters=TorchTrainerConfig(),
        model_name="blip2",
        metrics=[load_evaluation_metrics(model, ds)],
    )

    module = BLIP2PLModule(config)
    return module


def test_batch_generation(blip2_module):
    for batch in blip2_module.train_dataloader():
        assert isinstance(batch.data, dict)


def test_load_training_dataset_and_check_output_values(blip2_module):
    for batch in blip2_module.train_dataloader():
        assert all(
            [
                key in ["input_ids", "attention_mask", "pixel_values", "labels"]
                for key in batch
            ]
        )


def test_load_validation_dataset_and_check_output_values(
    blip2_module: BLIP2PLModule, val_dataset: EasyVQADataset
):
    for batch in blip2_module.val_dataloader():
        assert all(
            [key in ["input_ids", "attention_mask", "pixel_values"] for key in batch]
        )
        assert all([key in val_dataset.answer_space for key in batch['labels]']])


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


def test_decode_batch_and_check_against_validation_examples(
    blip2_module: BLIP2PLModule,
    val_dataset: EasyVQADataset,
    processor: Blip2Processor,
):
    batch_size = blip2_module.config.batch_size
    idx = 0
    for batch in blip2_module.val_dataloader():
        _, labels = batch

        # retrieve correct index
        end = get_index(idx, batch_size, val_dataset)

        # Check decoded text
        assert labels not in val_dataset[idx:end]["prompt"]
        idx += batch_size
