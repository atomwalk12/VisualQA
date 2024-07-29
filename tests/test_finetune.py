import logging
from unittest.mock import MagicMock, patch
import pytest
from transformers import AutoProcessor

from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.lightning_trainer import BLIP2PLModule, LightningFineTune
from ..lib.representations import DatasetFactory, DatasetTypes, ModelTypes
from ..lib.types import ModuleConfig


@pytest.fixture(scope="module")
def train_dataset():
    # Get cached dataset is available, otherwise generate a new one
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
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
    dir = f"./data/{DatasetTypes.EASY_VQA.value}"
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
    MODEL_ID = ModelTypes.BLIP2_OPT

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return processor


@pytest.fixture(scope="module")
def blip2_module(train_dataset, val_dataset, processor, mock_model):
    config = ModuleConfig(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        model=mock_model,
        shuffle_train=True,
    )

    module = BLIP2PLModule(config)
    return module


@pytest.fixture(scope="module")
def mock_model():
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.loss.item = MagicMock(return_value=0.5)  # Mock loss value
    mock_output.loss.return_value = 0.5
    mock_model.return_value = mock_output
    return mock_model


def test_dataset_factory():
    # Let's assume the dataset easy easy-vqa
    dataset = DatasetTypes.EASY_VQA.value

    # define factory parameters
    train_args = {"split": "train", "classify": False, "load_raw": True}
    val_args = {"split": "val", "classify": False, "load_raw": True}

    # retrieve the dataset from the factory
    train_ds, val_ds = DatasetFactory.create_dataset(dataset, train_args, val_args)
    assert len(train_ds) == 38575
    assert len(val_ds) == 9673


def test_training_iteration(blip2_module: BLIP2PLModule, caplog):
    caplog.set_level(logging.INFO)

    for idx, batch in enumerate(blip2_module.train_dataloader()):
        with patch.object(blip2_module, "log") as mock_log:
            loss = blip2_module.training_step(batch, idx)

    batch_size = blip2_module.config.batch_size
    assert loss() == 0.5
    mock_log.assert_called_with("train_loss", 0.5, batch_size=batch_size)
    assert "Epoch 0, loss: 0.5" in caplog.text


@pytest.mark.skip
def test_lightning_finetune():
    finetuner = LightningFineTune()
    finetuner.finetune()
