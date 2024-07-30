import logging
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoProcessor


from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.lightning_trainer import BLIP2PLModule, LightningFineTune
from ..lib.representations import (
    DatasetFactory,
    DatasetTypes,
    HFRepos,
    ModelTypes,
    load_evaluation_metrics,
)
from ..lib.types import ModuleConfig

DATA_DIR = "./data"
TRAIN_SPLIT = "train[:30]"
VAL_SPLIT = "val[:30]"


def dataset(split: str, load_raw: bool = False):
    dir = f"{DATA_DIR}/{DatasetTypes.EASY_VQA.value}"
    ds = EasyVQADataset(split=split, load_raw=load_raw)
    try:
        ds = ds.load(dir)
    except FileNotFoundError:
        ds = EasyVQADataset(split=split, load_raw=True)
        ds.initialize_for_training()
        ds.save(dir)
    return ds


@pytest.fixture(scope="module")
def train_dataset():
    return dataset(split=TRAIN_SPLIT)


@pytest.fixture(scope="module")
def val_dataset():
    return dataset(split=VAL_SPLIT)


@pytest.fixture(scope="module")
def processor():
    MODEL_ID = HFRepos.BLIP2_OPT.value
    return AutoProcessor.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def mock_model():
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.loss.item = MagicMock(return_value=0.5)  # Mock loss value
    mock_output.loss.return_value = 0.5
    mock_model.return_value = mock_output
    return mock_model


@pytest.fixture(scope="module")
def blip2_module(train_dataset, val_dataset, processor, mock_model):
    blip2 = ModelTypes.BLIP2.value
    easy_vqa = DatasetTypes.EASY_VQA.value
    config = ModuleConfig(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        model=mock_model,
        shuffle_train=True,
        metrics=[load_evaluation_metrics(model=blip2, dataset=easy_vqa)],
    )
    return BLIP2PLModule(config)


def test_dataset_factory():
    dataset = DatasetTypes.EASY_VQA.value
    train_args = {"split": "train", "classify": False, "load_raw": True}
    val_args = {"split": "val", "classify": False, "load_raw": True}
    train_ds, val_ds = DatasetFactory.create_dataset(dataset, train_args, val_args)
    assert len(train_ds) == 38575
    assert len(val_ds) == 9673


def test_training_iteration(blip2_module: BLIP2PLModule, caplog):
    caplog.set_level(logging.INFO)

    # Register testing logger and iterate over each batch
    with patch.object(blip2_module, "log") as mock_log:
        for idx, batch in enumerate(blip2_module.train_dataloader()):
            loss = blip2_module.training_step(batch, idx)

    # Now check the results
    batch_size = blip2_module.config.batch_size
    assert loss() == 0.5
    mock_log.assert_called_with("train_loss", 0.5, batch_size=batch_size)
    assert "Epoch 0, loss: 0.5" in caplog.text


@pytest.mark.skip
def test_validation_iteration(blip2_module: BLIP2PLModule, caplog):
    caplog.set_level(logging.INFO)

    # Register logging object and start iterating
    with patch.object(blip2_module, "log") as mock_log:
        for idx, batch in enumerate(blip2_module.val_dataloader()):
            scores = blip2_module.validation_step(batch, idx)

    batch_size = blip2_module.config.batch_size
    mock_log.assert_called_with("train_loss", 0.5, batch_size=batch_size)
    assert "Epoch 0, loss: 0.5" in caplog.text


@pytest.mark.skip
def test_lightning_finetune():
    finetuner = LightningFineTune()
    finetuner.finetune()
