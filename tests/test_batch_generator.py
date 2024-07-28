import pytest
from ..lib.datasets_qa.easyvqa import EasyVQADataset
from ..lib.lightning_trainer import BLIP2PLModule
from ..lib.types import ModuleConfig


@pytest.fixture(scope="module")
def train_dataset():
    dir = "./data/easyvqa"
    train_ds = EasyVQADataset(split="train[:10]", load_raw=False)
    try:
        train_ds = train_ds.load(dir)
    except FileNotFoundError:
        train_ds = EasyVQADataset(split="train[:10]", load_raw=True)
        train_ds.initialize_for_training()
        train_ds = train_ds.save(dir)

    return train_ds


def test_generate_batches(train_dataset):

    batch_size = 4
    config = ModuleConfig(train_dataset=train_dataset.dataset,
                          val_dataset=None, batch_size=batch_size)
    module = BLIP2PLModule(config)
    for batch in module.train_dataloader():
        assert len(batch) <= batch_size
