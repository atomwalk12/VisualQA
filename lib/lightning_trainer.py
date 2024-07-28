import lightning as L
from torch.utils.data import DataLoader

from .types import ModuleConfig


class BLIP2PLModule(L.LightningModule):
    def __init__(self, config: ModuleConfig):
        super().__init__()
        self.config = config

    def collate_fn(self, batch):
        return batch

    def train_dataloader(self):
        assert self.config.train_dataset is not None
        return DataLoader(
            self.config.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        assert self.config.val_dataset is not None
        return DataLoader(
            self.config.val_dataset,
            collate_fn=collate_fn,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )


def collate_fn(batch):
    return batch
