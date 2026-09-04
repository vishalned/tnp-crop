from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.synthetic_episode_dataset import SyntheticEpisodeDataset


class TNPDataModule(LightningDataModule):
    """`LightningDataModule` serving synthetic weather-like episodes so the
    TNP-D architecture can be smoke-tested end-to-end before the real
    WOFOST-simulated episode dataset exists.

    Each dataset item is already a full batch (see `SyntheticEpisodeDataset`),
    so `batch_size=None` is passed to every `DataLoader`.
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_modalities: int = 6,
        min_context: int = 5,
        max_context: int = 40,
        min_target: int = 5,
        max_target: int = 40,
        num_train_batches: int = 500,
        num_val_batches: int = 50,
        num_test_batches: int = 50,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[SyntheticEpisodeDataset] = None
        self.data_val: Optional[SyntheticEpisodeDataset] = None
        self.data_test: Optional[SyntheticEpisodeDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        common = dict(
            batch_size=self.hparams.batch_size,
            num_modalities=self.hparams.num_modalities,
            min_context=self.hparams.min_context,
            max_context=self.hparams.max_context,
            min_target=self.hparams.min_target,
            max_target=self.hparams.max_target,
        )
        if self.data_train is None:
            self.data_train = SyntheticEpisodeDataset(
                num_batches=self.hparams.num_train_batches, seed=self.hparams.seed, **common
            )
        if self.data_val is None:
            self.data_val = SyntheticEpisodeDataset(
                num_batches=self.hparams.num_val_batches, seed=self.hparams.seed + 1, **common
            )
        if self.data_test is None:
            self.data_test = SyntheticEpisodeDataset(
                num_batches=self.hparams.num_test_batches, seed=self.hparams.seed + 2, **common
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=None, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=None, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=None, num_workers=self.hparams.num_workers)
