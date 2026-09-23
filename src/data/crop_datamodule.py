import json
import os
from typing import Any, Dict, Optional, Sequence

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.crop_episode_dataset import (
    CropStore,
    EpisodeConfig,
    EpisodeSampler,
    EvalEpisodeDataset,
    TrainEpisodeDataset,
    collate_episodes,
    compute_norm_stats,
)
from src.data.components.crop_vocab import NUM_MODALITIES


class CropEpisodeDataModule(LightningDataModule):
    """Episodes of WOFOST-simulated crop seasons for the TNP-D, from a store
    built by `build_training_store.py`.

    Year split (by season/sowing year), fixed for the whole pipeline:
    - train pool (default 2005-2016): the only years training episodes draw
      context and target years from;
    - val (2017-2018): walk-forward monitoring during training -- target =
      a val year, context = years before it -- never used for gradients;
    - test (2019-2020): walk-forward evaluation after training only.

    Normalization statistics come from the train pool only, and are
    written to `norm_stats_out` (JSON) and kept in the datamodule's
    `state_dict`, so they are saved inside every Lightning checkpoint and
    restored with it. Pass `norm_stats_path` to reuse existing stats instead
    of recomputing them.

    One DataLoader item is one episode; `collate_episodes` pads them into
    batches of `batch_size` episodes.
    """

    def __init__(
        self,
        store_dir: str,
        train_years: Sequence[int] = (2005, 2016),
        val_years: Sequence[int] = (2017, 2018),
        test_years: Sequence[int] = (2019, 2020),
        countries: Sequence[str] = ("France", "Germany", "Belgium", "United Kingdom", "Denmark", "Netherlands"),
        min_points: int = 3,
        max_points: int = 10,
        min_context_years: int = 5,
        max_context_years: Optional[int] = None,
        profiles: Sequence[str] = ("weekly", "dekadal", "irregular"),
        static_variables: Sequence[str] = ("clay", "nitrogen", "ph", "soc", "water_holding_capacity", "elevation", "slope"),
        label_modalities: Sequence[str] = ("yield", "phenology_maturity"),
        window_days: int = 322,
        pre_season_days: int = 14,
        irregular_bucket_days: int = 7,
        irregular_max_gap_days: int = 10,
        require_maturity: bool = True,
        batch_size: int = 8,
        train_episodes_per_epoch: int = 2000,
        val_episodes: int = 200,
        test_episodes: int = 200,
        num_workers: int = 0,
        seed: int = 0,
        num_modalities: int = NUM_MODALITIES,
        norm_stats_path: Optional[str] = None,
        norm_stats_out: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        if num_modalities != NUM_MODALITIES:
            raise ValueError(f"num_modalities must be {NUM_MODALITIES} (the fixed crop vocabulary), got {num_modalities}.")
        self.store: Optional[CropStore] = None
        self.norm_stats: Optional[dict] = None
        self.data_train = self.data_val = self.data_test = None

    @staticmethod
    def _years(bounds: Sequence[int]) -> list:
        first, last = bounds
        return list(range(int(first), int(last) + 1))

    def episode_config(self) -> EpisodeConfig:
        h = self.hparams
        return EpisodeConfig(
            countries=list(h.countries),
            min_points=h.min_points,
            max_points=h.max_points,
            min_context_years=h.min_context_years,
            max_context_years=h.max_context_years,
            profiles=list(h.profiles),
            static_variables=list(h.static_variables),
            label_modalities=list(h.label_modalities),
            window_days=h.window_days,
            pre_season_days=h.pre_season_days,
            irregular_bucket_days=h.irregular_bucket_days,
            irregular_max_gap_days=h.irregular_max_gap_days,
            require_maturity=h.require_maturity,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.store is not None:
            return
        h = self.hparams
        self.store = CropStore(h.store_dir, self.episode_config())
        train_years = self._years(h.train_years)

        if self.norm_stats is None:  # not restored from a checkpoint
            if h.norm_stats_path:
                with open(h.norm_stats_path) as f:
                    self.norm_stats = json.load(f)
            else:
                self.norm_stats = compute_norm_stats(self.store, train_years)
        if h.norm_stats_out:
            os.makedirs(os.path.dirname(os.path.abspath(h.norm_stats_out)), exist_ok=True)
            with open(h.norm_stats_out, "w") as f:
                json.dump(self.norm_stats, f, indent=2)

        all_years = list(self.store.years)
        self.data_train = TrainEpisodeDataset(
            EpisodeSampler(self.store, self.norm_stats, train_years, train_years),
            episodes_per_epoch=h.train_episodes_per_epoch,
            seed=h.seed,
        )
        # walk-forward: target = a val/test year, context = any year before it
        self.data_val = EvalEpisodeDataset(
            EpisodeSampler(self.store, self.norm_stats, self._years(h.val_years), all_years), h.val_episodes, seed=h.seed + 1
        )
        self.data_test = EvalEpisodeDataset(
            EpisodeSampler(self.store, self.norm_stats, self._years(h.test_years), all_years), h.test_episodes, seed=h.seed + 2
        )

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_episodes,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test)

    def state_dict(self) -> Dict[str, Any]:
        return {"norm_stats": self.norm_stats}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.norm_stats = state_dict.get("norm_stats")
