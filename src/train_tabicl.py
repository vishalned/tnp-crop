"""Continued pretraining (fine-tuning) of TabICLv2 on the WOFOST crop store.

Unlike `src/train.py` (a Lightning `Trainer` over the TNP-D datamodule and
`LightningModule`), this drives `tabicl`'s own `FinetunedTabICLRegressor.fit`
loop directly: that method already *is* a full training loop (AdamW,
cosine-with-warmup, AMP, early stopping, HF-checkpoint-schema saving), so
there is nothing for a Lightning `Trainer` to add here -- see the "TabICL"
section of the README for why this stays outside `train.py` instead of
becoming another `configs/model/*.yaml`.

What this script does own (so the run is still Hydra-config-driven like the
rest of the repo): building the flat feature table from the store
(`src/data/components/tabicl_table.py`), the train/val/test split, and
instantiating `FinetunedTabICLRegressor` from `configs/tabicl.yaml`.
"""

import json
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import rootutils
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# src.utils' RankedLogger requires this (normally set by lightning's own Trainer/Fabric
# init, which this single-process, non-Lightning entrypoint never constructs).
rank_zero_only.rank = 0

from src.data.components.crop_episode_dataset import CropStore, EpisodeConfig
from src.data.components.tabicl_table import TabICLTableConfig, build_table, split_table, table_to_xy
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


def run(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the table, fine-tune, evaluate on the test split.

    :param cfg: A DictConfig configuration composed by Hydra (`configs/tabicl.yaml`).
    :return: A tuple of (metrics dict, dict of instantiated objects).
    """
    if cfg.get("seed") is not None:
        np.random.seed(cfg.seed)

    log.info(f"Loading store <{cfg.store_dir}>")
    episode_cfg = EpisodeConfig(
        countries=list(cfg.countries),
        static_variables=list(cfg.static_variables),
        label_modalities=list(cfg.label_modalities),
        require_maturity=cfg.require_maturity,
        window_days=cfg.window_days,
        pre_season_days=cfg.pre_season_days,
    )
    store = CropStore(cfg.store_dir, episode_cfg)

    log.info(f"Flattening store into a tabular ({cfg.profile}) feature table")
    table = build_table(store, TabICLTableConfig(profile=cfg.profile, label_modalities=list(cfg.label_modalities)))
    splits = split_table(table, cfg.train_years, cfg.val_years, cfg.test_years)
    for name, df in splits.items():
        log.info(f"  {name}: {len(df)} rows")

    X_train, y_train = table_to_xy(splits["train"], cfg.target)
    X_val, y_val = table_to_xy(splits["val"], cfg.target)
    X_test, y_test = table_to_xy(splits["test"], cfg.target)

    log.info(f"Instantiating <{cfg.finetune._target_}>")
    model = hydra.utils.instantiate(cfg.finetune)

    log.info("Fine-tuning...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, output_dir=cfg.paths.output_dir)

    log.info("Evaluating on the test split...")
    preds = model.predict(X_test)
    metrics = {
        "test/mse": float(mean_squared_error(y_test, preds)),
        "test/mae": float(mean_absolute_error(y_test, preds)),
        "test/r2": float(r2_score(y_test, preds)),
    }
    log.info(f"Test metrics: {metrics}")
    with open(f"{cfg.paths.output_dir}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, {"cfg": cfg, "store": store, "model": model}


@hydra.main(version_base="1.3", config_path="../configs", config_name="tabicl.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for TabICL fine-tuning.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
