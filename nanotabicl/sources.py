"""Where a training step's batch comes from: crop tables, prior tables (the TabICLv2 prior), or a mix.

`prior_fraction` (config group `source`) is the share of steps with a prior batch: 0 = crop only (the default), 1 = prior only, in between = a mix.
Prior batches have random shapes (`prior_shapes: random`, TabICL-style) or the shapes of the crop sampler's batches (`prior_shapes: matched`: the
control for crop training, and the SCM half of the 50 % mix). The matched shapes are drawn with the training seed plus `shape_seed_offset`
(0: the crop run's own batches, step by step; 10000: fresh draws of the same sampler).
"""

import numpy as np
from torch.utils.data import DataLoader, IterableDataset


class CropStream(IterableDataset):
    """Endless stream of crop batches (seeded)."""

    def __init__(self, tables, batch_size: int, seed: int):
        super().__init__()
        self.tables, self.batch_size, self.seed = tables, batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield self.tables.sample_batch(rng, self.batch_size)


def choose_source(rng: np.random.Generator, prior_fraction: float) -> str:
    """"prior" or "crop" for one step. The coin is only flipped for a real mix, so a pure crop or prior run draws nothing."""
    if prior_fraction >= 1:
        return "prior"
    if prior_fraction <= 0:
        return "crop"
    return "prior" if rng.random() < prior_fraction else "crop"


def build_streams(cfg, crop, n_steps: int) -> dict:
    """{"crop": iterator, "prior": iterator} of batches for the sources the run uses. `cfg` is the Hydra config (source, recipe, seed)."""
    src, rec, seed = cfg.source, cfg.recipe, cfg.seed
    streams = {}
    if src.prior_fraction < 1:
        if crop is None:
            raise ValueError("this run needs data.table (crop tables) for its crop batches.")
        streams["crop"] = iter(DataLoader(CropStream(crop, rec.batch_size, seed), batch_size=None))
    if src.prior_fraction > 0:
        if src.prior_shapes == "matched":
            from nanotabicl.prior_tables import MatchedPriorTables, crop_batch_shapes

            shapes = crop_batch_shapes(crop, rec.batch_size, n_steps, seed + src.shape_seed_offset)
            print(f"prior tables with the crop batches' shapes: rows {min(s[0] for s in shapes)}-{max(s[0] for s in shapes)}, "
                  f"features {sorted(set(s[1] for s in shapes))}")
            streams["prior"] = iter(DataLoader(MatchedPriorTables(shapes, rec.batch_size, seed), batch_size=None,
                                               num_workers=src.num_workers))
        else:
            from nanotabicl.prior_tables import PriorTables

            ds = PriorTables(rec.batch_size, max_rows=src.prior_max_rows, max_features=src.prior_max_features, seed=seed)
            streams["prior"] = iter(DataLoader(ds, batch_size=None, num_workers=src.num_workers, persistent_workers=src.num_workers > 0))
    return streams
