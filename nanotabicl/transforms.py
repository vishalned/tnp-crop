"""Transforms of a sampled training batch (Hydra config group `transform`, collected under `transforms`).

A transform is called as `transform(batch, idx, rng)` right after `CropTables.sample_batch` has built the batch
{"x": [B, rows, features], "y": [B, rows], "n_train"} (idx = [B, rows] row indices of the tables) and returns the batch. `bind(tables)` is
called once by `CropTables` with the table object, for transforms that need its data. The transforms run in the order of the config.
"""

import numpy as np
import torch


class PermuteYields:
    """Sanity test: shuffle the yields among all rows (context and query) of every table: no feature -> yield signal is left."""

    def bind(self, tables) -> None:
        pass

    def __call__(self, batch: dict, idx: np.ndarray, rng: np.random.Generator) -> dict:
        y = batch["y"].numpy()
        batch["y"] = torch.from_numpy(np.stack([row[rng.permutation(len(row))] for row in y]))
        return batch


class ObservationNoise:
    """Observational noise on fPAR, ssm and rsm (nanotabicl/obs_noise.py): sigma (and for fPAR a background offset bg) once per table, the noise
    per value on the daily series, the CY-Bench features recomputed. The draws are returned as batch["noise_params"] for the training log."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.noise = None

    def bind(self, tables) -> None:
        from nanotabicl.obs_noise import ObservationNoise as Noise

        self.noise = Noise(tables.df, tables.feature_names, tables.normalizer, horizon=tables.config.horizon, seed=self.seed)

    def __call__(self, batch: dict, idx: np.ndarray, rng: np.random.Generator) -> dict:
        x = batch["x"].numpy()  # a copy made by the sampler: written in place
        params = np.stack([self.noise.draw() for _ in range(len(idx))])
        for b in range(len(idx)):
            for col, values in self.noise.apply(idx[b], params[b]).items():
                x[b, :, col] = values
        batch["noise_params"] = torch.from_numpy(params)
        return batch
