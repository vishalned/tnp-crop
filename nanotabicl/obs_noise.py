"""Observational noise on fPAR, surface soil moisture and root-zone soil moisture, applied to the training tables.

Real satellite / reanalysis series are noisy; WOFOST's are not. Per training table, once:

    fpar:  sigma ~ U(0, 0.3), bg ~ U(0.05, 0.2)    fpar_obs = clip(fpar * (1 + N(0, sigma)) + bg, 0, 1)
    ssm:   sigma ~ U(0, 0.3)                       ssm_obs  = ssm * (1 + N(0, sigma))
    rsm:   sigma ~ U(0, 0.3)                       rsm_obs  = rsm * (1 + N(0, sigma))

(sigma and bg are drawn once per table, the noise N(0, sigma) per value, i.e. per observation: per dekad for fPAR, per day for
ssm and rsm). Everything else (weather, cwb, et0, statics, yield) is unchanged. The table's fPAR is in % (CY-Bench's unit), so the
formula runs in % (bg x 100, clip to [0, 100]).

The noise is applied to the daily series *before* CY-Bench's features are built from them, so the features stay what the
benchmark computes from a noisy series: `fpar_mean`, `cum_fpar_max` (the running sum of the noisy dekads), `ssm_mean`, `rsm_mean`,
per 30-day block. Only these columns of a table change; they are recomputed with `cybench_features`' own aggregation and the
table's normaliser. With sigma = 0 and bg = 0 the result equals the clean features (tested). Validation/test tables stay clean.
"""

import numpy as np
import pandas as pd

from nanotabicl.cybench_features import (
    DEKAD_DAYS,
    START_OFFSET_DAYS,
    WINDOW_DAYS,
    _aggregate,
    _daily,
    cutoff_days,
    season_dates,
    variable_of,
)

FPAR_MAX = 100.0  # fPAR in %
# (source of the observation days, feature variable, statistic) of every recomputed feature
FEATURES = (("dekad", "fpar", "mean"), ("dekad", "fpar", "cum_max"), ("daily", "ssm", "mean"), ("daily", "rsm", "mean"))
NAMES = {("fpar", "mean"): "fpar_mean", ("fpar", "cum_max"): "cum_fpar_max", ("ssm", "mean"): "ssm_mean", ("rsm", "mean"): "rsm_mean"}


class ObservationNoise:
    def __init__(self, df: pd.DataFrame, feature_names: list, normalizer: dict, horizon: str = "eos",
                 sigma_max: float = 0.3, bg_range: tuple = (0.05, 0.2), seed: int = 0):
        self.sigma_max, self.bg_range = sigma_max, bg_range
        self.rng = np.random.default_rng([seed, 7919])
        # the window and the 30-day blocks exactly as `cybench_features` builds them
        sos_date, eos_date = season_dates(df)
        start = sos_date - np.timedelta64(START_OFFSET_DAYS, "D")
        season_length = (eos_date - sos_date).astype(int)
        end = eos_date - cutoff_days(horizon, season_length).astype("timedelta64[D]")
        window_start = pd.to_datetime(df["window_start"]).values.astype("datetime64[D]")
        self.series = {v: _daily(df, v).astype(np.float32) for v in ("fpar", "ssm", "rsm")}
        n_days = self.series["fpar"].shape[1]
        dates = window_start[:, None] + np.arange(n_days).astype("timedelta64[D]")[None, :]
        in_window = (dates >= start[:, None]) & (dates <= end[:, None])
        block = (end[:, None] - dates).astype(int) // WINDOW_DAYS
        day_of_month = (dates - dates.astype("datetime64[M]")).astype(int) + 1
        self.observed = {"daily": in_window, "dekad": in_window & np.isin(day_of_month, DEKAD_DAYS)}
        self.n_blocks = {s: int(block[o].max()) + 1 for s, o in self.observed.items()}
        self.block = block.astype(np.int16)
        # which columns of the table are recomputed: (source, variable, statistic, block) -> column index
        index = {n: j for j, n in enumerate(feature_names)}
        self.columns = []
        for source, var, stat in FEATURES:
            for b in range(self.n_blocks[source]):
                name = f"{NAMES[(var, stat)]}_{b}"
                if name in index:
                    self.columns.append((source, var, stat, b, index[name], variable_of(name)))
        self.normalizer = normalizer

    def draw(self) -> np.ndarray:
        """(sigma_fpar, bg, sigma_ssm, sigma_rsm) of one table."""
        return np.array([self.rng.uniform(0, self.sigma_max), self.rng.uniform(*self.bg_range),
                         self.rng.uniform(0, self.sigma_max), self.rng.uniform(0, self.sigma_max)])

    def noisy_series(self, rows: np.ndarray, params: np.ndarray, rng: np.random.Generator = None) -> dict:
        """The rows' daily fpar / ssm / rsm with the table's noise (NaN outside the observation days)."""
        rng = rng or self.rng
        sigma_fpar, bg, sigma_ssm, sigma_rsm = params
        out = {}
        fpar = self.series["fpar"][rows].astype(np.float64)
        noisy = np.clip(fpar * (1 + rng.normal(0, 1, fpar.shape) * sigma_fpar) + bg * FPAR_MAX, 0, FPAR_MAX)
        out["fpar"] = np.where(self.observed["dekad"][rows], noisy, np.nan)
        for var, sigma in (("ssm", sigma_ssm), ("rsm", sigma_rsm)):
            x = self.series[var][rows].astype(np.float64)
            out[var] = np.where(self.observed["daily"][rows], x * (1 + rng.normal(0, 1, x.shape) * sigma), np.nan)
        return out

    def apply(self, rows: np.ndarray, params: np.ndarray) -> dict:
        """{column index: normalised feature values of the rows} for the recomputed features (NaN -> 0)."""
        series = self.noisy_series(rows, params)
        cum = np.where(np.isnan(series["fpar"]), np.nan, np.nancumsum(np.where(self.observed["dekad"][rows], series["fpar"], np.nan), axis=1))
        block = self.block[rows]
        values = {}
        for source, var, stat, b, col, norm_name in self.columns:
            groups = self.observed[source][rows] & (block == b)
            x = cum if stat == "cum_max" else series[var]
            v = _aggregate(x, groups, "max" if stat == "cum_max" else "mean")
            mean, std = self.normalizer[norm_name]
            values[col] = np.nan_to_num((v - mean) / std if std > 0 else v * 0.0, nan=0.0).astype(np.float32)
        return values
