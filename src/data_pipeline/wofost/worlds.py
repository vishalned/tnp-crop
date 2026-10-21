"""Sampled "worlds": one location + one set of hidden settings, simulated for every year.

A world is (location, settings). The settings are drawn once per world and stay fixed over all its
years, so a region's yield/feature relation is the same in every year of it, as in a real region:

- cultivar: one of the grain-maize varieties of the WOFOST parameter file;
- production: "PP" (potential = irrigated) with probability = the region's irrigated share, else
  "WLP" (water-limited, the current model);
- crop-parameter factors: one multiplicative factor per parameter, drawn with a Latin hypercube over
  the worlds of a location (table parameters: only the value column is scaled);
- a sowing shift around the crop calendar's start of season.

Per year inside a world only a small sowing jitter is drawn on top of the world's shift. None of the
settings is a model input; they are saved per world (`worlds.parquet`).

Every random draw comes from its own stream, seeded by (seed, location, component), so
- the same seed gives the same worlds whatever the worker count or location subset;
- datasets that differ in one setting (A_w, B, C, D, E: `configs/wofost_worlds/`) share every other
  draw, e.g. B and C have the same cultivars and sowing dates.
"""

import datetime
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc

# grain-maize varieties of WOFOST_crop_parameters (wofost81, maize.yaml); they differ only in TSUM1/TSUM2.
# Left out: Maize_VanHeemst_1988 (tropical, storage organ = whole cob) and Fodder_maize_nl.
GRAIN_MAIZE_VARIETIES = ["Grain_maize_201", "Grain_maize_202", "Grain_maize_203", "Grain_maize_204",
                         "Grain_maize_205"]

DEFAULT_FACTOR_RANGES = {
    "TSUM1": [0.9, 1.1],  # narrower: the cultivar choice already varies the temperature sums
    "TSUM2": [0.9, 1.1],
    "AMAX_REF": [0.85, 1.15],  # WOFOST 8.1's max leaf assimilation (it has no AMAXTB)
    "EFFTB": [0.85, 1.15],
    "SPAN": [0.85, 1.15],
    "SLATB": [0.85, 1.15],
    "RDMCR": [0.85, 1.15],
}
TABLE_PARAMETERS = {"AMAXTB", "EFFTB", "SLATB"}  # PCSE flat [x0, y0, x1, y1, ...] lists

# one random stream per component, so switching one setting on doesn't change the others' draws
_STREAMS = {"cultivar": 1, "production": 2, "factors": 3, "sowing_shift": 4, "year_jitter": 5}


@dataclass
class WorldsConfig:
    """One dataset's settings (`configs/wofost_worlds/dataset_*.yaml`)."""

    name: str = "A"
    crop: str = "maize"
    locations_csv: str = "data/raw/locations/locations_maize.csv"
    extra_locations_csvs: list = field(default_factory=list)  # dataset E: appended after locations_csv
    output_dir: str = "data/raw/wofost_worlds/A"
    start_year: int = 2005
    end_year: int = 2020
    seed: int = 42
    max_locations: Optional[int] = None  # only the first N locations
    location_indices: Optional[list] = None  # or exactly these (rows of the locations list)
    workers: Optional[int] = None

    # sowing: "legacy" = the original runner (dataset A): per year `num_jitters` distinct sowing dates
    # within +-sowing_jitter_days; "world" = one shift per world (+-world_shift_days) + a per-year
    # jitter (+-year_jitter_days), one run per world and year
    sowing_scheme: str = "world"
    num_jitters: int = 3
    sowing_jitter_days: int = 10
    worlds_per_location: int = 10
    world_shift_days: int = 10
    year_jitter_days: int = 3

    sample_cultivar: bool = False
    cultivars: list = field(default_factory=lambda: list(GRAIN_MAIZE_VARIETIES))
    default_cultivar: str = "Grain_maize_201"

    sample_factors: bool = False
    factor_ranges: dict = field(default_factory=lambda: dict(DEFAULT_FACTOR_RANGES))
    rdmcr_step_cm: float = 5.0  # sampled max rooting depths are rounded to this (min. split-layer thickness)

    irrigation: bool = False
    irrigation_share_csv: Optional[str] = None  # columns: region_id, irrigated_share
    irrigation_region_column: str = "country"  # locations-CSV column matched to region_id
    default_irrigated_share: float = 0.0  # regions not in the CSV (all regions without one)

    max_immature_fraction: float = 0.5  # drop a world if more years than this don't mature by EOS ...
    min_yield_t_ha: float = 0.5
    max_low_yield_fraction: float = 0.5  # ... or have a yield below min_yield_t_ha
    max_failed_fraction: float = 0.5  # ... or failed to simulate

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "WorldsConfig":
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.sowing_scheme not in ("legacy", "world"):
            raise ValueError(f"sowing_scheme must be 'legacy' or 'world', got {self.sowing_scheme!r}.")
        if self.sowing_scheme == "legacy" and (self.sample_cultivar or self.sample_factors or self.irrigation):
            raise ValueError("sowing_scheme 'legacy' (dataset A) runs the original setup only; "
                             "cultivar/factor/irrigation sampling needs sowing_scheme 'world'.")
        for name, (lo, hi) in self.factor_ranges.items():
            if not 0 < lo <= hi:
                raise ValueError(f"factor range of {name} must satisfy 0 < low <= high, got {[lo, hi]}.")
        if self.end_year < self.start_year:
            raise ValueError(f"end_year ({self.end_year}) must be >= start_year ({self.start_year}).")

    @property
    def years(self) -> list:
        return list(range(self.start_year, self.end_year + 1))


def load_locations(config: WorldsConfig) -> pd.DataFrame:
    """The locations list: `locations_csv`, then each of `extra_locations_csvs`; `location_index` =
    row number in that list (so the original CSV's rows keep their index)."""
    frames = [pd.read_csv(p) for p in [config.locations_csv] + list(config.extra_locations_csvs)]
    locations = pd.concat(frames, ignore_index=True)
    locations["location_index"] = np.arange(len(locations))
    return locations


def selected(locations: pd.DataFrame, config: WorldsConfig) -> pd.DataFrame:
    if config.location_indices is not None:
        return locations[locations["location_index"].isin(config.location_indices)]
    if config.max_locations is not None:
        return locations.iloc[: config.max_locations]
    return locations


def load_irrigated_shares(config: WorldsConfig) -> dict:
    """region_id -> irrigated maize share (0-1) from `irrigation_share_csv`; empty without one."""
    if not config.irrigation_share_csv:
        return {}
    df = pd.read_csv(config.irrigation_share_csv)
    return dict(zip(df["region_id"].astype(str), df["irrigated_share"].astype(float)))


def _rng(config: WorldsConfig, location_index: int, stream: str) -> np.random.Generator:
    return np.random.default_rng([config.seed, int(location_index), _STREAMS[stream]])


def sample_factors(config: WorldsConfig, location_index: int, n_worlds: int) -> list:
    """Per world a dict parameter -> factor: a Latin hypercube over the location's worlds (each
    parameter's range split in n_worlds strata, one world per stratum), scaled to the ranges."""
    names = list(config.factor_ranges)
    if not config.sample_factors or not names:
        return [{} for _ in range(n_worlds)]
    seed = int(_rng(config, location_index, "factors").integers(2**32))
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n_worlds)
    lo = np.array([config.factor_ranges[n][0] for n in names])
    hi = np.array([config.factor_ranges[n][1] for n in names])
    values = qmc.scale(unit, lo, hi) if np.any(hi > lo) else np.tile(lo, (n_worlds, 1))
    return [dict(zip(names, map(float, row))) for row in values]


def sample_worlds(location: dict, config: WorldsConfig, irrigated_shares: Optional[dict] = None) -> list:
    """The `worlds_per_location` worlds of one location: list of dicts with the hidden settings
    (`world_id`, cultivar, production, `factor_<P>`, sowing shift) and the per-year sowing offsets
    (`sowing_offsets`: year -> days from the season start)."""
    loc = int(location["location_index"])
    n = config.worlds_per_location
    region = str(location.get(config.irrigation_region_column, ""))
    share = (irrigated_shares or {}).get(region, config.default_irrigated_share) if config.irrigation else 0.0

    cultivar_rng = _rng(config, loc, "cultivar")
    cultivars = (cultivar_rng.choice(config.cultivars, size=n) if config.sample_cultivar
                 else [config.default_cultivar] * n)
    production_u = _rng(config, loc, "production").random(n)
    shifts = _rng(config, loc, "sowing_shift").integers(-config.world_shift_days, config.world_shift_days + 1, size=n)
    jitters = _rng(config, loc, "year_jitter").integers(-config.year_jitter_days, config.year_jitter_days + 1,
                                                        size=(n, len(config.years)))
    factors = sample_factors(config, loc, n)

    worlds = []
    for w in range(n):
        worlds.append({
            "world_id": f"L{loc:05d}_W{w:02d}",
            "world_index": w,
            "location_index": loc,
            "longitude": location["longitude"],
            "latitude": location["latitude"],
            "region": region,
            "cultivar": str(cultivars[w]),
            "production": "PP" if production_u[w] < share else "WLP",
            "irrigated_share": float(share),
            **{f"factor_{name}": f for name, f in factors[w].items()},
            "sowing_shift_days": int(shifts[w]),
            "seed": config.seed,
            "sowing_offsets": {year: int(shifts[w] + jitters[w, i]) for i, year in enumerate(config.years)},
        })
    return worlds


def scale_crop_parameters(base: dict, factors: dict, soil_depth_cm: Optional[float] = None,
                          rdmcr_step_cm: float = 5.0) -> dict:
    """PCSE overrides (name -> value) for a world: each parameter in `factors` times its factor.
    Tables (`TABLE_PARAMETERS`, flat [x0, y0, x1, y1, ...]) have only their value column (y) scaled;
    the x column (development stage, or temperature for EFFTB) is left as is. `RDMCR` is rounded to
    `rdmcr_step_cm` and clipped to [rdmcr_step_cm, soil depth]."""
    out = {}
    for name, f in factors.items():
        value = base[name]
        if name in TABLE_PARAMETERS:
            out[name] = [v * f if i % 2 else v for i, v in enumerate(value)]
        elif name == "RDMCR":
            depth = max(rdmcr_step_cm, round(value * f / rdmcr_step_cm) * rdmcr_step_cm)
            out[name] = float(min(depth, soil_depth_cm) if soil_depth_cm else depth)
        else:
            out[name] = float(value * f)
    return out


def season_end(year: int, sos_doy: int, eos_doy: int) -> datetime.date:
    """The crop calendar's end of season for sowing year `year` (EOS < SOS = the next year)."""
    eos_year = year + 1 if eos_doy < sos_doy else year
    return datetime.date(eos_year, 1, 1) + datetime.timedelta(days=int(eos_doy) - 1)


def world_filter(runs: pd.DataFrame, config: WorldsConfig) -> tuple:
    """Whether to keep a world, from the manifest rows of its years (columns status, maturity_date,
    yield_kg_per_ha, year, sos_doy, eos_doy). Returns (keep, reasons, fractions); reasons are
    "failed", "immature" (no maturity by the crop calendar's EOS) and "low_yield"."""
    n = len(runs)
    ok = runs["status"] == "success"
    matured = pd.Series(False, index=runs.index)
    for i, r in runs[ok].iterrows():
        if isinstance(r["maturity_date"], str) and r["maturity_date"]:
            end = season_end(int(r["year"]), int(r["sos_doy"]), int(r["eos_doy"]))
            matured[i] = datetime.date.fromisoformat(r["maturity_date"]) <= end
    low = ok & (runs["yield_kg_per_ha"].fillna(0) / 1000.0 < config.min_yield_t_ha)
    fractions = {"failed": float((~ok).mean()), "immature": float((ok & ~matured).sum() / n),
                 "low_yield": float(low.sum() / n)}
    limits = {"failed": config.max_failed_fraction, "immature": config.max_immature_fraction,
              "low_yield": config.max_low_yield_fraction}
    reasons = [k for k, frac in fractions.items() if frac > limits[k]]
    return not reasons, reasons, fractions


def save_config(config: WorldsConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
