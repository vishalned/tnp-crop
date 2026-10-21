"""Tests for nanotabicl/cybench_features.py: hand-checked values, and the same
table as AgML-CY-Bench-Modeling's own feature code (skipped if `cybench`
isn't installed: `uv pip install -e ../AgML-CY-Bench-Modeling --no-deps`)."""

import datetime
import os
from functools import reduce

import numpy as np
import pandas as pd
import pytest

from nanotabicl.cybench_features import SOURCES, cybench_features

DAILY = ["tmin", "tmax", "tavg", "prec", "rad", "et0", "cwb", "fpar", "ssm", "rsm"]
# (sos, eos) per location: windows of 131, 141 and 151 days -> 5, 5 and 6 blocks
CALENDARS = [(100.0, 170.0), (95.0, 175.0), (90.0, 180.0)]


def calendar_table(calendars=CALENDARS, years=range(2005, 2009), jitters=2, seed=0):
    """Processed rows in process_wofost_dataset.py's calendar layout: day 0 =
    SOS - 60, NaN after each row's EOS."""
    rng = np.random.default_rng(seed)
    n_days = max(int(e) - (int(s) - 60) + 1 for s, e in calendars)
    rows = []
    for loc, (sos, eos) in enumerate(calendars):
        lat, lon, awc, bulk_density = rng.uniform(45, 55), rng.uniform(-2, 12), rng.uniform(10, 25), rng.uniform(1.1, 1.6)
        for year in years:
            start = datetime.date(year, 1, 1) + datetime.timedelta(days=int(sos) - 1 - 60)
            length = int(eos) - (int(sos) - 60) + 1
            for jitter in range(jitters):
                tavg = 10 + 10 * np.sin(np.arange(n_days) / 45) + rng.normal(0, 3, n_days)
                daily = {"tavg": tavg, "tmin": tavg - rng.uniform(2, 8, n_days), "tmax": tavg + rng.uniform(2, 8, n_days),
                         "prec": rng.exponential(2, n_days) * (rng.random(n_days) < 0.5), "rad": rng.uniform(2e6, 2.5e7, n_days),
                         "et0": rng.uniform(0.5, 5, n_days), "fpar": np.clip(np.arange(n_days) - 70, 0, 90) * rng.uniform(0.5, 1),
                         "ssm": rng.uniform(15, 40, n_days), "rsm": rng.uniform(250, 400, n_days)}
                daily["cwb"] = daily["prec"] - daily["et0"]
                row = {"location_index": loc, "crop": "maize", "year": year, "sowing_year": year, "jitter_index": jitter,
                       "latitude": lat, "longitude": lon, "awc": awc, "bulk_density": bulk_density,
                       "sos_doy": sos, "eos_doy": eos, "window_start": start.isoformat()}
                for name in DAILY:
                    values = daily[name].copy()
                    values[length:] = np.nan
                    row.update({f"{name}_d{d:03d}": v for d, v in enumerate(values)})
                rows.append(row)
    return pd.DataFrame(rows)


def test_hand_checked_values():
    df = calendar_table()
    x, names = cybench_features(df, drop_sparse=False)
    row = df.iloc[0]  # sos 100, eos 170: 131 days, blocks 0-3 full, block 4 = the first 11 days
    daily = {v: row[[f"{v}_d{d:03d}" for d in range(131)]].to_numpy(float) for v in DAILY}
    dates = pd.date_range(row["window_start"], periods=131)
    last30 = slice(101, 131)
    assert x[0, names.index("prec_sum_0")] == pytest.approx(daily["prec"][last30].sum(), rel=1e-5)
    assert x[0, names.index("tmin_min_1")] == pytest.approx(daily["tmin"][71:101].min(), rel=1e-5)
    assert x[0, names.index("prec_sum_4")] == pytest.approx(daily["prec"][:11].sum(), rel=1e-5, abs=1e-6)
    gdd = np.clip(daily["tavg"] - 10, 0, 35)
    assert x[0, names.index("gdd_sum_0")] == pytest.approx(gdd[last30].sum(), rel=1e-5)
    assert x[0, names.index("cum_gdd_max_0")] == pytest.approx(gdd.sum(), rel=1e-5)  # running sum from SOS - 60
    dekad = np.isin(dates.day, (1, 11, 21))
    assert x[0, names.index("fpar_mean_0")] == pytest.approx(daily["fpar"][last30][dekad[last30]].mean(), rel=1e-5)
    assert x[0, names.index("cum_fpar_max_0")] == pytest.approx(daily["fpar"][dekad].sum(), rel=1e-5)
    assert np.isnan(x[0, names.index("prec_sum_5")])  # only the 151-day calendar has a 6th block
    assert x[0, names.index("sos_sin")] == pytest.approx(np.sin(2 * np.pi * 100 / 365), rel=1e-5)
    # column order: meteo, fpar, soil moisture (each sorted by name, then block), statics
    assert names[0] == "cum_cwb_mean_0" and names.index("cum_fpar_max_0") < names.index("rsm_mean_0") < names.index("sos_sin")

    x, names = cybench_features(df)  # block 5 exists for 1/3 of the rows: > 5% NaN, dropped
    assert not any(n.endswith("_5") for n in names) and not np.isnan(x).any()
    x_mid, names_mid = cybench_features(df, horizon="middle-of-season")
    assert len(names_mid) < len(names)


def _benchmark_table(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """The same rows through the benchmark's own code (DataFactory's steps)."""
    from omegaconf import OmegaConf

    from cybench.datasets.alignment import align_to_crop_season_window, compute_crop_season_window
    from cybench.datasets.data_factory import DataFactory
    from cybench.datasets.feature_transformation import feature_transform

    df = df.assign(adm_id=df["location_index"].astype(str) + "_j" + df["jitter_index"].astype(str))
    calendar = df[["adm_id", "sos_doy", "eos_doy"]].drop_duplicates("adm_id").rename(columns={"sos_doy": "sos", "eos_doy": "eos"})
    seasons = compute_crop_season_window(calendar, int(df["year"].min()), int(df["year"].max()), "sos-60", horizon)
    crop = OmegaConf.create({"gdd": {"tbase": 10.0, "upper_limit": 35.0}})

    parts = []
    for resolution, created, stats in SOURCES.values():
        select = [v for v in stats if v not in {c[0] for c in created}]
        long = []
        for _, row in df.iterrows():
            n = len([c for c in df.columns if c.startswith("tavg_d")])
            frame = pd.DataFrame({"adm_id": row["adm_id"], "date": pd.date_range(row["window_start"], periods=n)})
            for v in select:
                frame[v] = row[[f"{v}_d{d:03d}" for d in range(n)]].to_numpy(float)
            frame = frame.dropna()  # days after this row's EOS: no data
            if resolution == "dekad":
                frame = frame[frame["date"].dt.day.isin([1, 11, 21])]
            long.append(frame)
        ts = pd.concat(long, ignore_index=True)
        ts["year"] = ts["date"].dt.year
        ts = align_to_crop_season_window(ts[["adm_id", "year", "date"] + select], seasons)
        if created:
            ts = DataFactory._create_features(
                ts, OmegaConf.create([{"name": n, "type": t, "input": i} for n, t, i in created]), crop)
        ts = ts.merge(seasons[["adm_id", "year", "end_of_sequence_date"]], on=["adm_id", "year"], how="left")
        ts = DataFactory._aggregate_time_series(ts, stats, 30)
        parts.append(DataFactory._tabularize(ts, aggregate=30))

    statics = df.drop_duplicates("adm_id")
    non_temporal = [
        feature_transform(feature_transform(calendar.copy(), {"type": "cyclical", "feature": "sos", "period": 365}),
                          {"type": "cyclical", "feature": "eos", "period": 365}),
        feature_transform(statics[["adm_id", "latitude", "longitude"]].copy(), {"type": "spherical", "lat": "latitude", "lon": "longitude"}),
        statics[["adm_id", "awc", "bulk_density"]],
    ]
    non_temporal = reduce(lambda a, b: pd.merge(a, b, on="adm_id"), non_temporal).set_index("adm_id")
    table = (pd.concat(parts, axis=1).reset_index().merge(non_temporal.reset_index(), on="adm_id", how="left")
             .set_index(["adm_id", "year"]))
    table, _ = DataFactory._drop_sparse_tabular_columns(table, 0.05)
    keys = pd.MultiIndex.from_arrays([df["adm_id"], df["year"]])
    return table.loc[keys]


@pytest.mark.parametrize("horizon", ["eos", "middle-of-season"])
def test_same_table_as_the_benchmark(horizon):
    pytest.importorskip("cybench")
    df = calendar_table()
    x, names = cybench_features(df, horizon=horizon)
    expected = _benchmark_table(df, horizon)
    assert names == list(expected.columns)
    np.testing.assert_allclose(x, expected.to_numpy(np.float64), rtol=1e-5, atol=1e-6)


WOFOST_MAIZE = os.environ.get("WOFOST_MAIZE_TABLE", "/projects/prjs1788/tnp-crop/processed/wofost_maize_daily.parquet")
CYBENCH_DATA = os.environ.get("CYBENCH_DATA", "/projects/prjs1788/CropBench/data/cybench/cybench-data")


@pytest.mark.slow
def test_wofost_maize_feature_names_equal_cybench_de():
    """The real tables: WOFOST maize features (names, in order) = the benchmark's for CY-Bench maize DE."""
    pytest.importorskip("cybench")
    if not (os.path.exists(WOFOST_MAIZE) and os.path.isdir(os.path.join(CYBENCH_DATA, "maize", "DE"))):
        pytest.skip("needs the processed WOFOST maize table and the CY-Bench data")
    from nanotabicl.cybench_check import benchmark_feature_table

    df = pd.read_parquet(WOFOST_MAIZE)
    _, names = cybench_features(df)
    assert names == list(benchmark_feature_table(CYBENCH_DATA, "maize", "DE").columns)
