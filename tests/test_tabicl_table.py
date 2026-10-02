"""Tests for the TabICL tabular table builder, on a synthetic store.

Pure data-shaping checks only (no network, no `tabicl` import): the actual
fine-tuning loop is exercised by `scripts/smoke_test_tabicl.py`, which needs
the `tabicl` extra and downloads the pretrained checkpoint.
"""

import numpy as np
import pytest

from src.data.components.crop_episode_dataset import CropStore, EpisodeConfig
from src.data.components.crop_vocab import LABEL_COLUMNS, WEATHER_VARIABLES
from src.data.components.synthetic_crop_store import make_synthetic_store
from src.data.components.tabicl_table import ID_COLUMNS, TabICLTableConfig, build_table, split_table, table_to_xy

COUNTRIES = ("France", "Germany", "Belgium", "United Kingdom", "Denmark", "Netherlands")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store_dir = make_synthetic_store(str(tmp_path_factory.mktemp("store")))
    return CropStore(store_dir, EpisodeConfig(countries=COUNTRIES))


@pytest.fixture(scope="module")
def table(store):
    return build_table(store)


def test_one_row_per_usable_cell_jitter(store, table):
    expected = sum(len(idxs) for idxs in store.cell_jitters.values())
    assert len(table) == expected
    assert len(table) > 0


def test_weather_columns_match_window_and_bucket_size(store, table):
    cfg = store.config
    n_buckets = cfg.window_days // cfg.weekly_days
    weather_cols = [c for c in table.columns if c not in ID_COLUMNS and c not in LABEL_COLUMNS.values()]
    assert len(weather_cols) == n_buckets * len(WEATHER_VARIABLES) + len(store.static_columns())


def test_dekadal_profile_has_fewer_weather_columns_than_weekly(store):
    weekly = build_table(store, TabICLTableConfig(profile="weekly"))
    dekadal = build_table(store, TabICLTableConfig(profile="dekadal"))
    assert len(dekadal.columns) < len(weekly.columns)


def test_irregular_profile_is_rejected(store):
    with pytest.raises(ValueError, match="irregular"):
        build_table(store, TabICLTableConfig(profile="irregular"))


def test_label_columns_present_and_finite(table):
    for col in LABEL_COLUMNS.values():
        if col in table.columns:
            assert np.isfinite(table[col].to_numpy(dtype=np.float64)).all()


def test_split_table_is_year_disjoint_and_within_bounds(table):
    splits = split_table(table, train_years=(2005, 2016), val_years=(2017, 2018), test_years=(2019, 2020))
    bounds = {"train": (2005, 2016), "val": (2017, 2018), "test": (2019, 2020)}
    all_years = set()
    for name, df in splits.items():
        lo, hi = bounds[name]
        years = set(df["season_year"].unique())
        assert years and years.issubset(set(range(lo, hi + 1)))
        assert not (years & all_years)  # disjoint from the splits already seen
        all_years |= years


def test_table_to_xy_drops_id_and_other_label_columns(table):
    X, y = table_to_xy(table, target="yield_t_per_ha")
    assert not set(X.columns) & set(ID_COLUMNS)
    assert "maturity_days" not in X.columns  # the other configured label modality
    assert "yield_t_per_ha" not in X.columns
    assert (y == table["yield_t_per_ha"]).all()
    assert len(X) == len(table)
