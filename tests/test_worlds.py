"""Tests for the sampled WOFOST worlds (src/data_pipeline/wofost/worlds.py and its runner)."""

import numpy as np
import pandas as pd
import pytest

from nanotabicl.crop_tables import CropTableConfig, CropTables
from src.data_pipeline.wofost import generate_worlds_dataset as runner
from src.data_pipeline.wofost.utils_wofost.pcse_runner import layer_thicknesses, merge_split_layers, split_soil_layer_at
from src.data_pipeline.wofost.worlds import (
    DEFAULT_FACTOR_RANGES,
    WorldsConfig,
    sample_factors,
    sample_worlds,
    scale_crop_parameters,
    world_filter,
)
from tests.test_nanotabicl import calendar_table

LOCATION = {"location_index": 7, "longitude": 4.3, "latitude": 50.2, "country": "Belgium", "sos_doy": 102,
            "eos_doy": 309}


def config(**kw):
    return WorldsConfig(**{"worlds_per_location": 10, "sample_cultivar": True, "sample_factors": True, **kw})


# ----- factor sampling and table scaling --------------------------------------------------------------


def test_factors_are_a_latin_hypercube_inside_their_ranges():
    cfg = config()
    factors = sample_factors(cfg, 7, cfg.worlds_per_location)
    assert len(factors) == 10 and all(set(f) == set(DEFAULT_FACTOR_RANGES) for f in factors)
    for name, (lo, hi) in cfg.factor_ranges.items():
        values = np.array([f[name] for f in factors])
        assert ((values >= lo) & (values <= hi)).all()
        strata = np.floor((values - lo) / (hi - lo) * 10).astype(int)  # Latin hypercube: one world per tenth
        assert sorted(np.minimum(strata, 9)) == list(range(10))
    assert factors == sample_factors(cfg, 7, 10)  # seeded
    assert factors != sample_factors(cfg, 8, 10)  # but different per location
    assert sample_factors(config(sample_factors=False), 7, 3) == [{}, {}, {}]


def test_scaling_scales_only_the_value_column_of_tables():
    base = {"SLATB": [0.0, 0.0026, 0.78, 0.0012, 2.0, 0.0012], "EFFTB": [0.0, 0.45, 40.0, 0.45],
            "TSUM1": 695.0, "SPAN": 33.0, "RDMCR": 100.0}
    out = scale_crop_parameters(base, {"SLATB": 1.1, "EFFTB": 0.9, "TSUM1": 1.05, "SPAN": 0.85, "RDMCR": 1.13})
    assert out["SLATB"][0::2] == base["SLATB"][0::2]  # development stages unchanged
    assert np.allclose(out["SLATB"][1::2], np.array(base["SLATB"][1::2]) * 1.1)
    assert out["EFFTB"][0::2] == [0.0, 40.0] and np.allclose(out["EFFTB"][1::2], [0.405, 0.405])
    assert out["TSUM1"] == pytest.approx(729.75) and out["SPAN"] == pytest.approx(28.05)
    assert out["RDMCR"] == 115.0  # 113 cm rounded to the 5 cm step
    assert scale_crop_parameters({"RDMCR": 190.0}, {"RDMCR": 1.15}, soil_depth_cm=200.0)["RDMCR"] == 200.0
    assert base["SLATB"] == [0.0, 0.0026, 0.78, 0.0012, 2.0, 0.0012]  # input untouched


def test_soil_layer_is_split_at_the_rooting_depth():
    layers = [{"Thickness": t, "SMfromPF": [t]} for t in (5, 10, 15, 30, 40, 100)]
    soil = {"RDMSOL": 200, "SoilProfileDescription": {"SoilLayers": layers}}
    split = split_soil_layer_at(soil, 85.0)["SoilProfileDescription"]["SoilLayers"]
    assert [layer["Thickness"] for layer in split] == [5, 10, 15, 30, 25, 15, 100]
    assert split[4]["SMfromPF"] == split[5]["SMfromPF"] == [40]  # same properties
    assert len(soil["SoilProfileDescription"]["SoilLayers"]) == 6  # input untouched
    assert split_soil_layer_at(soil, 100.0) == soil  # already a boundary

    # outputs of the split profile are merged back to the original layers
    thick, orig = layer_thicknesses(split_soil_layer_at(soil, 85.0)), layer_thicknesses(soil)
    wc = {f"WC_layer{i}": [0.1 * t] for i, t in enumerate(thick)}  # 0.1 cm water per cm of soil
    sm = {f"SM_layer{i}": [0.2 if i == 4 else 0.3] for i in range(len(thick))}
    merged = merge_split_layers(pd.DataFrame({**wc, **sm}), thick, orig)
    assert [c for c in merged if c.startswith("WC")] == [f"WC_layer{i}" for i in range(6)]
    assert np.allclose([merged[f"WC_layer{i}"][0] for i in range(6)], [0.1 * t for t in orig])
    assert merged["SM_layer4"][0] == pytest.approx((0.2 * 25 + 0.3 * 15) / 40)


# ----- settings are constant across years within a world -----------------------------------------------


def test_world_settings_are_drawn_once_per_world():
    cfg = config(year_jitter_days=3, world_shift_days=10)
    worlds = sample_worlds(LOCATION, cfg)
    assert len({w["world_id"] for w in worlds}) == 10
    for w in worlds:
        assert w["cultivar"] in cfg.cultivars and w["production"] == "WLP"  # irrigation off
        offsets = np.array(list(w["sowing_offsets"].values()))
        assert sorted(w["sowing_offsets"]) == cfg.years
        assert (np.abs(offsets - w["sowing_shift_days"]) <= 3).all() and abs(w["sowing_shift_days"]) <= 10
    # datasets that differ in one setting share the other draws (B -> C: same cultivars and sowing)
    b = sample_worlds(LOCATION, config(sample_factors=False))
    assert [(w["cultivar"], w["sowing_offsets"]) for w in b] == [(w["cultivar"], w["sowing_offsets"]) for w in worlds]
    irrigated = sample_worlds(LOCATION, config(irrigation=True, default_irrigated_share=1.0))
    assert all(w["production"] == "PP" for w in irrigated)


def test_every_year_of_a_world_runs_with_the_same_settings(tmp_path, monkeypatch):
    """The runner passes each world's cultivar, production and parameter overrides unchanged to all
    of its years (WOFOST itself is replaced by a stub)."""
    calls = []

    def fake_episode(**kw):
        calls.append(kw)
        return {"daily_path": "d.csv", "summary_path": "s.json",
                "summary": {"sowing_date": "2005-04-12", "yield_kg_per_ha": 8000.0, "final_dvs": 2.0,
                            "flowering_date": "2005-07-20", "maturity_date": "2005-10-01", "awc": 0.2,
                            "bulk_density": 1.3}}

    class FakeCropData(dict):
        pass

    base = {"TSUM1": 695.0, "TSUM2": 800.0, "AMAX_REF": 80.0, "EFFTB": [0.0, 0.45, 40.0, 0.45], "SPAN": 33.0,
            "SLATB": [0.0, 0.0026, 2.0, 0.0012], "RDMCR": 100.0}
    monkeypatch.setattr(runner, "generate_wofost_episode", fake_episode)
    monkeypatch.setattr(runner, "_prefetch_weather_for_window", lambda *a, **k: None)
    monkeypatch.setattr(runner, "load_crop_data_provider", lambda *a: FakeCropData(base))
    pd.DataFrame([{k: v for k, v in LOCATION.items() if k != "location_index"}] * 2).to_csv(tmp_path / "loc.csv",
                                                                                            index=False)
    cfg = config(locations_csv=str(tmp_path / "loc.csv"), output_dir=str(tmp_path / "out"), worlds_per_location=3,
                 start_year=2005, end_year=2009, workers=1)
    out = runner.generate_worlds_dataset(cfg)

    assert len(calls) == 2 * 3 * 5  # locations x worlds x years
    by_world = {}
    for kw in calls:
        by_world.setdefault(kw["file_tag"], []).append(kw)
    for world_id, runs in by_world.items():
        assert len(runs) == 5 and sorted(kw["year"] for kw in runs) == list(range(2005, 2010))
        for key in ("variety_name", "production", "param_overrides", "sowing_doy", "eos_doy"):
            assert all(kw[key] == runs[0][key] for kw in runs), key
    worlds = pd.read_parquet(tmp_path / "out" / "worlds.parquet")
    assert len(worlds) == 6 and worlds["kept"].all()
    assert {"factor_TSUM1", "cultivar", "production", "sowing_shift_days", "seed"} <= set(worlds.columns)
    assert (out["manifest"].groupby("world_id")["cultivar"].nunique() == 1).all()


# ----- filter ----------------------------------------------------------------------------------------


def _runs(maturity, yields, status=None):
    n = len(maturity)
    return pd.DataFrame({"status": status or ["success"] * n, "maturity_date": maturity, "yield_kg_per_ha": yields,
                         "year": [2005 + i for i in range(n)], "sos_doy": 102, "eos_doy": 309})


def test_filter_drops_immature_low_yield_and_failed_worlds():
    cfg = WorldsConfig()  # > 50 % of years, < 0.5 t/ha
    ok = _runs([f"{2005 + i}-10-01" for i in range(4)], [8000.0] * 4)
    assert world_filter(ok, cfg)[0]
    # EOS = day 309 = 5 Nov: maturity on 20 Nov is too late; 2 of 4 years late = 50 %, not more -> kept
    late2 = _runs(["2005-11-20", "2006-11-20", "2007-10-01", "2008-10-01"], [8000.0] * 4)
    assert world_filter(late2, cfg)[0]
    late3 = _runs(["2005-11-20", None, "2007-11-20", "2008-10-01"], [8000.0] * 4)
    keep, reasons, fractions = world_filter(late3, cfg)
    assert not keep and reasons == ["immature"] and fractions["immature"] == 0.75
    low = _runs([f"{2005 + i}-10-01" for i in range(4)], [100.0, 200.0, 300.0, 8000.0])
    assert world_filter(low, cfg)[1] == ["low_yield"]
    failed = _runs([None] * 4, [None] * 4, status=["failed", "failed", "failed", "success"])
    assert "failed" in world_filter(failed, cfg)[1]
    assert world_filter(low, WorldsConfig(max_low_yield_fraction=0.8))[0]  # configurable


# ----- training tables: at most one world per site ---------------------------------------------------


def test_training_tables_have_at_most_one_world_per_site(tmp_path):
    regions = ["France"] * 36 + ["Belgium"] * 12  # 3 worlds per site: France 12 sites, Belgium 4
    df = calendar_table(regions, jitters=1)
    df["site_index"] = df["location_index"] // 3
    path = tmp_path / "worlds.parquet"
    df.to_parquet(path, index=False)
    for episodes in ("structured", "random"):
        tables = CropTables(str(path), CropTableConfig(episodes=episodes, min_context_years=3, max_context_years=6))
        assert set(tables.country_points) == {"France"} and tables.dropped_regions == {"Belgium": 4}
        rng = np.random.default_rng(0)
        for _ in range(20):
            b = tables.sample_batch(rng, batch_size=2)
            for x in b["x"]:
                rows = [int(np.flatnonzero((tables.features == r.numpy()).all(1))[0]) for r in x]
                sites = {tables.site_of[p] for p in tables.point[rows]}
                assert len(sites) == len(set(tables.point[rows])) <= 12  # one world per site
