"""The refactored sampler (samplers / transforms) reproduces the reference batches recorded with the pre-refactor flag-style code
(tests/golden/make_golden_v0.py, tag archive/pre-cleanup): same tables, same features, same yields, same noise draws, same mix coins."""

import hashlib
import json
import os
import tempfile

import numpy as np
import pytest

from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.prior_tables import crop_batch_shapes
from nanotabicl.samplers import LogUniformPoints
from nanotabicl.sources import choose_source
from nanotabicl.transforms import ObservationNoise, PermuteYields
from tests.test_nanotabicl import calendar_table

GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "golden", "batches_v0.json")))
BASE = dict(min_context_years=3, max_context_years=6)
VARIANTS = {  # name -> (config overrides, sampler, transforms)
    "uniform": ({}, None, []),
    "loguniform": ({}, LogUniformPoints(min_points=1), []),
    "permute": ({}, None, [PermuteYields()]),
    "noise": ({}, None, [ObservationNoise(seed=0)]),
    "loguniform_noise_permute": ({}, LogUniformPoints(min_points=1), [PermuteYields(), ObservationNoise(seed=3)]),
    "random_episodes": (dict(episodes="random"), None, []),
    "france_only": (dict(countries=["France"], min_points=10), None, []),
}


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def table_path():
    path = os.path.join(tempfile.mkdtemp(), "t.parquet")
    calendar_table(["France"] * 12 + ["Germany"] * 11 + ["Belgium"] * 4).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("name", list(VARIANTS))
def test_sampler_variant_reproduces_the_reference_batches(table_path, name):
    overrides, sampler, transforms = VARIANTS[name]
    tables = CropTables(table_path, CropTableConfig(**{**BASE, **overrides}), sampler=sampler, transforms=transforms)
    rng = np.random.default_rng(0)
    for ref in GOLDEN[name]:
        b = tables.sample_batch(rng, 2)
        assert list(b["x"].shape) == ref["x_shape"] and int(b["n_train"]) == ref["n_train"]
        assert sha(b["x"].numpy()) == ref["x"] and sha(b["y"].numpy()) == ref["y"]
        assert float(b["x"].sum()) == ref["x_sum"] and float(b["y"].sum()) == ref["y_sum"]
        if "noise_params" in ref:
            assert sha(b["noise_params"].numpy()) == ref["noise_params"]


def test_matched_shapes_and_the_mix_coin_reproduce_the_reference(table_path):
    crop = CropTables(table_path, CropTableConfig(**BASE), sampler=LogUniformPoints(min_points=1))
    assert [list(s) for s in crop_batch_shapes(crop, 2, 6, 0)] == GOLDEN["shapes_seed0"]
    assert [list(s) for s in crop_batch_shapes(crop, 2, 6, 10_000)] == GOLDEN["shapes_seed10000"]
    rng = np.random.default_rng(0)
    assert [choose_source(rng, 0.5) for _ in range(30)] == GOLDEN["coin_sequence_p05"]
    assert choose_source(rng, 0.0) == "crop" and choose_source(rng, 1.0) == "prior"
