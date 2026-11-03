"""Reference batches of the crop-table sampler, generated with the code BEFORE the Hydra refactor (git tag archive/pre-cleanup).

    python -m tests.golden.make_golden_v0          # writes tests/golden/batches_v0.json

Historical: it uses the old flag-style CropTableConfig (points_sampling, obs_noise, permute_yields), which no longer exists after the
refactor. tests/test_golden.py builds the same variants with the new components and must reproduce these numbers exactly.
"""
import hashlib
import json
import os
import tempfile

import numpy as np

from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.prior_tables import crop_batch_shapes
from tests.test_nanotabicl import calendar_table

BASE = dict(min_context_years=3, max_context_years=6)
VARIANTS = {
    "uniform": dict(),
    "loguniform": dict(points_sampling="loguniform", min_table_points=1),
    "permute": dict(permute_yields=True),
    "noise": dict(obs_noise=True, obs_noise_seed=0),
    "loguniform_noise_permute": dict(points_sampling="loguniform", min_table_points=1, obs_noise=True, obs_noise_seed=3, permute_yields=True),
    "random_episodes": dict(episodes="random"),
    "france_only": dict(countries=["France"], min_points=10),
}


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    regions = ["France"] * 12 + ["Germany"] * 11 + ["Belgium"] * 4
    path = os.path.join(tempfile.mkdtemp(), "t.parquet")
    calendar_table(regions).to_parquet(path, index=False)
    out = {}
    for name, extra in VARIANTS.items():
        tables = CropTables(path, CropTableConfig(**{**BASE, **extra}))
        rng = np.random.default_rng(0)
        batches = []
        for _ in range(3):
            b = tables.sample_batch(rng, 2)
            rec = {"x_shape": list(b["x"].shape), "n_train": int(b["n_train"]), "x": sha(b["x"].numpy()), "y": sha(b["y"].numpy()),
                   "x_sum": float(b["x"].sum()), "y_sum": float(b["y"].sum())}
            if "noise_params" in b:
                rec["noise_params"] = sha(b["noise_params"].numpy())
            batches.append(rec)
        out[name] = batches
    crop = CropTables(path, CropTableConfig(**BASE, points_sampling="loguniform", min_table_points=1))
    out["shapes_seed0"] = [list(s) for s in crop_batch_shapes(crop, 2, 6, 0)]
    out["shapes_seed10000"] = [list(s) for s in crop_batch_shapes(crop, 2, 6, 10_000)]
    rng = np.random.default_rng(0)
    out["coin_sequence_p05"] = ["prior" if rng.random() < 0.5 else "crop" for _ in range(30)]
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "batches_v0.json"), "w"), indent=1)
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
