"""Held-out simulated walk-forward tables from the densified WOFOST table (same table builder as training).

Every table: one country, one query year T in the WOFOST val/test years (2017-2020), queries = a chunk of the
country's points at T, context = the same points at all earlier years (`CropTables.walk_forward_tables`,
the one used for validation/testing). N tables are drawn at random (seeded) and saved in shards.

    poetry run python /.../build_sim_tables.py --table .../wofost_eu_grid_dense/processed/wofost_maize_daily.parquet \
        --out /projects/prjs1788/tnp-crop/feature_importance/sim_tables --n-tables 100 --shards 4
"""

import argparse
import json

import numpy as np

from nanotabicl.crop_tables import CropTableConfig, CropTables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019, 2020])
    p.add_argument("--n-tables", type=int, default=100)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tables = CropTables(args.table, CropTableConfig())  # the training config (train-year normaliser, eos)
    all_tables = list(tables.walk_forward_tables(args.years, max_context_rows=4000, seed=args.seed))
    print(f"{len(all_tables)} held-out tables available")
    pick = np.sort(np.random.default_rng(args.seed).choice(len(all_tables), min(args.n_tables, len(all_tables)), replace=False))
    index = []
    for s, ids in enumerate(np.array_split(pick, args.shards)):
        arrays = {"columns": np.array(tables.feature_names)}
        for i in ids:
            t = all_tables[i]
            tid = f"{t['country']}_{t['year']}_{i}"
            arrays[f"{tid}_x"] = t["x"][0].numpy().astype(np.float32)
            arrays[f"{tid}_y"] = t["y"][0].numpy().astype(np.float32)
            arrays[f"{tid}_n_train"] = np.array(t["n_train"])
            index.append({"table_id": tid, "country": t["country"], "year": int(t["year"]), "shard": s,
                          "n_context": int(t["n_train"]), "n_query": int(len(t["rows"]) - t["n_train"])})
        np.savez_compressed(f"{args.out}/sim_tables_{s}.npz", **arrays)
    with open(f"{args.out}/index.json", "w") as f:
        json.dump(index, f, indent=1)
    print("saved", len(index), "tables;", len(tables.feature_names), "features")


if __name__ == "__main__":
    main()
