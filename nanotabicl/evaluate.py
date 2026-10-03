"""Walk-forward evaluation of a NanoTabICLv2 checkpoint on WOFOST tables.

    python -m nanotabicl.evaluate --checkpoint logs/nanotabicl/run/best.pt --table data/processed/wofost_wheat_daily.parquet
    python -m nanotabicl.evaluate --checkpoint official --table ...   # zero-shot official TabICLv2 weights

For every country and eval year T (default: the test years 2019-2020), the
context is the country's points at every earlier year (one jitter per
cell) and the queries are the same points at T. Reports RMSE/MAE/R2 of the
yield and of a baseline (each point's mean yield over its context years);
writes `metrics.json` and `predictions.csv` to `--out-dir`.
"""

import argparse
import json
import os

import torch

from nanotabicl import checkpoint
from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.train import evaluate_walk_forward


def main():
    p = argparse.ArgumentParser(description="Walk-forward evaluation of NanoTabICLv2 on WOFOST tables.")
    p.add_argument("--checkpoint", required=True, help="our checkpoint (.pt), an official .ckpt, or 'official'")
    p.add_argument("--table", required=True)
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    p.add_argument("--target", default="yield_t_per_ha")
    p.add_argument("--bucket-days", type=int, default=None, help="default: the value the checkpoint was trained with, else 7")
    p.add_argument("--max-context-rows", type=int, default=4000)
    p.add_argument("--min-points", type=int, default=10, help="regions with fewer points are left out (as in training)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None, help="default: next to the checkpoint")
    p.add_argument("--hf-cache-dir", default=None)
    args = p.parse_args()

    model, _ = checkpoint.build_model(args.checkpoint, cache_dir=args.hf_cache_dir)
    trained = {}
    if os.path.exists(args.checkpoint):
        trained = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("args", {})
    bucket_days = args.bucket_days or trained.get("bucket_days", 7)
    tables = CropTables(args.table, CropTableConfig(target=args.target, bucket_days=bucket_days,
                                                    min_points=trained.get("min_points", args.min_points)))
    model.to(args.device)

    metrics, preds = evaluate_walk_forward(model, tables, args.years, args.device, args.max_context_rows)
    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.checkpoint)) if os.path.exists(args.checkpoint)
                               else os.path.join("logs", "nanotabicl", "eval_official"))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(map(str, args.years))
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    preds.to_csv(os.path.join(out_dir, f"predictions_{tag}.csv"), index=False)
    by_country = preds.assign(se=(preds.y_pred - preds.y_true) ** 2).groupby("country")["se"].mean() ** 0.5
    print(json.dumps(metrics, indent=2))
    print("RMSE per country:\n" + by_country.round(3).to_string())
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
