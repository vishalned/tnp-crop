"""(Continued) pretraining of NanoTabICLv2 for regression.

    # continue pretraining the official TabICLv2 weights on WOFOST tables
    python -m nanotabicl.train --init official --data crop --table data/processed/wofost_wheat_daily.parquet

    # from scratch, on the TabICLv2 prior and WOFOST tables mixed half/half
    python -m nanotabicl.train --init scratch --size small --data mix --prior-fraction 0.5 --table ...

Training loop in the style of nanoTabPFN's `train.py`: every step draws one
batch of tables (crop tables and/or prior tables), standardizes the target
of each table with its context rows' mean/std (as TabICL's regressor does),
and minimizes the pinball loss of the 999 predicted quantiles on the query
rows -- the same objective as the official TabICL pretraining
(`tabicl/train/_run.py`). AdamW with linear warmup + cosine decay, gradient
clipping, bf16 autocast on CUDA.

Every `--eval-every` steps it runs the walk-forward evaluation on the val
years (see `evaluate.py`) and keeps the checkpoint with the lowest val RMSE.
Writes `last.pt`, `best.pt`, `config.json` and `log.csv` to `--out-dir`.
"""

import argparse
import contextlib
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from nanotabicl import checkpoint
from nanotabicl.crop_tables import CropTableConfig, CropTables


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss of [B, n, q] quantiles at levels linspace(0, 1, q + 2)[1:-1] against [B, n] targets."""
    q = quantiles.shape[-1]
    alphas = torch.linspace(0, 1, q + 2, device=quantiles.device, dtype=quantiles.dtype)[1:-1]
    errors = target.unsqueeze(-1) - quantiles
    return torch.maximum(alphas * errors, (alphas - 1) * errors).mean()


def standardize_targets(y: torch.Tensor, n_train: int) -> tuple:
    """y standardized per table with its context rows' statistics, plus (mean, std) to invert."""
    mean = y[:, :n_train].mean(dim=1, keepdim=True)
    std = y[:, :n_train].std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-6)
    return (y - mean) / std, mean, std


def autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else contextlib.nullcontext()


def forward_quantiles(model, x: torch.Tensor, y: torch.Tensor, n_train: int, device: str) -> tuple:
    """Quantile predictions for the query rows in standardized units, the standardized query targets,
    and the (mean, std) used."""
    y_std, mean, std = standardize_targets(y, n_train)
    with autocast(device):
        quantiles = model(x, y_std[:, :n_train]).float()
    return quantiles, y_std[:, n_train:], mean, std


@torch.no_grad()
def evaluate_walk_forward(model, tables: CropTables, years, device: str, max_context_rows: int = 4000) -> tuple:
    """Walk-forward evaluation on `years`: metrics dict and a per-row predictions DataFrame.
    Point prediction = mean of the predicted quantiles, back in the target's units."""
    import pandas as pd

    model.eval()
    records = []
    for table in tables.walk_forward_tables(years, max_context_rows=max_context_rows):
        x, y, n = table["x"].to(device), table["y"].to(device), table["n_train"]
        quantiles, _, mean, std = forward_quantiles(model, x, y, n, device)
        pred = (quantiles.mean(-1) * std + mean)[0].cpu().numpy()
        context_rows, query_rows = table["rows"][:n], table["rows"][n:]
        # baseline: each point's mean yield over its context years
        ctx_point_mean = pd.Series(tables.y[context_rows]).groupby(tables.point[context_rows]).mean()
        for row, p in zip(query_rows, pred):
            records.append({
                "row": int(row), "location_index": int(tables.point[row]), "country": table["country"],
                "year": int(table["year"]), "y_true": float(tables.y[row]), "y_pred": float(p),
                "baseline_point_mean": float(ctx_point_mean.get(tables.point[row], np.nan)),
            })
    model.train()
    preds = pd.DataFrame(records)
    if preds.empty:
        return {}, preds

    def scores(col):
        err = preds[col] - preds["y_true"]
        ss_tot = ((preds["y_true"] - preds["y_true"].mean()) ** 2).sum()
        return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()),
                "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}

    metrics = {"n_queries": len(preds), **scores("y_pred"),
               **{f"baseline_{k}": v for k, v in scores("baseline_point_mean").items()}}
    return metrics, preds


class CropStream(IterableDataset):
    def __init__(self, tables: CropTables, batch_size: int, seed: int):
        super().__init__()
        self.tables, self.batch_size, self.seed = tables, batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield self.tables.sample_batch(rng, self.batch_size)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    p = argparse.ArgumentParser(description="(Continued) pretraining of NanoTabICLv2 for regression.")
    p.add_argument("--init", default="official", help="official | scratch | path to a checkpoint (ours or official)")
    p.add_argument("--size", default="small", choices=list(checkpoint.SIZES), help="model size for --init scratch")
    p.add_argument("--data", default="crop", choices=["crop", "prior", "mix"])
    p.add_argument("--prior-fraction", type=float, default=0.5, help="share of prior batches with --data mix")
    p.add_argument("--table", default=None, help="processed WOFOST table (process_wofost_dataset.py output)")
    p.add_argument("--target", default="yield_t_per_ha")
    p.add_argument("--bucket-days", type=int, default=7, help="weather aggregation: 7 = weekly, 10 = dekadal")
    p.add_argument("--train-years", type=int, nargs=2, default=[2005, 2016])
    p.add_argument("--val-years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--crop-episodes", default="structured", choices=["structured", "random"],
                   help="structured: context = earlier years, queries = target year; random: TabICL-style random row split")
    p.add_argument("--min-points", type=int, default=10, help="points per table; regions with fewer are dropped")
    p.add_argument("--max-points", type=int, default=None, help="default: up to all points of a region")
    p.add_argument("--max-rows", type=int, default=4096, help="cap on rows per crop table (memory)")
    p.add_argument("--min-context-years", type=int, default=5)
    p.add_argument("--max-context-years", type=int, default=11)
    p.add_argument("--prior-max-rows", type=int, default=1024)
    p.add_argument("--prior-max-features", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="~1e-5..1e-4 when continuing from official, ~1e-3 from scratch")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers generating prior tables")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="logs/nanotabicl/run")
    p.add_argument("--hf-cache-dir", default=None, help="where to cache the official checkpoint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model, model_config = checkpoint.build_model(args.init, args.size, args.hf_cache_dir)
    if model_config.get("max_classes", 0) != 0:
        raise ValueError("This loop trains regression; use a regression checkpoint (max_classes=0).")
    model.to(args.device).train()
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, init={args.init}")

    crop = None
    if args.table:
        crop = CropTables(args.table, CropTableConfig(
            train_years=args.train_years, target=args.target, bucket_days=args.bucket_days,
            episodes=args.crop_episodes, min_points=args.min_points, max_points=args.max_points, max_rows=args.max_rows,
            min_context_years=args.min_context_years, max_context_years=args.max_context_years))
        sizes = {c: len(p) for c, p in crop.country_points.items()}
        print(f"crop table: {len(crop.df)} rows, {crop.features.shape[1]} features, points per region {sizes}"
              + (f", dropped (< {args.min_points} points): {crop.dropped_regions}" if crop.dropped_regions else ""))
    elif args.data != "prior":
        raise ValueError("--data crop/mix needs --table.")

    streams = {}
    if args.data in ("crop", "mix"):
        streams["crop"] = iter(DataLoader(CropStream(crop, args.batch_size, args.seed), batch_size=None))
    if args.data in ("prior", "mix"):
        from nanotabicl.prior_tables import PriorTables
        prior_ds = PriorTables(args.batch_size, max_rows=args.prior_max_rows, max_features=args.prior_max_features, seed=args.seed)
        streams["prior"] = iter(DataLoader(prior_ds, batch_size=None, num_workers=args.num_workers,
                                           persistent_workers=args.num_workers > 0))
    rng = np.random.default_rng(args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_path = os.path.join(args.out_dir, "log.csv")
    best_rmse = float("inf")
    running, t0 = [], time.time()
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["step", "source", "loss", "lr", "val_rmse", "val_r2", "val_baseline_rmse", "seconds"])
        for step in range(args.steps):
            source = "prior" if args.data == "prior" or (args.data == "mix" and rng.random() < args.prior_fraction) else "crop"
            batch = next(streams[source])
            x, y, n = batch["x"].to(args.device), batch["y"].to(args.device), int(batch["n_train"])
            quantiles, y_query, _, _ = forward_quantiles(model, x, y, n, args.device)
            loss = pinball_loss(quantiles, y_query)

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running.append(loss.item())

            val = {}
            if crop is not None and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
                val, _ = evaluate_walk_forward(model, crop, args.val_years, args.device)
                if val and val["rmse"] < best_rmse:
                    best_rmse = val["rmse"]
                    checkpoint.save(os.path.join(args.out_dir, "best.pt"), model, model_config, step=step + 1, val=val, args=vars(args))
                print(f"step {step + 1}: val rmse {val.get('rmse', float('nan')):.3f} r2 {val.get('r2', float('nan')):.3f} "
                      f"(baseline rmse {val.get('baseline_rmse', float('nan')):.3f})")
            if (step + 1) % args.log_every == 0 or val:
                print(f"step {step + 1}/{args.steps} | {source} | loss {np.mean(running):.4f} | lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"| {time.time() - t0:.0f}s")
                log.writerow([step + 1, source, np.mean(running), optimizer.param_groups[0]["lr"], val.get("rmse"),
                              val.get("r2"), val.get("baseline_rmse"), round(time.time() - t0, 1)])
                log_file.flush()
                running = []
    checkpoint.save(os.path.join(args.out_dir, "last.pt"), model, model_config, step=args.steps, args=vars(args))
    print(f"done; checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
