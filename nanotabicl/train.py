"""(Continued) pretraining of NanoTabICLv2 for regression, configured with Hydra (configs/).

    # continue pretraining the official TabICLv2 weights on WOFOST tables: one experiment = one config file
    python -m nanotabicl.train experiment=exp_2a_loguniform_points seed=0

    # any value can be overridden, groups swapped and transforms added on the command line
    python -m nanotabicl.train data.table=<parquet> sampler=loguniform source=mix_matched +transform@transforms.noise=obs_noise

    # from scratch, on the TabICLv2 prior and WOFOST tables mixed half/half (recipe=scratch is an untested starting point)
    python -m nanotabicl.train model=scratch_small recipe=scratch source=mix data.table=<parquet>

Training loop in the style of nanoTabPFN's `train.py`: every step draws one batch of tables (crop tables and/or prior tables, `sources.py`),
standardizes the target of each table with its context rows' mean/std (as TabICL's regressor does), and minimizes the pinball loss of the 999
predicted quantiles on the query rows -- the same objective as the official TabICL pretraining (`tabicl/train/_run.py`). AdamW with linear
warmup + cosine decay, gradient clipping, bf16 autocast on CUDA.

Memory: a batch of `recipe.batch_size` tables is processed `recipe.micro_batch_size` tables at a time, with gradients accumulated over the
micro-batches before one optimizer step (same update as the whole batch at once). With `recipe.checkpoint_activations` every transformer block
keeps only its input and recomputes the rest in the backward pass (less memory, ~30% slower). The log reports the peak GPU memory.

Every `recipe.eval_every` steps it runs the walk-forward evaluation on the val years (see `evaluate.py`) and keeps the checkpoint with the lowest
val RMSE. Writes `last.pt`, `best.pt`, `config.json` (the resolved config), `log.csv` (and `noise_params.csv` with observational noise) to
`out_dir`, plus `step<N>.pt` for every N in `recipe.save_steps`. `recipe.stop_at` ends the run early with the learning-rate schedule of
`recipe.steps`.

How the batches are built is chosen by config groups, not flags: `sampler` (points per table), `transforms` (noise, permuted yields, ...) and
`source` (crop / prior / mix). See configs/ and nanotabicl/{samplers,transforms,sources}.py.
"""

import contextlib
import csv
import functools
import json
import math
import os
import time

import hydra
import numpy as np
import torch
import torch.utils.checkpoint
from omegaconf import DictConfig, OmegaConf

from nanotabicl import checkpoint
from nanotabicl.crop_tables import CropTableConfig, CropTables
from nanotabicl.model import TransformerBlock
from nanotabicl.sources import build_streams, choose_source


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


def checkpoint_activations(model) -> None:
    """Activation checkpointing: every transformer block stores only its input in the forward pass and
    recomputes its internals in the backward pass. Gradients are unchanged."""
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.forward = functools.partial(torch.utils.checkpoint.checkpoint, module.forward, use_reentrant=False)


def train_step(model, optimizer, x: torch.Tensor, y: torch.Tensor, n_train: int, device: str,
               micro_batch_size: int, grad_clip: float) -> float:
    """One optimizer step on a batch of tables, `micro_batch_size` tables at a time with gradients
    accumulated (the same update as the whole batch at once). Returns the batch's mean loss."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, x.shape[0], micro_batch_size):
        xs, ys = x[start:start + micro_batch_size], y[start:start + micro_batch_size]
        quantiles, y_query, _, _ = forward_quantiles(model, xs, ys, n_train, device)
        loss = pinball_loss(quantiles, y_query) * (xs.shape[0] / x.shape[0])  # weighted so the sum = batch mean
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return total


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


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def build_crop_tables(cfg: DictConfig):
    """The crop tables of a run: the data config, the sampler and the transforms of the config."""
    d = cfg.data
    if not d.table:
        return None
    sampler = hydra.utils.instantiate(cfg.sampler)
    transforms = [hydra.utils.instantiate(t) for t in cfg.transforms.values()]
    return CropTables(d.table, CropTableConfig(
        train_years=tuple(d.train_years), countries=list(d.countries) if d.countries else None, target=d.target, horizon=d.horizon,
        episodes=d.episodes, min_points=d.min_points, max_points=d.max_points, max_rows=d.max_rows,
        min_context_years=d.min_context_years, max_context_years=d.max_context_years), sampler=sampler, transforms=transforms)


def train(cfg: DictConfig) -> None:
    rec = cfg.recipe
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    warmup = max(1, round(rec.warmup_fraction * rec.steps)) if rec.warmup_fraction is not None else rec.warmup
    torch.manual_seed(cfg.seed)
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(resolved, f, indent=2)

    model, model_config = checkpoint.build_model(cfg.model.init, cfg.model.size, cfg.hf_cache_dir)
    if model_config.get("max_classes", 0) != 0:
        raise ValueError("This loop trains regression; use a regression checkpoint (max_classes=0).")
    model.to(device).train()
    if rec.checkpoint_activations:
        checkpoint_activations(model)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, init={cfg.model.init}, "
          f"micro-batch {rec.micro_batch_size} of batch {rec.batch_size}, activation checkpointing {rec.checkpoint_activations}")
    cuda = device.startswith("cuda")

    crop = build_crop_tables(cfg)
    if crop is not None:
        sizes = {c: len(p) for c, p in crop.country_points.items()}
        print(f"crop table: {len(crop.df)} rows, {crop.features.shape[1]} features, points per region {sizes}"
              + (f", dropped (< {cfg.data.min_points} points): {crop.dropped_regions}" if crop.dropped_regions else ""))
    n_steps = min(rec.stop_at or rec.steps, rec.steps)
    streams = build_streams(cfg, crop, n_steps)
    rng = np.random.default_rng(cfg.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=rec.lr, weight_decay=rec.weight_decay)
    best_rmse = float("inf")
    running, t0 = [], time.time()
    noise_file = noise_log = None
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["step", "source", "loss", "lr", "val_rmse", "val_r2", "val_baseline_rmse", "peak_mem_gb",
                      "table_rows", "table_features", "seconds"])
        for step in range(n_steps):
            source = choose_source(rng, cfg.source.prior_fraction)
            batch = next(streams[source])
            if "noise_params" in batch:  # this step's tables (one row per table), from an observational-noise transform
                if noise_log is None:
                    noise_file = open(os.path.join(out_dir, "noise_params.csv"), "w", newline="")
                    noise_log = csv.writer(noise_file)
                    noise_log.writerow(["step", "table", "sigma_fpar", "bg_fpar", "sigma_ssm", "sigma_rsm"])
                noise_log.writerows([[step + 1, t, *map(float, row)] for t, row in enumerate(batch["noise_params"].tolist())])
            x, y, n = batch["x"].to(device), batch["y"].to(device), int(batch["n_train"])
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, rec.lr, warmup, rec.steps)
            running.append(train_step(model, optimizer, x, y, n, device, rec.micro_batch_size, rec.grad_clip))

            val = {}
            if crop is not None and ((step + 1) % rec.eval_every == 0 or step + 1 == n_steps):
                val, _ = evaluate_walk_forward(model, crop, list(cfg.data.val_years), device)
                if val and val["rmse"] < best_rmse:
                    best_rmse = val["rmse"]
                    checkpoint.save(os.path.join(out_dir, "best.pt"), model, model_config, step=step + 1, val=val, args=resolved)
                print(f"step {step + 1}: val rmse {val.get('rmse', float('nan')):.3f} r2 {val.get('r2', float('nan')):.3f} "
                      f"(baseline rmse {val.get('baseline_rmse', float('nan')):.3f})")
            if step + 1 in rec.save_steps:
                checkpoint.save(os.path.join(out_dir, f"step{step + 1}.pt"), model, model_config, step=step + 1, val=val, args=resolved)
            if (step + 1) % rec.log_every == 0 or val:
                # peak GPU memory since the last log line (training steps and any evaluation in between)
                peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
                print(f"step {step + 1}/{n_steps} | {source} | loss {np.mean(running):.4f} | lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"| table {x.shape[1]} rows x {x.shape[2]} features | peak GPU mem {peak:.1f} GB | {time.time() - t0:.0f}s")
                log.writerow([step + 1, source, np.mean(running), optimizer.param_groups[0]["lr"], val.get("rmse"),
                              val.get("r2"), val.get("baseline_rmse"), round(peak, 2), x.shape[1], x.shape[2],
                              round(time.time() - t0, 1)])
                log_file.flush()
                if noise_file:
                    noise_file.flush()
                running = []
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
    if noise_file:
        noise_file.close()
    checkpoint.save(os.path.join(out_dir, "last.pt"), model, model_config, step=n_steps, args=resolved)
    print(f"done; checkpoints in {out_dir}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
