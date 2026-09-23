"""Smoke test for the crop TNP: build the datamodule from the training
configs, log the first sampled episodes, pull one batch, run one forward +
backward pass, and check the loss is finite.

    uv run python scripts/smoke_test_tnp_crop.py --store-dir data/processed/tnp_store_wheat --device cuda
    uv run python scripts/smoke_test_tnp_crop.py --synthetic   # no real data needed

Not a training run: it only checks that shapes, padding and the loss work,
and prints the token counts to keep an eye on the attention budget.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.synthetic_crop_store import make_synthetic_store  # noqa: E402
from src.data.crop_datamodule import CropEpisodeDataModule  # noqa: E402
from src.models.components.tnpd import TNPD, Batch  # noqa: E402

TOKEN_WARNING = 15_000


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the crop TNP datamodule + model.")
    parser.add_argument("--store-dir", type=str, default=None, help="Training store (default: the one in configs/data/crop.yaml).")
    parser.add_argument("--synthetic", action="store_true", help="Use a small synthetic store instead of real data.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-log-episodes", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=None, help="Override max_points (e.g. to fit a CPU).")
    parser.add_argument("--max-context-years", type=int, default=None, help="Override max_context_years.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_cfg = OmegaConf.load(os.path.join(root, "configs", "data", "crop.yaml"))
    net_cfg = OmegaConf.load(os.path.join(root, "configs", "model", "tnp_crop.yaml")).net
    tmp = tempfile.mkdtemp()
    if args.synthetic:
        data_cfg.store_dir = make_synthetic_store(os.path.join(tmp, "store"))
    elif args.store_dir:
        data_cfg.store_dir = args.store_dir
    else:
        data_cfg.store_dir = os.path.join(root, "data", "processed", "tnp_store_wheat")
    data_cfg.norm_stats_out = os.path.join(tmp, "norm_stats.json")
    data_cfg.num_workers = 0
    data_cfg.batch_size = args.batch_size
    if args.max_points is not None:
        data_cfg.max_points = args.max_points
    if args.max_context_years is not None:
        data_cfg.max_context_years = args.max_context_years
    kwargs = {k: v for k, v in OmegaConf.to_container(data_cfg, resolve=True).items() if k != "_target_"}

    print(f"store: {kwargs['store_dir']}")
    dm = CropEpisodeDataModule(**kwargs)
    t0 = time.time()
    dm.setup("fit")
    store = dm.store
    print(
        f"setup {time.time() - t0:.1f}s: {len(store.points)} points "
        f"{store.points['country'].value_counts().to_dict()}, years {store.years.min()}-{store.years.max()}, "
        f"countries sampled: {store.countries}; norm stats -> {kwargs['norm_stats_out']}"
    )

    # --- first episodes: sampled X, C, country and token counts -----------
    print(f"\nfirst {args.num_log_episodes} training episodes:")
    print(f"{'#':>3} {'country':<15} {'T':>5} {'X':>3} {'C':>3} {'profile':<9} {'context':>8} {'target':>7}")
    it = iter(dm.data_train)
    totals = []
    for i in range(args.num_log_episodes):
        info = next(it)["info"]
        total = info["num_context_tokens"] + info["num_target_tokens"]
        totals.append(total)
        flag = "  <-- over budget" if total > TOKEN_WARNING else ""
        print(
            f"{i:>3} {info['country']:<15} {info['target_year']:>5} {info['num_points']:>3} {info['context_years']:>3} "
            f"{info['profile']:<9} {info['num_context_tokens']:>8} {info['num_target_tokens']:>7}{flag}"
        )
    totals = np.array(totals)
    print(f"tokens per episode: mean {totals.mean():.0f}, max {totals.max()}, "
          f"{(totals > TOKEN_WARNING).sum()}/{len(totals)} over {TOKEN_WARNING}")

    # --- one batch, one forward + backward --------------------------------
    batch = next(iter(dm.train_dataloader()))
    print("\nbatch shapes: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items() if torch.is_tensor(v)))
    net_kwargs = {k: v for k, v in OmegaConf.to_container(net_cfg).items() if k != "_target_"}
    net_kwargs["num_modalities"] = kwargs["num_modalities"]
    net = TNPD(**net_kwargs).to(args.device)
    if not args.device.startswith("cuda"):
        # With attention dropout active, CPU attention falls back to a kernel
        # that materializes the full [N, N] attention matrix (GBs at 10k+
        # tokens). CUDA's memory-efficient kernel handles dropout + padding
        # masks without that, so only disable dropout off-GPU.
        net.eval()
        print("(CPU: dropout disabled for this check to avoid the N x N attention fallback)")
    typed = Batch.from_dict({k: v.to(args.device) for k, v in batch.items() if torch.is_tensor(v)})
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = net(typed)
    out["loss"].backward()
    elapsed = time.time() - t0
    loss = out["loss"].item()
    grads_ok = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    mem = f", peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if args.device.startswith("cuda") else ""
    print(f"forward+backward on {args.device}: loss {loss:.4f}, {elapsed:.1f}s{mem}")

    ok = np.isfinite(loss) and grads_ok and torch.isfinite(out["pred_tar"].mean).all()
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED (non-finite loss, predictions or gradients)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
