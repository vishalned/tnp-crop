"""Batches of synthetic regression tables from the TabICLv2 nanoprior (`prior.py`).

Every table of a batch shares its number of rows, features and context rows
(so the batch is one tensor); each table is an independent prior draw. The
prior filters out unlearnable datasets with an ExtraTrees check, which makes
generation CPU-bound: run it in DataLoader workers (`num_workers`).

`PriorTables`: random shapes (TabICL-style). `MatchedPriorTables`: the control
for crop training (`train.py --data prior-matched`), prior tables with exactly
the crop run's table shapes and context/query split, step by step.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from nanotabicl import prior


def sample_table(n_rows: int, n_features: int) -> tuple:
    """(x [n_rows, n_features], y [n_rows]) of one regression dataset from the prior."""
    columns = prior.rand_dataset_filtered(prior.rand_cat_sizes(n_features), [0], n_rows)
    x = torch.cat([columns[f"x_{i}"].float() for i in range(n_features)], dim=-1)
    return x, columns["y_0"].float().squeeze(-1)


class PriorTables(IterableDataset):
    """Endless stream of batches {"x": [B, n_rows, n_features], "y": [B, n_rows], "n_train"}."""

    def __init__(self, batch_size: int, min_rows: int = 64, max_rows: int = 1024, max_features: int = 100,
                 min_train_fraction: float = 0.5, max_train_fraction: float = 0.9, seed: int = 0):
        super().__init__()
        self.batch_size, self.min_rows, self.max_rows, self.max_features = batch_size, min_rows, max_rows, max_features
        self.train_fraction = (min_train_fraction, max_train_fraction)
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        # prior.py samples from numpy's and torch's global RNGs
        np.random.seed((self.seed * 1000 + worker_id) % 2**32)
        torch.manual_seed(self.seed * 1000 + worker_id)
        while True:
            n_rows = int(np.exp(np.random.uniform(np.log(self.min_rows), np.log(self.max_rows))))
            n_features = int(np.random.randint(1, self.max_features + 1))
            n_train = int(n_rows * np.random.uniform(*self.train_fraction))
            tables = [sample_table(n_rows, n_features) for _ in range(self.batch_size)]
            yield {"x": torch.stack([t[0] for t in tables]), "y": torch.stack([t[1] for t in tables]), "n_train": n_train}


class MatchedPriorTables(IterableDataset):
    """Control for training on crop tables: prior tables with the crop batches' shapes.

    `shapes[k]` = (n_rows, n_features, n_train) of the crop run's k-th batch (`crop_batch_shapes`),
    so step k gets prior tables with the same number of rows, features and context rows (first
    `n_train` rows = context, as in the crop batches). Each column is standardised over its table's
    rows, the counterpart of the crop features' normaliser (fitted on all train rows). With DataLoader
    workers, worker w generates batches w, w + n_workers, ...; the DataLoader returns them in order.
    """

    def __init__(self, shapes, batch_size: int, seed: int = 0):
        super().__init__()
        self.shapes, self.batch_size, self.seed = list(shapes), batch_size, seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        np.random.seed((self.seed * 1000 + worker_id) % 2**32)
        torch.manual_seed(self.seed * 1000 + worker_id)
        for n_rows, n_features, n_train in self.shapes[worker_id::n_workers]:
            tables = [sample_table(n_rows, n_features) for _ in range(self.batch_size)]
            x = torch.stack([t[0] for t in tables])
            x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp(min=1e-6)
            yield {"x": x, "y": torch.stack([t[1] for t in tables]), "n_train": n_train}


def crop_batch_shapes(crop, batch_size: int, n_batches: int, seed: int) -> list:
    """(n_rows, n_features, n_train) of the first `n_batches` batches the crop stream draws with `seed`
    (`train.CropStream`: the same RNG and calls), i.e. the crop run's table shapes step by step."""
    rng = np.random.default_rng(seed)
    shapes = []
    for _ in range(n_batches):
        batch = crop.sample_batch(rng, batch_size)
        shapes.append((batch["x"].shape[1], batch["x"].shape[2], int(batch["n_train"])))
    return shapes


if __name__ == "__main__":
    # Sample regression tables from the prior and save/plot them:
    #   python -m nanotabicl.prior_tables --num-tables 8 --rows 200 --features 5 --out prior_samples.npz --plot prior.png
    import argparse

    p = argparse.ArgumentParser(description="Sample regression tables from the TabICLv2 nanoprior.")
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--features", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="save the tables as .npz (x: [n, rows, features], y: [n, rows])")
    p.add_argument("--plot", default=None, help="save a PNG of each table's first feature vs y")
    a = p.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    xs, ys = [], []
    for i in range(a.num_tables):
        x, y = sample_table(a.rows, a.features)
        xs.append(x.numpy()); ys.append(y.numpy())
        print(f"table {i}: x {tuple(x.shape)}, y mean {y.mean():.3f} std {y.std():.3f}, "
              f"categorical-looking columns {sum(len(np.unique(c)) <= 100 and np.allclose(c, np.round(c)) for c in x.T.numpy())}")
    if a.out:
        np.savez(a.out, x=np.stack(xs), y=np.stack(ys))
        print(f"saved {a.out}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = min(4, a.num_tables)
        fig, axes = plt.subplots((a.num_tables + cols - 1) // cols, cols, figsize=(3 * cols, 2.6 * ((a.num_tables + cols - 1) // cols)), squeeze=False)
        for ax, x, y in zip(axes.flat, xs, ys):
            ax.scatter(x[:, 0], y, s=4)
            ax.set(xticks=[], yticks=[], xlabel="x_0", ylabel="y")
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"saved {a.plot}")
