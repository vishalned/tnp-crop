"""scikit-learn style NanoTabICL regressor, for the CY-Bench benchmark
(AgML-CY-Bench-Modeling's tabular foundation model wrapper):

    reg = NanoTabICLRegressor(checkpoint="logs/nanotabicl/run/best.pt", device="cuda")
    reg.fit(X_train, y_train)      # stores the context
    y_pred = reg.predict(X_test)

`X` is the benchmark's feature table (`dataset.x`: CY-Bench features,
already normalised by the benchmark). NaN is set to 0, as in training
(`CropTables`). Prediction is the same as `train.evaluate_walk_forward`: y
standardised with the context's mean/std, one forward pass (bf16 autocast on
CUDA), mean of the 999 quantiles, back in y's units.

The full context is always used. Query rows only attend to the context rows
inside the model, so if a forward pass runs out of GPU memory the queries
are split into batches (each with the full context) without changing the
predictions.
"""

from typing import Optional

import numpy as np
import torch

from nanotabicl import checkpoint as ckpt
from nanotabicl.train import forward_quantiles


class NanoTabICLRegressor:
    def __init__(self, checkpoint: str = "official", device: Optional[str] = None,
                 query_batch_size: Optional[int] = None, cache_dir: Optional[str] = None):
        """`checkpoint`: "official" (converted TabICLv2 weights), an official
        .ckpt or one of our .pt checkpoints. `query_batch_size`: start with
        query batches of this size (default: all queries at once)."""
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_batch_size = query_batch_size
        self.cache_dir = cache_dir
        self.model = None

    @staticmethod
    def _as_array(X) -> np.ndarray:
        x = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(x, nan=0.0)  # as CropTables

    def fit(self, X, y) -> "NanoTabICLRegressor":
        self.X_ = self._as_array(X)
        self.y_ = np.asarray(y, dtype=np.float32).ravel()
        if len(self.X_) != len(self.y_):
            raise ValueError(f"X has {len(self.X_)} rows, y has {len(self.y_)}.")
        if self.model is None:
            self.model, _ = ckpt.build_model(self.checkpoint, cache_dir=self.cache_dir)
            self.model.to(self.device).eval()
        return self

    @torch.no_grad()
    def _predict_batch(self, x_query: np.ndarray) -> np.ndarray:
        n_train = len(self.X_)
        x = torch.from_numpy(np.concatenate([self.X_, x_query]))[None].to(self.device)
        y = torch.from_numpy(np.concatenate([self.y_, np.zeros(len(x_query), np.float32)]))[None].to(self.device)
        quantiles, _, mean, std = forward_quantiles(self.model, x, y, n_train, self.device)
        return (quantiles.mean(-1) * std + mean)[0].float().cpu().numpy()

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NanoTabICLRegressor.predict called before fit.")
        x = self._as_array(X)
        if x.shape[1] != self.X_.shape[1]:
            raise ValueError(f"X has {x.shape[1]} features, the context {self.X_.shape[1]}.")
        batch = self.query_batch_size or len(x)
        while True:
            try:
                return np.concatenate([self._predict_batch(x[i : i + batch]) for i in range(0, len(x), batch)])
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"NanoTabICLRegressor: out of GPU memory, retrying with query batches of {batch}")
