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

`n_estimators=None` (default): that single forward pass on the table as it is.
`n_estimators=k`: TabICL's test-time ensemble (`ensemble.py`, the same as
`TabICLRegressor(n_estimators=k)`): k views of the table (column orders x
normalisations, outliers clipped), one forward pass each, predictions averaged.

The full context is always used. Query rows only attend to the context rows
inside the model, so if a forward pass runs out of GPU memory the queries
are split into batches (each with the full context) without changing the
predictions.
"""

from typing import List, Optional

import numpy as np
import torch

from nanotabicl import checkpoint as ckpt
from nanotabicl.ensemble import Ensemble
from nanotabicl.train import forward_quantiles


class NanoTabICLRegressor:
    def __init__(self, checkpoint: str = "official", device: Optional[str] = None,
                 query_batch_size: Optional[int] = None, cache_dir: Optional[str] = None,
                 n_estimators: Optional[int] = None, norm_methods: Optional[List[str]] = None,
                 feat_shuffle_method: str = "latin", outlier_threshold: float = 4.0,
                 random_state: Optional[int] = 42):
        """`checkpoint`: "official" (converted TabICLv2 weights), an official
        .ckpt or one of our .pt checkpoints. `query_batch_size`: start with
        query batches of this size (default: all queries at once).
        `n_estimators` and the rest: TabICL's test-time ensemble (see `ensemble.Ensemble`);
        None = no ensemble, one pass on the unchanged table."""
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_batch_size = query_batch_size
        self.cache_dir = cache_dir
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
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
        self.ensemble_ = None
        if self.n_estimators is not None:
            self.ensemble_ = Ensemble(self.n_estimators, self.norm_methods, self.feat_shuffle_method,
                                      self.outlier_threshold, self.random_state).fit(self.X_)
        if self.model is None:
            self.model, _ = ckpt.build_model(self.checkpoint, cache_dir=self.cache_dir)
            self.model.to(self.device).eval()
        return self

    @torch.no_grad()
    def _predict_batch(self, x_context: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        n_train = len(x_context)
        x = torch.from_numpy(np.concatenate([x_context, x_query]).astype(np.float32))[None].to(self.device)
        y = torch.from_numpy(np.concatenate([self.y_, np.zeros(len(x_query), np.float32)]))[None].to(self.device)
        quantiles, _, mean, std = forward_quantiles(self.model, x, y, n_train, self.device)
        return (quantiles.mean(-1) * std + mean)[0].float().cpu().numpy()

    def _predict_view(self, x_context: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        batch = self.query_batch_size or len(x_query)
        while True:
            try:
                return np.concatenate([self._predict_batch(x_context, x_query[i : i + batch])
                                       for i in range(0, len(x_query), batch)])
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"NanoTabICLRegressor: out of GPU memory, retrying with query batches of {batch}")

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NanoTabICLRegressor.predict called before fit.")
        x = self._as_array(X)
        if x.shape[1] != self.X_.shape[1]:
            raise ValueError(f"X has {x.shape[1]} features, the context {self.X_.shape[1]}.")
        if self.ensemble_ is None:
            return self._predict_view(self.X_, x)
        # y is in its own units in every view, so the members' predictions average directly
        return np.mean([self._predict_view(c, q) for c, q in self.ensemble_.transform(x)], axis=0)
