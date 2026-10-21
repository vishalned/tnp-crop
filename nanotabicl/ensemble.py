"""TabICL's test-time ensemble, for NanoTabICLRegressor.

Ported from the official `tabicl` package (v2.1.1, `tabicl/_sklearn/preprocessing.py`:
`EnsembleGenerator`, `PreprocessingPipeline`, `CustomStandardScaler`, `OutlierRemover`,
`UniqueFeatureFilter`, `Shuffler`; BSD-3, (c) the TabICL authors), so the ensemble is the same as
`TabICLRegressor(n_estimators=...)`. `tests/test_nanotabicl.py` checks the views against `tabicl`'s
own `EnsembleGenerator` when it is installed.

Each ensemble member ("estimator") is one view of the same table, fitted on the context rows only:

1. constant context columns are dropped;
2. standardised (clipped to +-100);
3. `norm_method`: "none", or "power" (Yeo-Johnson, sklearn `PowerTransformer`, standardised);
4. outliers clipped softly beyond `outlier_threshold` standard deviations;
5. columns permuted (`feat_shuffle_method`: rows of a random Latin square by default, so every
   column takes a different position in every view).

With the default `norm_methods = ["none", "power"]` and 8 members, that is 4 column orders x 2
normalisations. The model predicts every view; the predictions (in y's units) are averaged.
"""

import itertools
import random
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.preprocessing import PowerTransformer


class _StandardScaler:
    def __init__(self, clip: float = 100.0, epsilon: float = 1e-6):
        self.clip, self.epsilon = clip, epsilon

    def fit(self, x):
        self.mean_, self.scale_ = x.mean(axis=0), x.std(axis=0) + self.epsilon
        return self

    def transform(self, x):
        return np.clip((x - self.mean_) / self.scale_, -self.clip, self.clip)


class _OutlierRemover:
    """Two-stage z-score bounds, then soft (log) clipping beyond them."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold

    def fit(self, x):
        ddof = 1 if x.shape[0] > 1 else 0
        mean, std = np.nanmean(x, axis=0), np.maximum(np.nanstd(x, axis=0, ddof=ddof), 1e-6)
        clean = x.copy()
        clean[(x < mean - self.threshold * std) | (x > mean + self.threshold * std)] = np.nan
        mean, std = np.nanmean(clean, axis=0), np.maximum(np.nanstd(clean, axis=0, ddof=ddof), 1e-6)
        self.lower_, self.upper_ = mean - self.threshold * std, mean + self.threshold * std
        return self

    def transform(self, x):
        x = np.maximum(-np.log1p(np.abs(x)) + self.lower_, x)
        return np.minimum(np.log1p(np.abs(x)) + self.upper_, x)


class _Preprocessing:
    def __init__(self, norm_method: str, outlier_threshold: float):
        if norm_method not in ("none", "power"):
            raise ValueError(f"norm_method must be 'none' or 'power', got {norm_method!r}.")
        self.norm_method, self.outlier_threshold = norm_method, outlier_threshold

    def fit(self, x):
        self.scaler_ = _StandardScaler().fit(x)
        x = self.scaler_.transform(x)
        self.normalizer_ = None
        if self.norm_method == "power":
            self.x_min_, self.x_max_ = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
            self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            x = self.normalizer_.fit_transform(x)
        self.outliers_ = _OutlierRemover(self.outlier_threshold).fit(x)
        self.x_fit_ = self.outliers_.transform(x)
        return self

    def transform(self, x):
        x = self.scaler_.transform(x)
        if self.normalizer_ is not None:
            try:
                x = self.normalizer_.transform(x)
            except ValueError:  # as tabicl: values out of the fitted range
                x = self.normalizer_.transform(np.clip(x, self.x_min_, self.x_max_))
        return self.outliers_.transform(x)


def feature_shuffles(n_features: int, n_estimators: int, method: str = "latin",
                     random_state: Optional[int] = None, max_for_latin: int = 4000) -> List[List[int]]:
    """Column orders, as tabicl's `Shuffler.shuffle`."""
    rng = random.Random(random_state)
    indices = list(range(n_features))
    if method == "latin" and n_features > max_for_latin:
        method = "random"
    if method == "none" or n_estimators == 1:
        return [indices]
    if method == "shift":
        return [indices[-i:] + indices[:-i] for i in range(n_features)]
    if method == "random":
        if n_features <= 5:
            perms = [list(p) for p in itertools.permutations(indices)]
            return rng.sample(perms, min(n_estimators, len(perms)))
        return [rng.sample(indices, n_features) for _ in range(n_estimators)]
    if method != "latin":
        raise ValueError(f"Unknown feat_shuffle_method {method!r}: 'latin', 'random', 'shift' or 'none'.")

    def rls(symbols):  # random Latin square, recursively
        if len(symbols) == 1:
            return [symbols]
        sym = rng.choice(symbols)
        symbols.remove(sym)
        square = rls(symbols)
        square.append(square[0].copy())
        for i in range(len(symbols) + 1):
            square[i].insert(i, sym)
        return square

    import sys
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 100000))
    try:
        square = rls(list(range(n_features)))
    finally:
        sys.setrecursionlimit(limit)
    square = deepcopy(square)
    rng.shuffle(square)
    columns = list(zip(*square))
    rng.shuffle(columns)
    return [list(c) for c in columns]


class Ensemble:
    """The views of a context/query table, as tabicl's `EnsembleGenerator` (regression)."""

    def __init__(self, n_estimators: int = 8, norm_methods: Optional[List[str]] = None,
                 feat_shuffle_method: str = "latin", outlier_threshold: float = 4.0,
                 random_state: Optional[int] = 42):
        self.n_estimators = n_estimators
        self.norm_methods = [norm_methods] if isinstance(norm_methods, str) else (norm_methods or ["none", "power"])
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, x: np.ndarray) -> "Ensemble":
        x = np.asarray(x, dtype=np.float64)
        self.keep_ = (np.array([len(np.unique(x[:, j])) > 1 for j in range(x.shape[1])])
                      if x.shape[0] > 1 else np.ones(x.shape[1], bool))
        x = x[:, self.keep_]
        rng = random.Random(self.random_state)
        shuffles = feature_shuffles(x.shape[1], self.n_estimators, self.feat_shuffle_method, self.random_state)
        configs = [(s, None) for s in shuffles]
        rng.shuffle(configs)
        configs = list(itertools.product(configs, self.norm_methods))[: self.n_estimators]
        self.views_ = OrderedDict()  # norm method -> column orders, in tabicl's order
        for method in list(set(m for _, m in configs)):
            self.views_[method] = [c[0] for c, m in configs if m == method]
        self.preprocessing_ = {m: _Preprocessing(m, self.outlier_threshold).fit(x) for m in self.views_}
        return self

    def transform(self, x_query: np.ndarray) -> List[tuple]:
        """[(context view, query view), ...]: one pair per ensemble member."""
        x_query = np.asarray(x_query, dtype=np.float64)[:, self.keep_]
        out = []
        for method, orders in self.views_.items():
            pre = self.preprocessing_[method]
            context, query = pre.x_fit_, pre.transform(x_query)
            out += [(context[:, order], query[:, order]) for order in orders]
        return out
