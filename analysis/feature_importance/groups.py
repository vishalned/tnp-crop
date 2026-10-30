"""Feature groups for the grouped permutation importance.

Columns are CY-Bench's tabular features `<variable>_<stat>_<block>` plus the static features
(`nanotabicl/cybench_features.py`). Two views:

* ``single``: one group per dynamic variable (all its statistics and time blocks) and one ``static``
  group. `prec` holds `prec_sum` and `prec_max`. `cum_*` and `gdd` are their own variables.
* ``family``: variables derived from each other, permuted together (overlaps with ``single``).
"""

import numpy as np

from nanotabicl.cybench_features import SOURCES, STATIC_FEATURES, variable_of

# family group -> single groups it merges
FAMILIES = {
    "fam_precip": ["prec", "cum_prec"],
    "fam_temp_gdd": ["tavg", "gdd", "cum_gdd"],
    "fam_fpar": ["fpar", "cum_fpar"],
    "fam_water_balance": ["cwb", "cum_cwb"],
}

_STAT_TO_VARIABLE = {f"{var}_{stat}": var for _, _, stats in SOURCES.values() for var, ss in stats.items() for stat in ss}


def column_group(name: str) -> str:
    """Single-view group of a column name."""
    if name in STATIC_FEATURES:
        return "static"
    key = variable_of(name)
    if key not in _STAT_TO_VARIABLE:
        raise KeyError(f"Column {name!r} belongs to no group.")
    return _STAT_TO_VARIABLE[key]


def build_groups(names: list) -> dict:
    """{(group, view): column indices} for the column names; the single view partitions the columns."""
    single = {}
    for j, n in enumerate(names):
        single.setdefault(column_group(n), []).append(j)
    groups = {(g, "single"): np.array(c) for g, c in single.items()}
    for fam, members in FAMILIES.items():
        cols = [j for m in members for j in single.get(m, [])]
        if cols:
            groups[(fam, "family")] = np.array(sorted(cols))
    return groups


def permute_group(x: np.ndarray, columns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy of the rows x [context; query] with ONE row permutation applied to all `columns`.
    Other columns are untouched; call it on the stacked context+query rows so both get the same shuffle."""
    out = x.copy()
    out[:, columns] = x[rng.permutation(len(x))][:, columns]
    return out
