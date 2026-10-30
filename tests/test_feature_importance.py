import numpy as np
import pytest

from analysis.feature_importance.groups import build_groups, column_group, permute_group
from nanotabicl.cybench_features import SOURCES, STATIC_FEATURES


def _names(blocks=9):
    names = []
    for _, _, stats in SOURCES.values():
        for var, ss in stats.items():
            for s in ss:
                names += [f"{var}_{s}_{b}" for b in range(blocks)]
    return names + list(STATIC_FEATURES)


def test_groups_partition_columns_single_view():
    names = _names()
    groups = build_groups(names)
    single = {g: c for (g, v), c in groups.items() if v == "single"}
    assert len(single) == 16 and "static" in single and len(single["prec"]) == 18
    all_cols = np.sort(np.concatenate(list(single.values())))
    assert (all_cols == np.arange(len(names))).all()
    fam = groups[("fam_precip", "family")]
    assert set(fam) == set(single["prec"]) | set(single["cum_prec"])
    assert column_group("cum_prec_max_3") == "cum_prec" and column_group("prec_sum_0") == "prec"


@pytest.mark.parametrize("group", ["prec", "static", "fpar"])
def test_permutation_changes_only_group_and_same_shuffle_for_context_and_query(group):
    names = _names()
    cols = build_groups(names)[(group, "single")]
    rng = np.random.default_rng(0)
    n_ctx, n_q = 30, 8
    x = rng.normal(size=(n_ctx + n_q, len(names))).astype(np.float32)
    out = permute_group(x, cols, np.random.default_rng(1))
    other = np.setdiff1d(np.arange(len(names)), cols)
    assert (out[:, other] == x[:, other]).all()
    # one permutation of rows for all group columns: the group's rows are a row permutation of the original block
    perm = np.array([np.where((x[:, cols] == row).all(axis=1))[0][0] for row in out[:, cols]])
    assert sorted(perm) == list(range(n_ctx + n_q))
    assert (out[:, cols] == x[perm][:, cols]).all()
    assert not (perm == np.arange(len(perm))).all()
    # context and query rows are shuffled together (rows move between the two parts)
    assert (perm[:n_ctx] >= n_ctx).any()
    # deterministic given the seed
    assert (permute_group(x, cols, np.random.default_rng(1)) == out).all()
