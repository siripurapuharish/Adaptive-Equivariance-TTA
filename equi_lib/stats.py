import numpy as np


def auroc(scores, labels) -> float:
    """Area under the ROC curve (higher score = more likely positive), via the Mann-Whitney U
    statistic with average ranks for ties. Exact; works for tens of millions of values.
    Returns nan if either class is empty."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank of each distinct value
    rank_sum_pos = avg_rank[inverse][y].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
