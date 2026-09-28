import itertools
import math
import numpy as np


def cv(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def _safe_skew(values):
    values = np.asarray(values, dtype=float)
    std = float(np.std(values))
    if len(values) < 3 or std <= 1e-12:
        return 0.0
    mean = float(np.mean(values))
    return float(np.mean(((values - mean) / std) ** 3))


def _gini(values):
    a = np.sort(np.asarray(values, dtype=float))
    n = a.size
    s = float(a.sum())
    if n == 0 or s <= 1e-12:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * a) / (n * s)) - (n + 1.0) / n)


def _safe_corr(a, b):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size < 2 or aa.size != bb.size:
        return 0.0
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(aa, bb)[0, 1])
    if math.isnan(value):
        return 0.0
    return value


def _qkp_input_features(weights, profits, pair, capacity):
    n = len(weights)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(profits, dtype=float)
    c = float(capacity)
    total_weight = float(w.sum()) if n else 0.0
    max_weight = float(w.max()) if n else 0.0
    mean_weight = float(w.mean()) if n else 0.0
    weight_std = float(w.std()) if n else 0.0
    pair_count = n * (n - 1) // 2
    triplet_count = math.comb(n, 3) if n >= 3 else 0

    nonzero_pairs = []
    pair_weights = []
    high_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            value = float(pair.get((i, j), 0.0))
            if value > 0.0:
                nonzero_pairs.append((i, j, value))
                pair_weights.append(float(w[i] + w[j]))

    q_vals = [value for _, _, value in nonzero_pairs]
    q_mean = float(np.mean(q_vals)) if q_vals else 0.0
    q_std = float(np.std(q_vals)) if q_vals else 0.0
    q_sum = float(sum(q_vals))
    if q_vals:
        threshold = q_mean + q_std
        for i, j, value in nonzero_pairs:
            if value >= threshold:
                high_edges.add((i, j))

    high_triangle_count = 0
    triplet_over_capacity_count = 0
    if triplet_count:
        for i, j, k in itertools.combinations(range(n), 3):
            if (i, j) in high_edges and (i, k) in high_edges and (j, k) in high_edges:
                high_triangle_count += 1
            if float(w[i] + w[j] + w[k]) > c:
                triplet_over_capacity_count += 1

    pair_over_capacity_count = 0
    if pair_count:
        for i in range(n):
            for j in range(i + 1, n):
                if float(w[i] + w[j]) > c:
                    pair_over_capacity_count += 1

    total_profit = float(p.sum()) + q_sum
    return {
        "n": n,
        "capacity_ratio": c / max(total_weight, 1.0),
        "capacity_max_weight_ratio": c / max(max_weight, 1.0),
        "mean_weight_capacity_ratio": mean_weight / max(c, 1.0),
        "max_weight_capacity_ratio": max_weight / max(c, 1.0),
        "weight_std": weight_std,
        "weight_cv": cv(w) if n else 0.0,
        "weight_gini": _gini(w),
        "large_weight_ratio": float(np.mean(w > 0.5 * c)) if n else 0.0,
        "medium_large_weight_ratio": float(np.mean((w > 0.25 * c) & (w <= 0.5 * c))) if n else 0.0,
        "small_weight_ratio": float(np.mean(w <= 0.10 * c)) if n else 0.0,
        "linear_profit_mean": float(p.mean()) if n else 0.0,
        "linear_profit_cv": cv(p) if n else 0.0,
        "profit_weight_corr": _safe_corr(p, w),
        "pair_density": len(nonzero_pairs) / max(pair_count, 1),
        "pair_profit_mean": q_mean,
        "pair_profit_cv": cv(q_vals) if q_vals else 0.0,
        "pair_profit_skew": _safe_skew(q_vals) if q_vals else 0.0,
        "pair_profit_gini": _gini(q_vals),
        "linear_profit_share": float(p.sum()) / max(total_profit, 1.0),
        "quadratic_profit_share": q_sum / max(total_profit, 1.0),
        "pair_profit_weight_corr": _safe_corr(q_vals, pair_weights),
        "high_pair_profit_ratio": len(high_edges) / max(len(nonzero_pairs), 1),
        "zero_pair_profit_ratio": 1.0 - (len(nonzero_pairs) / max(pair_count, 1)),
        "high_pair_triangle_ratio": high_triangle_count / max(triplet_count, 1),
        "pair_over_capacity_ratio": pair_over_capacity_count / max(pair_count, 1),
        "triplet_over_capacity_ratio": triplet_over_capacity_count / max(triplet_count, 1),
    }
