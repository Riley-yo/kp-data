from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURE_REGISTRY_VERSION = "tsp-variant-problem-features-v2"

ATSP_FEATURE_NAMES = (
    "n",
    "directed_cost_mean",
    "directed_cost_cv",
    "directed_cost_skew",
    "directed_cost_kurtosis",
    "directed_cost_range_ratio",
    "directed_long_edge_ratio",
    "directed_short_edge_ratio",
    "directed_tie_ratio",
    "out_nearest_mean_ratio",
    "out_nearest_cv",
    "in_nearest_mean_ratio",
    "in_nearest_cv",
    "out_first_second_ratio",
    "in_first_second_ratio",
    "outgoing_mean_cv",
    "incoming_mean_cv",
    "in_out_mean_gap_ratio",
    "outgoing_node_eccentricity",
    "incoming_node_eccentricity",
    "reciprocal_abs_gap_mean_ratio",
    "reciprocal_abs_gap_max_ratio",
    "reciprocal_abs_gap_cv",
    "reciprocal_cost_correlation",
    "reciprocal_equal_ratio",
)

STSP_FEATURE_NAMES = (
    "n",
    "edge_cost_mean",
    "edge_cost_cv",
    "edge_cost_skew",
    "edge_cost_kurtosis",
    "edge_cost_range_ratio",
    "edge_long_ratio",
    "edge_short_ratio",
    "edge_tie_ratio",
    "nearest_mean_ratio",
    "nearest_cv",
    "first_second_ratio",
    "node_mean_cv",
    "node_eccentricity",
    "triangle_violation_ratio",
    "triangle_violation_mean_ratio",
    "mst_weight_ratio",
    "mst_edge_cv",
    "low_cost_graph_density",
    "low_cost_component_ratio",
    "low_cost_degree_cv",
    "nearest_reciprocity",
)

_ATSP_FORMULAS = {
    "n": ("number of nodes", "size"),
    "directed_cost_mean": ("mean of c[i,j] over every ordered pair i != j", "cost"),
    "directed_cost_cv": ("standard deviation of all directed costs divided by their mean", "scale_free"),
    "directed_cost_skew": ("third standardized moment of all directed costs", "scale_free"),
    "directed_cost_kurtosis": ("fourth standardized moment minus 3 of all directed costs", "scale_free"),
    "directed_cost_range_ratio": ("directed cost range divided by directed cost mean", "scale_free"),
    "directed_long_edge_ratio": ("fraction of directed costs at least 1.5 times the directed mean", "scale_free"),
    "directed_short_edge_ratio": ("fraction of directed costs at most 0.5 times the directed mean", "scale_free"),
    "directed_tie_ratio": ("1 minus distinct directed cost count divided by directed arc count", "scale_free"),
    "out_nearest_mean_ratio": ("mean per-node minimum outgoing cost divided by global directed mean", "scale_free"),
    "out_nearest_cv": ("coefficient of variation of per-node minimum outgoing costs", "scale_free"),
    "in_nearest_mean_ratio": ("mean per-node minimum incoming cost divided by global directed mean", "scale_free"),
    "in_nearest_cv": ("coefficient of variation of per-node minimum incoming costs", "scale_free"),
    "out_first_second_ratio": ("mean ratio of first to second cheapest outgoing cost", "scale_free"),
    "in_first_second_ratio": ("mean ratio of first to second cheapest incoming cost", "scale_free"),
    "outgoing_mean_cv": ("coefficient of variation of node outgoing row means", "scale_free"),
    "incoming_mean_cv": ("coefficient of variation of node incoming column means", "scale_free"),
    "in_out_mean_gap_ratio": ("mean absolute difference between each node's outgoing and incoming means divided by global mean", "scale_free"),
    "outgoing_node_eccentricity": ("maximum node outgoing mean divided by mean node outgoing mean", "scale_free"),
    "incoming_node_eccentricity": ("maximum node incoming mean divided by mean node incoming mean", "scale_free"),
    "reciprocal_abs_gap_mean_ratio": ("mean absolute c[i,j]-c[j,i] over unordered pairs divided by global mean", "scale_free"),
    "reciprocal_abs_gap_max_ratio": ("maximum absolute c[i,j]-c[j,i] divided by global mean", "scale_free"),
    "reciprocal_abs_gap_cv": ("coefficient of variation of reciprocal absolute gaps", "scale_free"),
    "reciprocal_cost_correlation": ("Pearson correlation between all directed costs and their reverse costs", "scale_free"),
    "reciprocal_equal_ratio": ("fraction of unordered pairs with c[i,j] equal to c[j,i]", "scale_free"),
}

_STSP_FORMULAS = {
    "n": ("number of nodes", "size"),
    "edge_cost_mean": ("mean of c[i,j] over unordered pairs i < j", "cost"),
    "edge_cost_cv": ("standard deviation of unique undirected costs divided by their mean", "scale_free"),
    "edge_cost_skew": ("third standardized moment of unique undirected costs", "scale_free"),
    "edge_cost_kurtosis": ("fourth standardized moment minus 3 of unique undirected costs", "scale_free"),
    "edge_cost_range_ratio": ("undirected cost range divided by undirected cost mean", "scale_free"),
    "edge_long_ratio": ("fraction of undirected costs at least 1.5 times the edge mean", "scale_free"),
    "edge_short_ratio": ("fraction of undirected costs at most 0.5 times the edge mean", "scale_free"),
    "edge_tie_ratio": ("1 minus distinct undirected cost count divided by undirected edge count", "scale_free"),
    "nearest_mean_ratio": ("mean per-node nearest-neighbor cost divided by global edge mean", "scale_free"),
    "nearest_cv": ("coefficient of variation of per-node nearest-neighbor costs", "scale_free"),
    "first_second_ratio": ("mean ratio of first to second nearest-neighbor cost", "scale_free"),
    "node_mean_cv": ("coefficient of variation of per-node incident-edge means", "scale_free"),
    "node_eccentricity": ("maximum node incident-edge mean divided by the mean of node means", "scale_free"),
    "triangle_violation_ratio": ("fraction of ordered distinct triples with c[i,k] > c[i,j]+c[j,k]", "scale_free"),
    "triangle_violation_mean_ratio": ("mean positive triangle excess divided by global edge mean", "scale_free"),
    "mst_weight_ratio": ("minimum-spanning-tree weight divided by (n-1) times global edge mean", "scale_free"),
    "mst_edge_cv": ("coefficient of variation of minimum-spanning-tree edge costs", "scale_free"),
    "low_cost_graph_density": ("density of edges no larger than the 25th cost percentile", "scale_free"),
    "low_cost_component_ratio": ("connected-component count of the low-cost graph divided by n", "scale_free"),
    "low_cost_degree_cv": ("coefficient of variation of node degrees in the low-cost graph", "scale_free"),
    "nearest_reciprocity": ("fraction of tied nearest-neighbor relations that are mutual", "scale_free"),
}


def _cost_mapping(instance: dict[str, Any]) -> tuple[int, dict[tuple[int, int], float]]:
    n = int(instance["n"])
    raw = instance["cost"]
    if isinstance(raw, list):
        cost = {
            (i, j): float(raw[i][j])
            for i in range(n)
            for j in range(n)
            if i != j
        }
    else:
        cost = {(int(i), int(j)): float(value) for (i, j), value in raw.items()}
    expected = {(i, j) for i in range(n) for j in range(n) if i != j}
    if set(cost) != expected:
        raise ValueError("TSP cost matrix is incomplete")
    if not all(math.isfinite(value) and value >= 0.0 for value in cost.values()):
        raise ValueError("TSP costs must be finite and nonnegative")
    return n, cost


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _cv(values: np.ndarray) -> float:
    mean = _mean(values)
    return float(np.std(values) / abs(mean)) if abs(mean) > 1e-12 else 0.0


def _skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    std = float(np.std(values)) if len(values) else 0.0
    if std <= 1e-12:
        return 0.0, 0.0
    z = (values - float(np.mean(values))) / std
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def _range_ratio(values: np.ndarray) -> float:
    mean = abs(_mean(values))
    if not len(values) or mean <= 1e-12:
        return 0.0
    return float((np.max(values) - np.min(values)) / mean)


def _tail_ratios(values: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return 0.0, 0.0
    mean = float(np.mean(values))
    if abs(mean) <= 1e-12:
        return 0.0, 0.0
    return (
        float(np.mean(values >= 1.5 * mean)),
        float(np.mean(values <= 0.5 * mean)),
    )


def _tie_ratio(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    return float(1.0 - len(np.unique(values)) / len(values))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _nearest_statistics(rows: list[list[float]], global_mean: float) -> dict[str, float]:
    first = np.asarray([sorted(row)[0] for row in rows], dtype=float)
    second = np.asarray([sorted(row)[1] for row in rows], dtype=float)
    ratios = np.divide(first, second, out=np.ones_like(first), where=np.abs(second) > 1e-12)
    return {
        "mean_ratio": float(np.mean(first) / abs(global_mean)) if abs(global_mean) > 1e-12 else 0.0,
        "cv": _cv(first),
        "first_second_ratio": float(np.mean(ratios)),
    }


def extract_atsp_features(instance: dict[str, Any]) -> dict[str, float]:
    n, cost = _cost_mapping(instance)

    arcs = np.asarray([cost[(i, j)] for i in range(n) for j in range(n) if i != j], dtype=float)
    mean = _mean(arcs)
    skew, kurtosis = _skew_kurtosis(arcs)
    long_ratio, short_ratio = _tail_ratios(arcs)
    outgoing = [[cost[(i, j)] for j in range(n) if j != i] for i in range(n)]
    incoming = [[cost[(i, j)] for i in range(n) if i != j] for j in range(n)]
    out_nearest = _nearest_statistics(outgoing, mean)
    in_nearest = _nearest_statistics(incoming, mean)
    out_means = np.asarray([np.mean(row) for row in outgoing], dtype=float)
    in_means = np.asarray([np.mean(row) for row in incoming], dtype=float)
    pair_gaps = np.asarray(
        [abs(cost[(i, j)] - cost[(j, i)]) for i in range(n) for j in range(i + 1, n)],
        dtype=float,
    )
    reverse_arcs = np.asarray(
        [cost[(j, i)] for i in range(n) for j in range(n) if i != j], dtype=float
    )
    denominator = max(abs(mean), 1e-12)
    out_mean = _mean(out_means)
    in_mean = _mean(in_means)
    return {
        "n": float(n),
        "directed_cost_mean": mean,
        "directed_cost_cv": _cv(arcs),
        "directed_cost_skew": skew,
        "directed_cost_kurtosis": kurtosis,
        "directed_cost_range_ratio": _range_ratio(arcs),
        "directed_long_edge_ratio": long_ratio,
        "directed_short_edge_ratio": short_ratio,
        "directed_tie_ratio": _tie_ratio(arcs),
        "out_nearest_mean_ratio": out_nearest["mean_ratio"],
        "out_nearest_cv": out_nearest["cv"],
        "in_nearest_mean_ratio": in_nearest["mean_ratio"],
        "in_nearest_cv": in_nearest["cv"],
        "out_first_second_ratio": out_nearest["first_second_ratio"],
        "in_first_second_ratio": in_nearest["first_second_ratio"],
        "outgoing_mean_cv": _cv(out_means),
        "incoming_mean_cv": _cv(in_means),
        "in_out_mean_gap_ratio": float(np.mean(np.abs(out_means - in_means)) / denominator),
        "outgoing_node_eccentricity": float(np.max(out_means) / abs(out_mean)) if abs(out_mean) > 1e-12 else 0.0,
        "incoming_node_eccentricity": float(np.max(in_means) / abs(in_mean)) if abs(in_mean) > 1e-12 else 0.0,
        "reciprocal_abs_gap_mean_ratio": _mean(pair_gaps) / denominator,
        "reciprocal_abs_gap_max_ratio": float(np.max(pair_gaps) / denominator),
        "reciprocal_abs_gap_cv": _cv(pair_gaps),
        "reciprocal_cost_correlation": _correlation(arcs, reverse_arcs),
        "reciprocal_equal_ratio": float(np.mean(pair_gaps == 0.0)),
    }


def _mst_edges(n: int, cost: dict[tuple[int, int], float]) -> np.ndarray:
    selected = {0}
    edges = []
    while len(selected) < n:
        candidates = [
            (cost[(i, j)], i, j)
            for i in selected
            for j in range(n)
            if j not in selected
        ]
        if not candidates:
            raise ValueError("symmetric complete graph is disconnected")
        value, _, node = min(candidates)
        selected.add(node)
        edges.append(float(value))
    return np.asarray(edges, dtype=float)


def _component_count(n: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    return len({find(node) for node in range(n)})


def _nearest_reciprocity(n: int, cost: dict[tuple[int, int], float]) -> float:
    nearest = {}
    for i in range(n):
        values = {j: cost[(i, j)] for j in range(n) if j != i}
        minimum = min(values.values())
        nearest[i] = {j for j, value in values.items() if value == minimum}
    relations = [(i, j) for i in range(n) for j in nearest[i]]
    if not relations:
        return 0.0
    return float(np.mean([i in nearest[j] for i, j in relations]))


def extract_stsp_features(instance: dict[str, Any]) -> dict[str, float]:
    n, cost = _cost_mapping(instance)
    if any(cost[(i, j)] != cost[(j, i)] for i in range(n) for j in range(i + 1, n)):
        raise ValueError("STSP feature extraction requires a symmetric matrix")
    if n < 4:
        raise ValueError("STSP feature extraction requires at least four nodes")

    edges = np.asarray([cost[(i, j)] for i in range(n) for j in range(i + 1, n)], dtype=float)
    mean = _mean(edges)
    denominator = max(abs(mean), 1e-12)
    skew, kurtosis = _skew_kurtosis(edges)
    long_ratio, short_ratio = _tail_ratios(edges)
    rows = [[cost[(i, j)] for j in range(n) if j != i] for i in range(n)]
    nearest = _nearest_statistics(rows, mean)
    row_means = np.asarray([np.mean(row) for row in rows], dtype=float)

    violations = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                violations.append(max(0.0, cost[(i, k)] - cost[(i, j)] - cost[(j, k)]))
    violation_values = np.asarray(violations, dtype=float)
    mst = _mst_edges(n, cost)
    threshold = float(np.quantile(edges, 0.25))
    low_edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if cost[(i, j)] <= threshold
    ]
    degrees = np.zeros(n, dtype=float)
    for i, j in low_edges:
        degrees[i] += 1.0
        degrees[j] += 1.0
    possible_edges = n * (n - 1) / 2.0
    row_mean = _mean(row_means)
    return {
        "n": float(n),
        "edge_cost_mean": mean,
        "edge_cost_cv": _cv(edges),
        "edge_cost_skew": skew,
        "edge_cost_kurtosis": kurtosis,
        "edge_cost_range_ratio": _range_ratio(edges),
        "edge_long_ratio": long_ratio,
        "edge_short_ratio": short_ratio,
        "edge_tie_ratio": _tie_ratio(edges),
        "nearest_mean_ratio": nearest["mean_ratio"],
        "nearest_cv": nearest["cv"],
        "first_second_ratio": nearest["first_second_ratio"],
        "node_mean_cv": _cv(row_means),
        "node_eccentricity": float(np.max(row_means) / abs(row_mean)) if abs(row_mean) > 1e-12 else 0.0,
        "triangle_violation_ratio": float(np.mean(violation_values > 0.0)),
        "triangle_violation_mean_ratio": _mean(violation_values[violation_values > 0.0]) / denominator,
        "mst_weight_ratio": float(np.sum(mst) / max((n - 1) * denominator, 1e-12)),
        "mst_edge_cv": _cv(mst),
        "low_cost_graph_density": float(len(low_edges) / possible_edges),
        "low_cost_component_ratio": float(_component_count(n, low_edges) / n),
        "low_cost_degree_cv": _cv(degrees),
        "nearest_reciprocity": _nearest_reciprocity(n, cost),
    }


def extract_variant_features(variant: str, instance: dict[str, Any]) -> dict[str, float]:
    normalized = str(variant).upper()
    values = extract_atsp_features(instance) if normalized == "ATSP" else extract_stsp_features(instance)
    expected = ATSP_FEATURE_NAMES if normalized == "ATSP" else STSP_FEATURE_NAMES
    if tuple(values) != expected:
        raise AssertionError("variant feature schema does not match the registry")
    if not np.isfinite(np.asarray(list(values.values()), dtype=float)).all():
        raise ValueError("variant features must be finite")
    return values


def feature_registry(variant: str) -> dict[str, Any]:
    normalized = str(variant).upper()
    if normalized not in {"ATSP", "STSP"}:
        raise ValueError(f"unsupported TSP variant: {variant}")
    names = ATSP_FEATURE_NAMES if normalized == "ATSP" else STSP_FEATURE_NAMES
    formulas = _ATSP_FORMULAS if normalized == "ATSP" else _STSP_FORMULAS
    return {
        "version": FEATURE_REGISTRY_VERSION,
        "problem": "TSP",
        "problem_variant": normalized,
        "performance_features": [],
        "coordinate_features": [],
        "feature_names": list(names),
        "features": [
            {
                "name": name,
                "formula": formulas[name][0],
                "scale_property": formulas[name][1],
                "applicable_variant": normalized,
                "source": "problem parameters only",
            }
            for name in names
        ],
        "numeric_conventions": {
            "zero_denominator": 0.0,
            "constant_skew_and_kurtosis": 0.0,
            "constant_correlation": "1 when the arrays are equal, otherwise 0",
            "all_outputs_must_be_finite": True,
        },
        "source": "repository problem-parameter feature extension; no solver-performance inputs",
    }
