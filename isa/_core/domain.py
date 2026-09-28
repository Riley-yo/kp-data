import math
import numpy as np
from errors import Unsupported
from tsp_features import extract_variant_features
from qkp_features import _qkp_input_features


def pack(value):
    if isinstance(value, dict):
        return {"__map__": [[pack(k), pack(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]}
    if isinstance(value, tuple):
        return {"__tuple__": [pack(v) for v in value]}
    if isinstance(value, (list, set)):
        return [pack(v) for v in (sorted(value, key=repr) if isinstance(value, set) else value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def unpack(value):
    if isinstance(value, dict) and "__map__" in value:
        return {unpack(k): unpack(v) for k, v in value["__map__"]}
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(unpack(v) for v in value["__tuple__"])
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def numbers(value):
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise Unsupported("nonfinite_parameter")
        return [float(value)]
    if isinstance(value, dict):
        return [x for _, v in sorted(value.items(), key=lambda kv: repr(kv[0])) for x in numbers(v)]
    if isinstance(value, (list, tuple, set)):
        return [x for v in value for x in numbers(v)]
    raise Unsupported("nonnumeric_parameter")


def stats(prefix, values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise Unsupported("empty_or_nonfinite_feature_input:" + prefix)
    mean, std = float(a.mean()), float(a.std())
    z = (a - mean) / std if std > 1e-12 else np.zeros_like(a)
    q = np.quantile(a, [0.25, 0.5, 0.75])
    return {prefix + "_" + k: float(v) for k, v in {
        "count": len(a), "mean": mean, "var": std ** 2, "std": std, "min": a.min(), "max": a.max(),
        "q25": q[0], "median": q[1], "q75": q[2], "zero_ratio": np.mean(a == 0),
        "negative_ratio": np.mean(a < 0), "unique_ratio": len(np.unique(a)) / len(a),
        "cv": std / abs(mean) if abs(mean) > 1e-12 else 0,
        "skew": np.mean(z ** 3), "kurtosis": np.mean(z ** 4) - 3 if std > 1e-12 else 0,
    }.items()}


def matrix(value):
    if isinstance(value, dict):
        keys = list(value)
        if keys and all(isinstance(k, tuple) and len(k) == 2 for k in keys):
            rows = sorted({k[0] for k in keys}, key=repr)
            cols = sorted({k[1] for k in keys}, key=repr)
            if len(rows) * len(cols) != len(keys):
                raise Unsupported("incomplete_dense_matrix")
            a = np.asarray([[value[i, j] for j in cols] for i in rows], dtype=float)
        else:
            a = np.asarray([numbers(v) for _, v in sorted(value.items(), key=lambda kv: repr(kv[0]))], dtype=float)
    else:
        a = np.asarray(value, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise Unsupported("expected_dense_matrix")
    return a


def vector(value):
    if isinstance(value, dict):
        value = [v for _, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise Unsupported("expected_vector")
    return a


def matrix_features(prefix, a):
    out = stats(prefix, a.ravel())
    out.update({prefix + "_rows": float(a.shape[0]), prefix + "_columns": float(a.shape[1]),
                prefix + "_density": float(np.mean(a != 0)),
                prefix + "_row_sum_cv": float(np.std(a.sum(1)) / max(np.mean(np.abs(a.sum(1))), 1e-12)),
                prefix + "_col_sum_cv": float(np.std(a.sum(0)) / max(np.mean(np.abs(a.sum(0))), 1e-12))})
    return out


def incidence_features(value):
    if isinstance(value, dict) and "__incidence_matrix__" in value:
        a = matrix(value["__incidence_matrix__"])
        counts = np.count_nonzero(a, axis=1).tolist()
        degree = np.count_nonzero(a, axis=0).tolist()
        weights = a[a != 0].tolist()
        group_count, universe_size = a.shape
    else:
        groups = list(value.values()) if isinstance(value, dict) else list(value)
        if not groups or any(not isinstance(x, (list, tuple, set, dict)) for x in groups):
            raise Unsupported("incidence_requires_explicit_sets")
        sets = [set(x.keys() if isinstance(x, dict) else x) for x in groups]
        universe = set.union(*sets)
        if not universe:
            raise Unsupported("empty_incidence_universe")
        counts = [len(x) for x in sets]
        degree = [sum(x in s for s in sets) for x in universe]
        weights = [1.0] * sum(counts)
        group_count, universe_size = len(sets), len(universe)
    if not weights:
        raise Unsupported("empty_incidence")
    return {**stats("set_size", counts), **stats("element_degree", degree),
            **stats("incidence_nonzero_weight", weights),
            "incidence_group_count": float(group_count), "incidence_universe_size": float(universe_size),
            "incidence_density": sum(counts) / (group_count * universe_size)}


def check_profile(spec, data):
    kind = spec["kind"]
    if kind == "transport":
        c, s, d = matrix(data["cost"]), vector(data["supply"]), vector(data["demand"])
        if c.shape != (len(s), len(d)):
            raise Unsupported("transport_dimensions")
        if (s < 0).any() or (d < 0).any() or d.sum() <= 0:
            raise Unsupported("transport_nonnegative_supply_demand")
    if kind == "knapsack":
        w, p = vector(data["weights"]), vector(data["profit"])
        if len(w) != len(p) or np.any(w <= 0) or data["capacity"] <= 0:
            raise Unsupported("knapsack_dimensions_or_capacity")
    if kind == "facility":
        c = matrix(data["cost"])
        if c.shape != (len(vector(data["capacity"])), len(vector(data["demand"]))) or len(vector(data["fixed_cost"])) != c.shape[0]:
            raise Unsupported("facility_dimensions")
    if kind == "flowshop":
        if np.any(matrix(data["processing_time"]) < 0):
            raise Unsupported("negative_processing_time")
    if kind == "landing":
        vectors = [vector(data[k]) for k in ("earliest", "target", "latest", "early_penalty", "late_penalty")]
        if len({len(x) for x in vectors}) != 1:
            raise Unsupported("landing_dimensions")
        if np.any(vectors[0] > vectors[1]) or np.any(vectors[1] > vectors[2]):
            raise Unsupported("landing_time_window")
    if kind == "routing":
        c = matrix(data["cost"])
        if c.shape[0] != c.shape[1] or c.shape[0] < 3:
            raise Unsupported("routing_requires_node_to_node_matrix")
        if "demand" in data and len(vector(data["demand"])) not in (c.shape[0], c.shape[0] - 1):
            raise Unsupported("routing_demand_dimensions")


def feature_row(spec, variant, data):
    key = spec["problem"]
    if spec["kind"] == "model":
        raise Unsupported("Expert formulation features are disabled")
    if key == "TSP":
        return extract_variant_features(variant, data)
    if key == "BPP":
        if len(data["sizes"]) > 400:
            raise Unsupported("packing_cubic_feature_size_limit")
        raise Unsupported("Use intrinsic.packing_features")
    if key == "QKP":
        if len(data["weights"]) > 400:
            raise Unsupported("quadratic_cubic_feature_size_limit")
        return _qkp_input_features(data["weights"], data["profits"], data["pair_profit"], data["capacity"])
    if key == "JSSP":
        if variant == "processing_matrix":
            a = matrix(data["processing_time"])
            return {**matrix_features("processing_time", a), **stats("job_load", a.sum(1)), **stats("machine_load", a.sum(0))}
        return {**stats("processing_time", numbers(data["processing_time"])), "conflict_count": float(len(data["conflicts"]))}
    out = {}
    for name, field in spec["fields"].items():
        value = data[name]
        if field["type"] == "matrix":
            out.update(matrix_features(name, matrix(value)))
        elif field["type"] == "incidence":
            out.update({name + "_" + k: v for k, v in incidence_features(value).items()})
        elif field["type"] == "edges":
            edges = list(value) if not isinstance(value, dict) else list(value.items())
            if not all(isinstance(e, (list, tuple)) and len(e) >= 2 for e in edges):
                raise Unsupported("edge_pairs_required")
            out[name + "_count"] = float(len(edges))
            vertices = {repr(x) for e in edges for x in e[:2]}
            out[name + "_vertex_count"] = float(len(vertices))
        elif field["type"] == "labels":
            out[name + "_count"] = float(len(value))
        else:
            out.update(stats(name, numbers(value)))
    if spec["kind"] == "transport":
        s, d = vector(data["supply"]), vector(data["demand"])
        out["supply_demand_ratio"] = float(s.sum() / d.sum())
    if spec["kind"] == "knapsack":
        w, p = vector(data["weights"]), vector(data["profit"])
        out["capacity_total_weight_ratio"] = float(data["capacity"] / w.sum())
        out.update(stats("profit_weight_ratio", p / w))
    if spec["kind"] == "flowshop":
        p = matrix(data["processing_time"])
        out.update(stats("job_load", p.sum(1)))
        out.update(stats("machine_load", p.sum(0)))
    if spec["kind"] == "landing":
        out.update(stats("window_width", vector(data["latest"]) - vector(data["earliest"])))
    return out
