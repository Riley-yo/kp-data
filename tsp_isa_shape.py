from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_THRESHOLDS = {
    "covariance_ratio_max": 1.25,
    "radial_uniform_max_deviation_max": 0.15,
    "fitted_disk_inside_ratio_min": 0.95,
    "sector_cv_above_theoretical_minimum_max": 0.15,
}

PARETO_TOLERANCES = {
    "covariance_ratio": 0.05,
    "radial_uniform_max_deviation": 0.02,
    "fitted_disk_inside_ratio": 0.02,
    "sector_cv_above_theoretical_minimum": 0.03,
}


def _sector_cv(angle: np.ndarray, sector_count: int) -> float:
    width = 2.0 * np.pi / sector_count
    normalized = np.mod(angle, 2.0 * np.pi)
    residues = np.mod(normalized, width)
    order = np.argsort(residues, kind="stable")
    sorted_residues = residues[order]
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, width)
    if (
        sorted_residues[0] <= tolerance
        or width - sorted_residues[-1] <= tolerance
        or np.any(np.diff(sorted_residues) <= tolerance)
    ):
        boundaries = np.concatenate(
            [sorted_residues, [sorted_residues[0] + width]]
        )
        phases = np.mod(
            (boundaries[:-1] + boundaries[1:]) / 2.0, width
        )
        return min(
            float(
                np.bincount(
                    np.floor(
                        np.mod(angle - phase, 2.0 * np.pi) / width
                    ).astype(int),
                    minlength=sector_count,
                ).std()
                / max(len(angle) / sector_count, 1e-12)
            )
            for phase in phases
        )
    sectors = np.floor(normalized / width).astype(int)
    counts = np.bincount(sectors, minlength=sector_count).astype(int)
    best = float(counts.std() / max(counts.mean(), 1e-12))
    base_sectors = np.floor(normalized / width).astype(int)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and sorted_residues[end] == sorted_residues[start]
        ):
            end += 1
        for point_index in order[start:end]:
            before = int(base_sectors[point_index])
            counts[before] -= 1
            counts[(before - 1) % sector_count] += 1
        best = min(
            best,
            float(counts.std() / max(counts.mean(), 1e-12)),
        )
        start = end
    return best


def shape_metrics(
    coordinates: np.ndarray,
    thresholds: dict[str, float] | None = None,
    sector_count: int = 12,
) -> dict[str, Any]:
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("coordinates must have shape (n, 2), n >= 2")
    if not np.isfinite(values).all() or np.iscomplexobj(coordinates):
        raise ValueError("coordinates must be finite real numbers")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= np.finfo(float).eps * eigenvalues[-1]:
        covariance_ratio = float(np.finfo(float).max)
    else:
        covariance_ratio = float(eigenvalues[-1] / eigenvalues[0])
    squared_radius = np.sum(centered * centered, axis=1)
    fitted_radius = float(math.sqrt(2.0 * float(squared_radius.mean())))
    normalized_squared_radius = np.sort(
        squared_radius / max(fitted_radius * fitted_radius, np.finfo(float).tiny)
    )
    radial_target = (np.arange(len(values)) + 0.5) / len(values)
    radial_deviation = float(np.max(np.abs(normalized_squared_radius - radial_target)))
    inside_ratio = float(np.mean(np.sqrt(squared_radius) <= 1.02 * fitted_radius))
    angle = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2.0 * np.pi)
    sector_cv = _sector_cv(angle, sector_count)
    remainder = len(values) % sector_count
    sector_minimum = float(
        math.sqrt(remainder * (sector_count - remainder)) / len(values)
    )
    sector_excess = max(0.0, sector_cv - sector_minimum)
    terms = {
        "log_covariance_ratio": math.log(covariance_ratio),
        "radial_uniform_max_deviation": radial_deviation,
        "outside_fitted_disk_ratio": 1.0 - inside_ratio,
        "sector_cv_above_theoretical_minimum": sector_excess,
    }
    score = float(
        terms["log_covariance_ratio"]
        + 2.0 * terms["radial_uniform_max_deviation"]
        + terms["outside_fitted_disk_ratio"]
        + terms["sector_cv_above_theoretical_minimum"]
    )
    return {
        "metric_version": "center-fit-scale-free-disk-v1",
        "sample_count": len(values),
        "fitted_radius": fitted_radius,
        "coordinate_min": values.min(axis=0).tolist(),
        "coordinate_max": values.max(axis=0).tolist(),
        "covariance_ratio": covariance_ratio,
        "radial_uniform_max_deviation": radial_deviation,
        "fitted_disk_inside_ratio": inside_ratio,
        "sector_count": sector_count,
        "sector_cv": sector_cv,
        "sector_cv_theoretical_minimum": sector_minimum,
        "sector_cv_above_theoretical_minimum": sector_excess,
        "composite_terms": terms,
        "composite_score": score,
        "target_thresholds": limits,
        "target_met": bool(
            covariance_ratio <= limits["covariance_ratio_max"]
            and radial_deviation <= limits["radial_uniform_max_deviation_max"]
            and inside_ratio >= limits["fitted_disk_inside_ratio_min"]
            and sector_excess <= limits["sector_cv_above_theoretical_minimum_max"]
        ),
    }


def normalized_deficit(metrics: dict[str, Any], thresholds=None) -> dict[str, Any]:
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    terms = {
        "covariance_ratio": max(
            metrics["covariance_ratio"] - limits["covariance_ratio_max"], 0.0
        ) / limits["covariance_ratio_max"],
        "radial_uniform_max_deviation": max(
            metrics["radial_uniform_max_deviation"]
            - limits["radial_uniform_max_deviation_max"],
            0.0,
        ) / limits["radial_uniform_max_deviation_max"],
        "fitted_disk_inside_ratio": max(
            limits["fitted_disk_inside_ratio_min"]
            - metrics["fitted_disk_inside_ratio"],
            0.0,
        ) / limits["fitted_disk_inside_ratio_min"],
        "sector_cv_above_theoretical_minimum": max(
            metrics["sector_cv_above_theoretical_minimum"]
            - limits["sector_cv_above_theoretical_minimum_max"],
            0.0,
        ) / limits["sector_cv_above_theoretical_minimum_max"],
    }
    return {"terms": terms, "total": float(sum(terms.values()))}


def acceptance_audit(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    min_delta: float = 0.0001,
    thresholds=None,
    tolerances=None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    tolerance = dict(PARETO_TOLERANCES if tolerances is None else tolerances)
    before_deficit = normalized_deficit(before, limits)
    after_deficit = normalized_deficit(after, limits)
    deltas = {
        "covariance_ratio": before["covariance_ratio"] - after["covariance_ratio"],
        "radial_uniform_max_deviation": before["radial_uniform_max_deviation"]
        - after["radial_uniform_max_deviation"],
        "fitted_disk_inside_ratio": after["fitted_disk_inside_ratio"]
        - before["fitted_disk_inside_ratio"],
        "sector_cv_above_theoretical_minimum": before[
            "sector_cv_above_theoretical_minimum"
        ]
        - after["sector_cv_above_theoretical_minimum"],
    }
    tolerance_violations = [
        name for name, delta in deltas.items() if delta < -tolerance[name] - 1e-15
    ]
    boundary_violations = []
    for name, threshold_name in (
        ("covariance_ratio", "covariance_ratio_max"),
        ("radial_uniform_max_deviation", "radial_uniform_max_deviation_max"),
        (
            "sector_cv_above_theoretical_minimum",
            "sector_cv_above_theoretical_minimum_max",
        ),
    ):
        if before[name] <= limits[threshold_name] and after[name] > limits[threshold_name]:
            boundary_violations.append(name)
    if (
        before["fitted_disk_inside_ratio"] >= limits["fitted_disk_inside_ratio_min"]
        and after["fitted_disk_inside_ratio"] < limits["fitted_disk_inside_ratio_min"]
    ):
        boundary_violations.append("fitted_disk_inside_ratio")
    improvement = before_deficit["total"] - after_deficit["total"]
    passed = (
        not tolerance_violations
        and not boundary_violations
        and (after_deficit["total"] <= 1e-15 or improvement >= min_delta)
    )
    return {
        "passed": passed,
        "before_deficit": before_deficit,
        "after_deficit": after_deficit,
        "deficit_improvement": improvement,
        "metric_improvements": deltas,
        "tolerances": tolerance,
        "tolerance_violations": tolerance_violations,
        "boundary_violations": boundary_violations,
    }
