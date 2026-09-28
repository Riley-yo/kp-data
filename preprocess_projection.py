from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import boxcox


@dataclass
class PrelimTransformer:
    feature_names: list[str]
    shifts: dict[str, float]
    lambdas: dict[str, float | None]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: list[str]):
        transformed = {}
        shifts = {}
        lambdas = {}
        for name in feature_names:
            values = frame[name].to_numpy(dtype=float)
            minimum = float(np.min(values))
            shift = 1.0 - minimum if minimum <= 0.0 else 0.0
            shifted = values + shift
            shifts[name] = shift
            if np.allclose(shifted, shifted[0]):
                transformed[name] = shifted
                lambdas[name] = None
            else:
                transformed[name], lambdas[name] = boxcox(shifted)

        means = {name: float(np.mean(values)) for name, values in transformed.items()}
        stds = {}
        for name, values in transformed.items():
            standard_deviation = float(np.std(values))
            stds[name] = standard_deviation if standard_deviation > 0.0 else 1.0
        return cls(list(feature_names), shifts, lambdas, means, stds)

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            values = frame[name].to_numpy(dtype=float) + self.shifts[name]
            values = np.maximum(values, 1e-9)
            exponent = self.lambdas[name]
            if exponent is None:
                transformed = values
            elif abs(exponent) < 1e-12:
                transformed = np.log(values)
            else:
                transformed = (np.power(values, exponent) - 1.0) / exponent
            columns.append((transformed - self.means[name]) / self.stds[name])
        return np.column_stack(columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "shifts": self.shifts,
            "lambdas": self.lambdas,
            "means": self.means,
            "stds": self.stds,
        }


@dataclass
class PilotResult:
    projection: np.ndarray
    coords: np.ndarray
    feature_names: list[str]
    error: float
    reconstruction: np.ndarray


def _projection_error(
    flattened_projection: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    feature_count: int,
) -> float:
    projection = flattened_projection.reshape(feature_count, 2)
    coordinates = features @ projection
    reconstruction, *_ = np.linalg.lstsq(coordinates, target, rcond=None)
    residual = target - coordinates @ reconstruction
    return float(np.sum(residual * residual))


def pilot(
    transformed_selected: pd.DataFrame,
    performance: pd.DataFrame,
    n_restarts: int = 5,
    seed: int = 0,
) -> PilotResult:
    features = transformed_selected.to_numpy(dtype=float)
    instance_count, feature_count = features.shape
    performance_values = performance.to_numpy(dtype=float)
    standardized_performance = (
        performance_values - performance_values.mean(axis=0)
    ) / np.where(
        performance_values.std(axis=0) > 0.0,
        performance_values.std(axis=0),
        1.0,
    )
    target = np.column_stack([features, standardized_performance])

    random = np.random.default_rng(seed)
    _, _, right_vectors = np.linalg.svd(
        features - features.mean(axis=0), full_matrices=False
    )
    starts = [right_vectors[:2].T.reshape(-1)]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(random.normal(scale=0.5, size=feature_count * 2))

    if instance_count >= 200 or instance_count * feature_count > 2500:
        projection = starts[0].reshape(feature_count, 2)
        coordinates = features @ projection
        reconstruction, *_ = np.linalg.lstsq(coordinates, target, rcond=None)
        return PilotResult(
            projection=projection,
            coords=coordinates,
            feature_names=list(transformed_selected.columns),
            error=_projection_error(starts[0], features, target, feature_count),
            reconstruction=reconstruction,
        )

    best = None
    for start in starts:
        result = minimize(
            _projection_error,
            start,
            args=(features, target, feature_count),
            method="BFGS",
            options={"maxiter": 250, "gtol": 1e-5},
        )
        if best is None or result.fun < best.fun:
            best = result

    projection = best.x.reshape(feature_count, 2)
    coordinates = features @ projection
    reconstruction, *_ = np.linalg.lstsq(coordinates, target, rcond=None)
    return PilotResult(
        projection=projection,
        coords=coordinates,
        feature_names=list(transformed_selected.columns),
        error=float(best.fun),
        reconstruction=reconstruction,
    )

