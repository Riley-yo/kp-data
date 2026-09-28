from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


REFERENCE = (
    "https://github.com/andremun/InstanceSpace/blob/"
    "e7fa8002d8979576ce53115f745ce6f9f3331df0/PILOT.m"
)


@dataclass
class PaperPilotResult:
    projection: np.ndarray
    coordinates: np.ndarray
    reconstruction: np.ndarray
    trials: list
    chosen_trial: int
    feature_names: list
    performance_names: list


def reconstruction_loss_gradient(theta, x, target):
    p = x.shape[1]
    a = theta[:2 * p].reshape(p, 2)
    b = theta[2 * p:].reshape(2, target.shape[1])
    z = x @ a
    residual = z @ b - target
    scale = 2.0 / target.size
    da = scale * x.T @ residual @ b.T
    db = scale * z.T @ residual
    return float(np.mean(residual ** 2)), np.concatenate([da.ravel(), db.ravel()])


def fit_paper_pilot(features, performance, n_restarts=5, seed=0, maxiter=2000):
    """Fit the numerical PILOT objective to already preprocessed X and Y.

    X contains selected intrinsic features only. Y is a separate measured
    performance response, preprocessed by the caller. No feature selection,
    response scaling, coordinate whitening or circularity optimisation occurs
    here. This is one stage of ISA, not a complete paper reproduction.
    """
    if not features.index.equals(performance.index):
        raise ValueError("Feature and performance instance IDs must match in order")
    if not features.index.is_unique:
        raise ValueError("Instance IDs must be unique; identical feature rows are allowed")
    if not features.columns.is_unique or not performance.columns.is_unique:
        raise ValueError("Feature and performance column names must be unique")
    if set(features.columns) & set(performance.columns):
        raise ValueError("Use separate intrinsic feature and performance column names")
    x = features.to_numpy(dtype=float, copy=True)
    y = performance.to_numpy(dtype=float, copy=True)
    if x.shape[0] < 3 or x.shape[1] < 2 or y.shape[1] == 0:
        raise ValueError("Need at least three instances, two features and one performance response")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing or non-finite values must be resolved without dropping original records")
    if np.linalg.matrix_rank(x - x.mean(axis=0)) < 2:
        raise ValueError("Intrinsic features cannot support a two-dimensional space")
    if not np.any(np.ptp(y, axis=0) > 0):
        raise ValueError("All performance responses are constant; no performance trend can be learned")
    if n_restarts < 1 or maxiter < 1:
        raise ValueError("n_restarts and maxiter must be positive")

    target = np.column_stack([x, y])
    high_distances = pdist(x)
    if np.std(high_distances) == 0:
        raise ValueError("Original PILOT distance-correlation trial selection is undefined")
    rng = np.random.default_rng(seed)
    trials, solutions = [], []
    for trial in range(n_restarts):
        start = rng.uniform(-1, 1, 2 * (x.shape[1] + target.shape[1]))
        result = minimize(
            reconstruction_loss_gradient, start, args=(x, target), jac=True,
            method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6},
        )
        a = result.x[:2 * x.shape[1]].reshape(x.shape[1], 2)
        b = result.x[2 * x.shape[1]:].reshape(2, target.shape[1])
        z = x @ a
        low_distances = pdist(z)
        correlation = float(np.corrcoef(high_distances, low_distances)[0, 1])
        valid = bool(result.success and np.isfinite(correlation)
                     and np.linalg.matrix_rank(z - z.mean(axis=0)) == 2)
        trials.append({
            "trial": trial, "converged": bool(result.success), "accepted_for_selection": valid,
            "message": str(result.message), "iterations": int(result.nit),
            "reconstruction_mse": float(result.fun),
            "distance_correlation": correlation if np.isfinite(correlation) else None,
        })
        solutions.append((a, z, b))
    eligible = [i for i, trial in enumerate(trials) if trial["accepted_for_selection"]]
    if not eligible:
        raise RuntimeError(f"No converged rank-two PILOT trial: {trials}")
    # The referenced numerical PILOT chooses trials by distance preservation.
    chosen = max(eligible, key=lambda i: trials[i]["distance_correlation"])
    a, z, b = solutions[chosen]
    return PaperPilotResult(a, z, b, trials, chosen, list(features.columns), list(performance.columns))
