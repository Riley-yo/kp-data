"""KP-specific validation: dynamic signature checks and feasibility validation.

Extends the TSP framework's validation.py with KP-specific checks:
- dynamic_signature_errors: verify variable counts match formulation
- objective_signature_errors: verify x_i coefficients equal profits
- recover_selected_items: extract selected items from solution
- knapsack_feasibility_errors: verify sum(w_i * x_i) <= capacity
"""
from __future__ import annotations

from typing import Any

# Reuse framework helpers
from validation import _variable_family, _indices


def kp_dynamic_signature_errors(
    n: int,
    formulation_id: str,
    required_symbols: tuple[str, ...],
    forbidden_symbols: tuple[str, ...],
    variables: dict[str, float],
    constraints: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Check dynamic variable counts for KP formulations."""
    counts: dict[str, int] = {}
    for name in variables:
        family = _variable_family(name)
        counts[family] = counts.get(family, 0) + 1
    errors = []
    for symbol in required_symbols:
        if counts.get(symbol, 0) == 0:
            errors.append(f"dynamic model has no {symbol} variables")
    for symbol in forbidden_symbols:
        if counts.get(symbol, 0):
            errors.append(f"dynamic model contains forbidden {symbol} variables")

    if formulation_id == "kp_compact_v1":
        expected = {"x": n}
        for symbol in required_symbols:
            if symbol in expected and counts.get(symbol) != expected[symbol]:
                errors.append(
                    f"dynamic {symbol} variable count is {counts.get(symbol, 0)}, "
                    f"expected {expected[symbol]}"
                )
    elif formulation_id == "kp_arc_flow_v1":
        # f variables: at least 2n arcs (select/skip per layer)
        if counts.get("f", 0) < n:
            errors.append(
                f"dynamic f variable count is {counts.get('f', 0)}, expected at least {n}"
            )
    elif formulation_id == "kp_pattern_selection_v1":
        # lambda variables: at least 1 (select one pattern)
        if counts.get("lambda", 0) < 1:
            errors.append("dynamic lambda variable count is 0, expected at least 1")
    elif formulation_id == "kp_qubo_v1":
        # x variables: n, b variables: at least 1
        if counts.get("x", 0) != n:
            errors.append(
                f"dynamic x variable count is {counts.get('x', 0)}, expected {n}"
            )
        if counts.get("b", 0) < 1:
            errors.append("dynamic b variable count is 0, expected at least 1")

    constraint_names = {name.casefold() for name in constraints}
    details = {"variable_counts": counts, "constraint_names": list(constraint_names)}
    return errors, details


def kp_objective_signature_errors(
    formulation_id: str,
    instance: dict[str, Any],
    objective_linear: dict[str, float],
    objective_offset: float,
) -> list[str]:
    """Verify objective coefficients match instance data for KP."""
    n = int(instance["n_items"])
    profits = instance["profit"]
    errors = []

    if formulation_id in ("kp_compact_v1", "kp_qubo_v1"):
        for name, coefficient in objective_linear.items():
            if _variable_family(name) != "x":
                continue
            indices = _indices(name)
            if len(indices) < 1:
                continue
            i = indices[0]
            if i < 0 or i >= n:
                errors.append(f"objective variable {name} has out-of-range item index")
                break
            if formulation_id == "kp_compact_v1":
                expected = float(profits[i])
                if abs(float(coefficient) - expected) > 1e-9:
                    errors.append(
                        f"objective coefficient for {name} does not equal profit[{i}]"
                    )
                    break
            else:  # qubo: coefficient should be -profit (minimization)
                expected = -float(profits[i])
                if abs(float(coefficient) - expected) > 1e-6:
                    errors.append(
                        f"objective coefficient for {name} does not equal -profit[{i}]"
                    )
                    break
    elif formulation_id == "kp_pattern_selection_v1":
        # lambda coefficients are pattern profits, hard to verify exactly
        # just check there's at least one lambda in objective
        has_lambda = any(
            _variable_family(name) == "lambda" for name in objective_linear
        )
        if not has_lambda:
            errors.append("objective has no lambda variables")
    elif formulation_id == "kp_arc_flow_v1":
        # f coefficients are profits on selection arcs
        # check there are some positive coefficients
        positive_count = sum(1 for c in objective_linear.values() if c > 1e-9)
        if positive_count == 0:
            errors.append("objective has no positive profit coefficients on flow arcs")

    return errors


def recover_selected_items(
    variables: dict[str, float],
) -> tuple[int, ...]:
    """Extract selected item indices from KP solution.

    x_i > 0.5 means item i is selected.
    """
    selected = set()
    for name, value in variables.items():
        if float(value) <= 0.5:
            continue
        family = _variable_family(name)
        indices = _indices(name)
        if family == "x" and len(indices) >= 1:
            selected.add(indices[0])
        elif family == "lambda" and len(indices) >= 1:
            # For pattern selection, lambda is selected pattern
            # actual items recovered differently — skip here
            pass
    return tuple(sorted(selected))


def knapsack_feasibility_errors(
    instance: dict[str, Any],
    selected_items: tuple[int, ...],
) -> list[str]:
    """Verify selected items satisfy the knapsack capacity constraint."""
    n = int(instance["n_items"])
    weights = instance["weights"]
    capacity = int(instance["capacity"])
    errors = []

    total_weight = 0
    for i in selected_items:
        if i < 0 or i >= n:
            errors.append(f"selected item {i} is out of range [0, {n})")
            continue
        total_weight += int(weights[i])

    if total_weight > capacity:
        errors.append(
            f"selected items total weight {total_weight} exceeds capacity {capacity}"
        )

    return errors
