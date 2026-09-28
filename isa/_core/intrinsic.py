import math
import numpy as np
import domain
from errors import Unsupported


def aligned_vectors(data, names):
    values = [data[k] for k in names]
    vectors = [domain.vector(v) for v in values]
    if len({len(v) for v in vectors}) != 1:
        raise Unsupported("input_entity_dimension_mismatch:" + "/".join(names))
    mappings = [v for v in values if isinstance(v, dict)]
    if mappings and (len(mappings) != len(values) or any(set(v) != set(mappings[0]) for v in mappings)):
        raise Unsupported("input_entity_key_mismatch:" + "/".join(names))
    return vectors


def validate_profile(variant, data):
    if variant == "budget_allocation":
        aligned_vectors(data, ["unit_cost", "benefit"])
    elif variant == "budgeted_project_selection":
        aligned_vectors(data, ["project_cost", "project_benefit"])
    elif variant == "bounded_resource_allocation":
        lower, upper = aligned_vectors(data, ["minimum_allocation", "maximum_allocation"])
        if (lower > upper).any():
            raise Unsupported("allocation_lower_above_upper")
    elif variant == "bounded_sales":
        _, lower, upper = aligned_vectors(data, ["unit_profit", "minimum_sales", "maximum_sales"])
        if (lower > upper).any():
            raise Unsupported("sales_lower_above_upper")
    elif variant == "price_cost_demand":
        aligned_vectors(data, list(data))
    elif variant in ("linear_generation", "capacity_acquisition"):
        capacity_key = "output_capacity" if variant == "linear_generation" else "unit_capacity"
        aligned_vectors(data, [capacity_key, "unit_cost"])
    elif variant == "land_limits":
        aligned_vectors(data, ["unit_profit", "maximum_crop_area"])
    elif variant == "yield_cost_price":
        aligned_vectors(data, ["yield_per_area", "planting_cost_per_area", "selling_price_per_output"])
    elif variant in ("resource_product_mix", "crop_resources"):
        a = domain.matrix(data["resource_requirement"])
        if a.shape not in [(len(domain.vector(data["unit_profit"])), len(domain.vector(data["resource_capacity"]))),
                           (len(domain.vector(data["resource_capacity"])), len(domain.vector(data["unit_profit"])) )]:
            raise Unsupported("resource_matrix_dimensions")
    elif variant == "explicit_input_matrix":
        a = domain.matrix(data["input_coefficients"])
        if a.shape != (len(domain.vector(data["input_rhs"])), len(domain.vector(data["input_objective"]))):
            raise Unsupported("explicit_input_matrix_dimensions")
    for name, value in data.items():
        if name == "initial_capacity":
            if min(domain.numbers(value)) < 0:
                raise Unsupported("negative_initial_capacity")
            continue
        if any(word in name for word in ("capacity", "budget", "total_land", "resource_available")):
            if min(domain.numbers(value)) <= 0:
                raise Unsupported("nonpositive_input_capacity:" + name)


def packing_features(data):
    sizes = np.asarray(data["sizes"], dtype=float)
    capacity = float(data["capacity"])
    if capacity <= 0 or not len(sizes) or np.any(sizes <= 0) or np.any(sizes > capacity):
        raise Unsupported("invalid_packing_dimensions")
    ratios = sizes / capacity
    total = float(ratios.sum())
    result = domain.stats("relative_size", ratios)
    result.update(item_count=float(len(sizes)), bin_capacity=capacity,
                  total_size_ratio=total, remaining_capacity_fraction=math.ceil(total) - total,
                  huge_item_ratio=float(np.mean(ratios > 0.5)),
                  large_item_ratio=float(np.mean((ratios > 1/3) & (ratios <= 0.5))),
                  medium_item_ratio=float(np.mean((ratios > 0.25) & (ratios <= 1/3))),
                  small_item_ratio=float(np.mean((ratios > 0.1) & (ratios <= 0.25))),
                  tiny_item_ratio=float(np.mean(ratios <= 0.1)))
    return result


def feature_row(spec, variant, data):
    key = spec["problem"]
    if spec["kind"] == "model" or variant == "model_matrix":
        raise Unsupported("expert_model_features_disabled")
    if key == "BPP":
        return packing_features(data)
    if spec["kind"] != "intrinsic_profiles":
        return domain.feature_row(spec, variant, data)
    fields = spec["profiles"][variant]
    validate_profile(variant, data)
    result = domain.feature_row({**spec, "kind": "parameters", "fields": fields}, variant, data)
    if variant in ("budget_allocation", "budgeted_project_selection"):
        cost = domain.vector(data["unit_cost" if variant == "budget_allocation" else "project_cost"])
        result.update(domain.stats("cost_budget_ratio", cost / data["budget"]))
    elif variant == "linear_generation":
        result["total_demand_capacity_ratio"] = sum(domain.numbers(data["demand"])) / sum(domain.numbers(data["output_capacity"]))
    elif variant == "land_limits":
        result.update(domain.stats("crop_area_land_ratio", domain.vector(data["maximum_crop_area"]) / data["total_land"]))
    elif variant == "bounded_resource_allocation":
        result.update(domain.stats("minimum_resource_share", domain.vector(data["minimum_allocation"]) / data["resource_available"]))
    return result
