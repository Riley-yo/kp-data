import math
from itertools import combinations
import re
import numpy as np
from domain import matrix, vector, stats, matrix_features


def compatibility_features(data):
    sizes = np.asarray(data["sizes"], dtype=float) / float(data["capacity"])
    n = len(sizes)
    pair_slack = 1.0 - sizes[:, None] - sizes[None, :]
    off_diagonal = ~np.eye(n, dtype=bool)
    compatible = (pair_slack >= -1e-12) & off_diagonal
    pairs = pair_slack[np.triu_indices(n, 1)]
    triples = np.array([1.0 - sizes[list(indices)].sum() for indices in combinations(range(n), 3)])
    partner_fraction = compatible.sum(1) / max(n - 1, 1)
    # Empty pair/triple sets have fraction zero; no_partner_item_ratio distinguishes this case.
    out = {"compat_no_partner_item_ratio": float(np.mean(compatible.sum(1) == 0)),
           "compat_size_gini": float(np.abs(sizes[:, None] - sizes[None, :]).sum() / (2 * n * sizes.sum()))}
    for name, slack in [("pair", pairs), ("triple", triples)]:
        feasible = np.maximum(slack[slack >= -1e-12], 0.0)
        out[f"compat_{name}_fit_ratio"] = float(np.mean(slack >= -1e-12)) if len(slack) else 0.0
        out[f"compat_{name}_exact_fit_ratio"] = float(np.mean(np.abs(slack) <= 1e-12)) if len(slack) else 0.0
        out[f"compat_{name}_feasible_slack_mean"] = float(feasible.mean()) if len(feasible) else 0.0
        out[f"compat_{name}_feasible_slack_std"] = float(feasible.std()) if len(feasible) else 0.0
    best_slack = [max(0.0, float(pair_slack[i, compatible[i]].min())) for i in range(n) if compatible[i].any()]
    for name, values in [("partner_fraction", partner_fraction), ("sorted_size_gap", np.diff(np.sort(sizes))),
                         ("best_pair_slack", np.asarray(best_slack))]:
        for quantile in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out[f"compat_{name}_q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile)) if len(values) else 0.0
        out[f"compat_{name}_mean"] = float(np.mean(values)) if len(values) else 0.0
        out[f"compat_{name}_std"] = float(np.std(values)) if len(values) else 0.0
    return out


def midranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def correlation(first, second):
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return 0.0
    first, second = first - first.mean(), second - second.mean()
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))


def knapsack_relation_features(data):
    weights, profits = data["weights"], data["profit"]
    if isinstance(weights, dict) or isinstance(profits, dict):
        assert isinstance(weights, dict) and isinstance(profits, dict)
        assert set(weights) == set(profits), "weight/profit item keys differ"
    w, p = vector(weights), vector(profits)
    capacity = float(data["capacity"])
    assert len(w) == len(p) and np.all(w > 0) and np.all(p >= 0) and capacity > 0
    n = len(w)
    r = w / capacity
    density = p / w
    density = density / density.mean() if density.mean() > 0 else np.zeros(n)
    slack = 1.0 - r[:, None] - r[None, :]
    i, j = np.triu_indices(n, 1)
    pair_slack = slack[i, j]
    fit = (slack >= -1e-12) & ~np.eye(n, dtype=bool)
    partner_fraction = fit.sum(1) / max(n - 1, 1)
    out = {}
    for name, values in [("relative_weight", r), ("relative_value_density", density),
                         ("pair_slack", pair_slack), ("partner_fraction", partner_fraction)]:
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out[f"kp_{name}_q{int(q * 100):02d}"] = float(np.quantile(values, q)) if len(values) else 0.0
        out[f"kp_{name}_mean"] = float(values.mean()) if len(values) else 0.0
        out[f"kp_{name}_std"] = float(values.std()) if len(values) else 0.0
    for name, values in [("weight", w), ("profit", p)]:
        total = values.sum()
        out[f"kp_{name}_gini"] = float(np.abs(values[:, None] - values[None, :]).sum() / (2 * n * total)) if total > 0 else 0.0
    out["kp_weight_profit_pearson"] = correlation(w, p)
    out["kp_weight_profit_spearman"] = correlation(midranks(w), midranks(p))
    out["kp_weight_profit_correlation_undefined"] = float(np.ptp(w) == 0 or np.ptp(p) == 0)
    out["kp_pair_fit_ratio"] = float(np.mean(pair_slack >= -1e-12)) if len(i) else 0.0
    out["kp_pair_exact_fit_ratio"] = float(np.mean(np.abs(pair_slack) <= 1e-12)) if len(i) else 0.0
    # This describes a pair of inputs; it does not justify removing an item from a 0-1 knapsack.
    better_in_both = ((w[i] < w[j]) & (p[i] > p[j])) | ((w[i] > w[j]) & (p[i] < p[j]))
    out["kp_pair_lower_weight_higher_profit_ratio"] = float(np.mean(better_in_both)) if len(i) else 0.0
    out["kp_identical_item_pair_ratio"] = float(np.mean((w[i] == w[j]) & (p[i] == p[j]))) if len(i) else 0.0
    out["kp_oversize_item_ratio"] = float(np.mean(r > 1 + 1e-12))
    out["kp_zero_profit_item_ratio"] = float(np.mean(p == 0))
    return out


def incidence_matrix(value):
    if isinstance(value, dict) and '__incidence_matrix__' in value:
        return (matrix(value['__incidence_matrix__']) != 0).astype(float)
    groups = list(value.values()) if isinstance(value, dict) else list(value)
    sets = [set(group.keys() if isinstance(group, dict) else group) for group in groups]
    universe = sorted(set.union(*sets), key=repr)
    return np.array([[item in group for item in universe] for group in sets], dtype=float)


def incidence_graph_features(value):
    a = incidence_matrix(value)
    rows, columns = a.shape
    row_degree, column_degree = a.sum(1), a.sum(0)
    relative_degree = np.r_[row_degree/columns, column_degree/rows]
    jaccard, identical, containment = [], [], []
    for side in [a, a.T]:
        degree = side.sum(1)
        intersection = side @ side.T
        i, j = np.triu_indices(len(side), 1)
        overlap = intersection[i,j]
        union = degree[i]+degree[j]-overlap
        # Two empty neighborhoods contribute zero similarity, not an invented edge.
        jaccard.extend(np.divide(overlap, union, out=np.zeros_like(overlap), where=union>0))
        identical.extend((union>0) & (overlap==degree[i]) & (overlap==degree[j]))
        containment.extend((overlap==np.minimum(degree[i],degree[j])) &
                           (np.minimum(degree[i],degree[j])>0) & (degree[i]!=degree[j]))
    out = {}
    for name, values in [('relative_degree', relative_degree), ('neighbor_jaccard', jaccard)]:
        values = np.asarray(values, dtype=float)
        for q in [0,.1,.25,.5,.75,.9,1]:
            out[f'graph_{name}_q{int(q*100):02d}'] = float(np.quantile(values,q)) if len(values) else 0.0
        out[f'graph_{name}_mean'] = float(values.mean()) if len(values) else 0.0
        out[f'graph_{name}_std'] = float(values.std()) if len(values) else 0.0
    out['graph_identical_nonempty_neighborhood_pair_ratio'] = float(np.mean(identical)) if identical else 0.0
    out['graph_strict_nonempty_containment_pair_ratio'] = float(np.mean(containment)) if containment else 0.0
    out['graph_isolated_node_ratio'] = float(np.mean(np.r_[row_degree,column_degree]==0))
    out['graph_leaf_node_ratio'] = float(np.mean(np.r_[row_degree,column_degree]==1))
    out['graph_leaf_incident_edge_ratio'] = float(np.sum(a*((row_degree[:,None]==1)|(column_degree[None,:]==1)))/a.sum())
    scale = np.sqrt(row_degree[:,None]*column_degree[None,:])
    normalized = np.divide(a, scale, out=np.zeros_like(a), where=scale>0)
    singular = np.linalg.svd(normalized, compute_uv=False)
    out['graph_normalized_second_singular_value'] = float(singular[1]) if len(singular)>1 else 0.0
    out['graph_normalized_spectral_gap'] = float(singular[0] - (singular[1] if len(singular)>1 else 0.0))
    energy = singular**2
    share = energy[energy>0]/energy.sum()
    out['graph_normalized_spectral_effective_rank'] = float(np.exp(-np.sum(share*np.log(share)))/min(rows,columns))
    out['graph_normalized_spectral_energy_mean'] = float(energy.mean())
    neighbors = [list(np.flatnonzero(row)+rows) for row in a]
    neighbors += [list(np.flatnonzero(column)) for column in a.T]
    unseen = set(range(rows+columns))
    components = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in neighbors[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(size)
    out['graph_component_count'] = float(len(components))
    out['graph_largest_component_node_ratio'] = max(components)/(rows+columns)
    return out


def question_arc_triples(question):
    return {(int(i), int(j)): float(c) for i, j, c in re.findall(r"From node_(\d+) to node_(\d+): ([\d.]+) units?\b", question)}


def input_arcs(data, question=None):
    nodes, value = data["nodes"], data["arc_cost"]
    assert len(nodes) == len(set(nodes))
    n = len(nodes)
    if isinstance(value, dict):
        if all(isinstance(k, tuple) and len(k) == 2 for k in value):
            arcs = dict(value)
        else:
            assert all(isinstance(row, dict) for row in value.values())
            arcs = {(i, j): cost for i, row in value.items() for j, cost in row.items()}
        form = "mapping"
    elif all(isinstance(row, tuple) and len(row) == 3 for row in value):
        arcs = {(i, j): cost for i, j, cost in value}
        assert len(arcs) == len(value)
        form = "arc_triples"
    elif np.asarray(value).shape == (n, n):
        arcs = {(i, j): value[r][c] for r, i in enumerate(nodes) for c, j in enumerate(nodes)}
        form = "dense_matrix"
    else:
        assert np.asarray(value).shape == (n, n - 1) and question is not None
        blocks = re.findall(r"- \*\*From Hub (\d+)\*\*:\s*(.*?)(?=\n- \*\*From Hub|\n### Constraints:)", question, flags=re.S)
        arcs = {}
        assert len(blocks) == n
        for row_index, (source, block) in enumerate(blocks):
            assert int(source) == nodes[row_index]
            entries = [(int(j), float(c)) for j, c in re.findall(r"To Hub (\d+): ([\d.]+) units", block)]
            assert [j for j, _ in entries] == [j for j in nodes if j != int(source)]
            np.testing.assert_array_equal([c for _, c in entries], value[row_index])
            arcs.update({(int(source), j): c for j, c in entries})
        form = "question_verified_compressed_rows"
    assert arcs and all(i in nodes and j in nodes and np.isfinite(cost) for (i, j), cost in arcs.items())
    return arcs, form


def seven_stats(name, values):
    values = np.asarray(values, dtype=float)
    out = {f"sp_{name}_q{int(q*100):02d}": float(np.quantile(values, q)) if len(values) else 0.0
           for q in [0., .25, .5, .75, 1.]}
    out[f"sp_{name}_mean"] = float(values.mean()) if len(values) else 0.0
    out[f"sp_{name}_std"] = float(values.std()) if len(values) else 0.0
    return out


def shortest_path_features(nodes, arcs):
    n = len(nodes)
    assert n >= 2
    pos = {node: i for i, node in enumerate(nodes)}
    cost = np.zeros((n, n))
    present = np.zeros((n, n), dtype=bool)
    for (i, j), value in arcs.items():
        assert i in pos and j in pos and np.isfinite(value)
        if i != j:
            cost[pos[i], pos[j]] = value
            present[pos[i], pos[j]] = True
    assert present.any()
    scale = float(np.mean(np.abs(cost[present])))
    relative = cost / scale if scale > 0 else np.zeros_like(cost)
    outgoing, incoming, out_gap, in_gap = [], [], [], []
    for i in range(n):
        out = np.sort(relative[i, present[i]])
        inc = np.sort(relative[present[:, i], i])
        outgoing.append(float(out.mean()) if len(out) else 0.)
        incoming.append(float(inc.mean()) if len(inc) else 0.)
        if len(out) >= 2:
            out_gap.append(out[1] - out[0])
        if len(inc) >= 2:
            in_gap.append(inc[1] - inc[0])
    i, j = np.where(np.triu(present & present.T, 1))
    difference = np.abs(relative[i, j] - relative[j, i])
    out = {}
    for name, values in [("outgoing_mean", outgoing), ("incoming_mean", incoming),
                         ("mean_cost_imbalance", np.abs(np.asarray(outgoing) - incoming)),
                         ("outgoing_minimum_gap", out_gap), ("incoming_minimum_gap", in_gap),
                         ("reciprocal_cost_difference", difference)]:
        out.update(seven_stats(name, values))
    out["sp_arc_density"] = float(present.sum() / (n * (n - 1)))
    out["sp_reciprocal_arc_fraction"] = float(np.sum(present & present.T) / present.sum())
    out["sp_no_outgoing_arc_node_fraction"] = float(np.mean(present.sum(1) == 0))
    out["sp_no_incoming_arc_node_fraction"] = float(np.mean(present.sum(0) == 0))
    out["sp_two_outgoing_arc_node_fraction"] = float(np.mean(present.sum(1) >= 2))
    out["sp_two_incoming_arc_node_fraction"] = float(np.mean(present.sum(0) >= 2))
    out["sp_all_listed_costs_zero"] = float(scale == 0)
    return out


def product_mix_features(data):
    result = {"item_count": float(len(data["profit"])), "resource_capacity": data["capacity"]}
    for name in ["profit", "resource_usage", "production_limit"]:
        result.update(stats(name, data[name]))
    usage = np.asarray(data["resource_usage"])
    limit = np.asarray(data["production_limit"])
    result.update(stats("unit_resource_share", usage / data["capacity"]))
    result.update(stats("maximum_resource_share", usage * limit / data["capacity"]))
    result["total_maximum_resource_share"] = float(np.sum(usage * limit) / data["capacity"])
    return result


def stated_milp_features(data):
    a = np.asarray(data["coefficients"], dtype=float)
    b = np.asarray(data["rhs"], dtype=float)
    c = np.asarray(data["objective"], dtype=float)
    lower = np.asarray(data["lower_bounds"], dtype=float)
    upper = data["upper_bounds"]
    relations = data["relation_senses"]
    assert a.ndim == 2 and min(a.shape) > 0
    assert b.shape == (a.shape[0],) and c.shape == lower.shape == (a.shape[1],)
    assert len(upper) == len(c) and len(relations) == len(b)
    assert set(relations) <= {"<=", "=="} and data["objective_sense"] in {"min", "max"}
    assert all(np.isfinite(values).all() for values in [a, b, c, lower])
    finite_upper = [value for value in upper if value is not None]
    widths = [value - lower[i] for i, value in enumerate(upper) if value is not None]
    assert np.isfinite(finite_upper).all() and all(value >= 0 for value in widths)
    out = matrix_features("input_coefficients", a)
    for name, values in [("input_rhs", b), ("input_objective", c), ("input_lower_bound", lower)]:
        out.update(stats(name, values))
    for name, values in [("stated_upper_bound", finite_upper), ("stated_bound_width", widths)]:
        # An empty set has zero aggregate descriptors, including count=0.
        # No absent upper bound is assigned a numerical value.
        out.update(stats(name, values) if values else {key: 0.0 for key in stats(name, [0])})
    out["input_equality_fraction"] = float(np.mean(np.asarray(relations) == "=="))
    out["input_objective_is_minimization"] = float(data["objective_sense"] == "min")
    return out


