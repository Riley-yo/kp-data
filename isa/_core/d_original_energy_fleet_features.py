import numpy as np

from domain import numbers, stats, unpack


def calculate_features(problem, parameters):
    data = unpack(parameters)
    if problem == 'FLEET':
        result = {}
        for key in ['cost', 'capacity', 'demand', 'vehicle_upper_bounds']:
            result.update(stats(key, numbers(data[key])))
        capacity = np.asarray(data['capacity'], float)
        upper = np.asarray(data['vehicle_upper_bounds'], float)
        result.update(stats('available_transport_capacity', capacity * upper))
        result['demand_available_capacity_ratio'] = float(data['demand'] / (capacity @ upper))
        result['integer_vehicle_domain'] = 1.
        return result
    result = {}
    for key in ['unit_cost', 'minimum_output', 'output_capacity', 'supply_lower_bound', 'supply_upper_bound']:
        result.update(stats(key, numbers(data[key])))
    lo, hi = np.asarray(data['minimum_output'], float), np.asarray(data['output_capacity'], float)
    result.update(stats('adjustable_output', hi - lo))
    rows = np.asarray(data['coupling_coefficients'], float).reshape(-1, len(lo))
    for prefix, values in [('physical_coupling_coefficient', rows.ravel()),
                           ('physical_coupling_rhs', data['coupling_rhs'])]:
        result.update(stats(prefix, values) if len(values) else {key: 0. for key in stats(prefix, [0.])})
    result['physical_coupling_count'] = float(len(rows))
    result['physical_coupling_density'] = float(np.mean(rows != 0)) if rows.size else 0.
    result['demand_total_capacity_ratio'] = float(data['supply_lower_bound'] / hi.sum())
    result['minimum_total_capacity_ratio'] = float(lo.sum() / hi.sum())
    result['supply_interval_capacity_ratio'] = float((data['supply_upper_bound'] - data['supply_lower_bound']) / hi.sum())
    result['continuous_output_domain'] = 1.
    return result


def canonical(problem, parameters):
    data = unpack(parameters)
    if problem == 'FLEET':
        n = len(data['cost'])
        return {'objective': data['cost'], 'objective_sense': 'min', 'coefficients': [data['capacity']],
                'rhs': [data['demand']], 'relation_senses': ['>='], 'lower_bounds': [0] * n,
                'upper_bounds': data['vehicle_upper_bounds'], 'variable_types': ['I'] * n}
    n = len(data['unit_cost'])
    return {'objective': data['unit_cost'], 'objective_sense': 'min',
            'coefficients': [[1.] * n, [1.] * n] + data['coupling_coefficients'],
            'rhs': [data['supply_lower_bound'], data['supply_upper_bound']] + data['coupling_rhs'],
            'relation_senses': ['>=', '<='] + data['coupling_senses'],
            'lower_bounds': data['minimum_output'], 'upper_bounds': data['output_capacity'],
            'variable_types': ['C'] * n}
