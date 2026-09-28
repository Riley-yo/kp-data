import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

def calculate_features(parameters):
    from domain import numbers, stats, unpack
    data = unpack(parameters)
    result = {}
    for field in ['arc_cost', 'balance', 'capacity']:
        result.update(stats(field, numbers(data[field])))
    result.update(calculate_model_features(data))
    return result

def calculate_model_features(data):
    nodes = list(data['nodes'])
    arcs = data['capacity']
    declared = data.get('node_inflow_capacity', {})
    if not set(declared).issubset(nodes):
        raise ValueError('Declared node inflow capacity has unknown node')
    implied = np.array([sum(float(cap) for (i, j), cap in arcs.items() if j == node) for node in nodes])
    effective = np.array([min(implied[k], float(declared[node])) if node in declared else implied[k]
                          for k, node in enumerate(nodes)])
    if not np.isfinite(effective).all() or np.any(effective < 0):
        raise ValueError('Nonnegative finite node inflow bounds required')
    ratio = np.divide(effective, implied, out=np.ones_like(effective), where=implied > 0)
    features = {'node_inflow_explicit_fraction': len(declared) / max(len(nodes), 1),
                'node_inflow_binding_bound_fraction': float(np.mean(effective < implied)) if len(nodes) else 0.,
                'node_inflow_effective_total_ratio': float(effective.sum() / implied.sum()) if implied.sum() else 1.}
    for prefix, values in [('node_inflow_effective_capacity', effective), ('node_inflow_capacity_to_incoming_arc_capacity', ratio)]:
        for name, value in [('mean', np.mean(values)), ('std', np.std(values)), ('min', np.min(values)),
                            ('max', np.max(values)), ('median', np.median(values)),
                            ('q25', np.quantile(values, .25)), ('q75', np.quantile(values, .75))]:
            features[f'{prefix}_{name}'] = float(value)
    return features
