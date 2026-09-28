import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError


AUTHOR_SOURCE = (
    'https://github.com/andremun/InstanceSpace/blob/'
    'e7fa8002d8979576ce53115f745ce6f9f3331df0/CLOISTER.m#L48-L61'
)


def hull_boundary(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError('Projected corner set has fewer than three non-collinear points')
    return points[ConvexHull(points).vertices]


def cloister_boundary_with_evidence(selected_features, projection, threshold=0.7):
    x = selected_features.to_numpy(dtype=float)
    a = np.asarray(projection, dtype=float)
    if x.ndim != 2 or a.shape != (x.shape[1], 2):
        raise ValueError('Expected feature matrix and a matching two-dimensional projection')
    if not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(np.ptp(x, axis=0) == 0):
        raise ValueError('CLOISTER requires finite nonconstant transformed feature columns')
    lower, upper = x.min(axis=0), x.max(axis=0)
    correlation = np.corrcoef(x, rowvar=False)
    all_vertices, retained_vertices = [], []
    for bits in itertools.product([0, 1], repeat=x.shape[1]):
        projected = np.where(bits, upper, lower) @ a
        all_vertices.append(projected)
        possible = all(not ((correlation[i, j] > threshold and bits[i] != bits[j])
                            or (correlation[i, j] < -threshold and bits[i] == bits[j]))
                       for i in range(len(bits)) for j in range(i))
        if possible:
            retained_vertices.append(projected)
    # The author's full-box hull is computed outside the fallback try block.
    full_box = hull_boundary(all_vertices)
    evidence = {
        'boundary_method': 'correlation_filtered_cloister',
        'correlation_threshold': threshold,
        'pvalue_filter_applied': False,
        'correlation_rule': 'Signed positive/negative endpoint rule from 2BP Section 4.1',
        'unpruned_corner_count': len(all_vertices),
        'retained_corner_count': len(retained_vertices),
        'correlation_filtered_failure': None,
        'fallback_used': False,
        'fallback_source': AUTHOR_SOURCE,
        'boundary_is_an_outer_estimate_not_an_instance_feasibility_certificate': True,
        'circle_template_used': False,
    }
    try:
        boundary = hull_boundary(retained_vertices)
    except (ValueError, QhullError) as error:
        boundary = full_box
        evidence.update(boundary_method='unpruned_box_author_fallback', fallback_used=True,
                        correlation_filtered_failure=str(error))
    return boundary, evidence
