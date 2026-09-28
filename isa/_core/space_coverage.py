import itertools

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def shape_diagnostics(coordinates):
    z = np.asarray(coordinates, dtype=float)
    if len(z) < 3 or np.linalg.matrix_rank(z - z.mean(axis=0)) < 2:
        raise ValueError('Shape diagnostics require a rank-two point cloud')
    hull = ConvexHull(z)
    polygon = z[hull.vertices]
    center = polygon.mean(axis=0)
    next_vertices = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * next_vertices[:, 1] - polygon[:, 1] * next_vertices[:, 0]
    signed_area = cross.sum() / 2
    if signed_area:
        center = ((polygon + next_vertices) * cross[:, None]).sum(axis=0) / (6 * signed_area)
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    radial = []
    # Ray intersections with the observed convex hull; no point coordinates are changed.
    for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        denominators = hull.equations[:, :2] @ direction
        positive = denominators > 1e-12
        distances = -(hull.equations[:, :2] @ center + hull.equations[:, 2])[positive] / denominators[positive]
        radial.append(float(distances.min()))
    radius = np.linalg.norm(z - center, axis=1)
    return {'n': len(z), 'hull_circularity': float(4 * np.pi * hull.volume / hull.area ** 2),
            'covariance_axis_ratio': float(np.sqrt(eig[0] / eig[1])),
            'hull_radius_cv': float(np.std(radial) / np.mean(radial)),
            'inner_half_radius_fraction': float(np.mean(radius <= radius.max() / 2)),
            'note': 'Diagnostics only. Circle: first two approach 1, hull radius CV approaches 0. '
                    'Convex hull scores alone do not establish a uniformly filled disk.',
            'used_as_optimization_objective': False}




def coverage_targets(coordinates, boundary, grid_size=15, count=12):
    z, boundary = np.asarray(coordinates), np.asarray(boundary)
    hull = ConvexHull(boundary)
    lo, hi = boundary.min(axis=0), boundary.max(axis=0)
    grid = np.array(list(itertools.product(np.linspace(lo[0], hi[0], grid_size),
                                         np.linspace(lo[1], hi[1], grid_size))))
    inside = np.all(grid @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-10, axis=1)
    grid = grid[inside]
    distances = cKDTree(z).query(grid)[0]
    order = np.argsort(-distances, kind='stable')
    spacing = np.linalg.norm(hi - lo) / grid_size
    chosen = []
    for i in order:
        if all(np.linalg.norm(grid[i] - grid[j]) >= spacing for j in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return grid[chosen], distances[chosen]
