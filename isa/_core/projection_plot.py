import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_indexed(path):
    first = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(path, index_col=0, dtype={first: str}, float_precision='round_trip')
    if not frame.index.is_unique or frame.index.hasnans or not frame.columns.is_unique:
        raise ValueError('IDs and columns must be unique and nonempty')
    return frame


def labeled_points(coordinates, origins):
    z, audit = read_indexed(coordinates), read_indexed(origins)
    if list(z.columns) != ['z1', 'z2'] or z.empty or not np.isfinite(z.to_numpy(float)).all():
        raise ValueError('Coordinates require finite z1,z2 columns and at least one row')
    if 'origin' not in audit or set(z.index) != set(audit.index):
        raise ValueError('Every coordinate must have exactly one origin; extra or missing IDs are rejected')
    labels = audit.loc[z.index, 'origin']
    if not labels.isin(['sage', 'sage_original', 'synthetic']).all():
        raise ValueError('Origin must be sage, sage_original or synthetic; unknown origins cannot be inferred')
    z['origin'] = labels.replace({'sage_original': 'sage'})
    return z


def draw_projection(folder, coordinates=None, origins=None, title='ISA projection'):
    folder = Path(folder)
    points = labeled_points(coordinates or folder / 'coordinates.csv', origins or folder / 'instance_audit.csv')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'font.size': 10, 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    counts = {}
    for origin, name, color, marker in [('sage', 'SAGE', '#0072B2', 'o'),
                                        ('synthetic', 'Synthetic', '#E69F00', '^')]:
        subset = points.loc[points.origin == origin]
        counts[origin] = len(subset)
        ax.scatter(subset.z1, subset.z2, s=15, alpha=.7, color=color, marker=marker,
                   linewidths=0, label=f'{name} (n={len(subset)})')
    assert sum(counts.values()) == len(points)
    ax.set(xlabel='ISA coordinate 1', ylabel='ISA coordinate 2', title=f'{title} | n={len(points)}')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'projection.png', dpi=300)
    fig.savefig(folder / 'projection.pdf')
    plt.close(fig)
    points.to_csv(folder / 'plot_points.csv', index_label='instance_id', float_format='%.17g')
    (folder / 'figure_manifest.json').write_text(json.dumps({
        'plotted_points': len(points), 'counts': counts, 'omitted_points': 0,
        'colors': {'sage': '#0072B2', 'synthetic': '#E69F00'},
        'coordinates_modified': False, 'equal_axis_scale': True,
        'note': 'Counts include overlapping points. An absent source has n=0 in the legend.'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
