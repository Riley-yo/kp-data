import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def apply_projection(frame, fitted_dir, output_dir, spec):
    from projection_plot import read_indexed, draw_projection
    from runtime import transform_features, write_json
    from space_coverage import shape_diagnostics

    fitted_dir, output_dir = Path(fitted_dir), Path(output_dir)
    names = ['summary.json', 'pilot.json', 'prelim.json', 'projection_matrix.csv',
             'selected_features.csv', 'coordinates.csv']
    hashes = {name: hashlib.sha256((fitted_dir / name).read_bytes()).hexdigest() for name in names}
    summary, pilot, prelim = [json.loads((fitted_dir / name).read_text(encoding='utf-8'))
                              for name in names[:3]]
    if summary.get('problem') != spec['problem'] or summary.get('variant') != spec['primary_variant']:
        raise ValueError('Frozen ISA problem and representation must match the entry')
    if not summary.get('projection_fitted') or pilot.get('circle_used_in_feature_selection') is not False or pilot.get('circle_used_in_projection') is not False:
        raise ValueError('A fitted paper ISA with unmodified projection is required')
    a, fit_x, fit_z = [read_indexed(fitted_dir / name) for name in names[3:]]
    if list(a.columns) != ['z1', 'z2'] or list(a.index) != pilot['selected_features']:
        raise ValueError('Projection matrix and declared selected features disagree')
    if not fit_x.index.equals(fit_z.index) or list(fit_x.columns) != list(a.index):
        raise ValueError('Frozen training coordinates and selected features are misaligned')
    if not all(np.isfinite(table.to_numpy(float)).all() for table in (a, fit_x, fit_z)):
        raise ValueError('Frozen projection artifacts must be finite')
    np.testing.assert_allclose(fit_x.to_numpy() @ a.to_numpy(), fit_z.to_numpy(), rtol=0, atol=1e-10)
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError('All incoming instances need unique IDs and feature names')
    missing = set(a.index) - set(frame.columns)
    if missing:
        raise ValueError(f'Recover required intrinsic features first: {sorted(missing)}')
    selected_prelim = {'features': {name: prelim['features'][name] for name in a.index}}
    x = transform_features(frame, selected_prelim)
    z = pd.DataFrame(x.to_numpy() @ a.to_numpy(), index=frame.index, columns=['z1', 'z2'])
    frame.to_csv(output_dir / 'features.csv', index_label='instance_id', float_format='%.17g')
    x.to_csv(output_dir / 'selected_features.csv', index_label='instance_id', float_format='%.17g')
    a.to_csv(output_dir / 'projection_matrix.csv', float_format='%.17g')
    z.to_csv(output_dir / 'coordinates.csv', index_label='instance_id', float_format='%.17g')
    write_json(output_dir / 'prelim.json', selected_prelim)
    metadata = {'method': 'frozen_paper_ISA_application', 'fitted_directory': str(fitted_dir.resolve()),
                'fitted_artifact_sha256': hashes, 'problem': spec['problem'],
                'selected_features': list(a.index), 'feature_records': len(frame),
                'all_input_rows_included': True, 'refitted': False, 'performance_required_for_application': False,
                'performance_features_used': False, 'coordinates': 'frozen PRELIM(X_selected) @ frozen A',
                'coordinates_modified': False, 'circle_used_in_projection': False,
                'note': 'All rows use the same feature definitions and frozen map. Missing parameters and out-of-domain values fail explicitly.'}
    write_json(output_dir / 'projection_application.json', metadata)
    try:
        shape = shape_diagnostics(z.to_numpy())
    except ValueError as error:
        shape = {'available': False, 'reason': str(error)}
    write_json(output_dir / 'shape_diagnostics.json', shape)
    draw_projection(output_dir, title=spec['problem'] + ' / frozen paper ISA')
    return {'status': 'paper_projection_applied_to_all_input_rows', 'projection_fitted': False,
            'projection_applied': True, 'all_feature_rows_included': True,
            'selected_feature_count': len(a), 'shape_diagnostics': shape,
            'performance_partition_generated': False}
