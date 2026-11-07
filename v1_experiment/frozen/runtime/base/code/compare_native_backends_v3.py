"""Describe the explicit native-kernel migration at fixed batch 16.

Only the recorded loader/backend metadata and renewed calibration SHA may change.
This never rewrites reference scores or applies an automatic numerical gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_numeric_batches_v3 import read_closed, read_json, sha, v2


ALLOWED_DIFFERENCES = {
    'calibration_file_sha256':
        'The same controls are recalibrated under the candidate native backend.',
    'engine.native_loader_sha256':
        'The native loader now records optional native-kernel dispatch.',
    'engine.parent_loader.code_sha256':
        'The native loader now records optional native-kernel dispatch.',
    'engine.parent_loader.optional_kernel_distributions':
        'Explicit optional-kernel distribution versions for this migration.',
    'engine.parent_loader.native_kernel_dispatch':
        'Explicit selected native callables for this migration; not a profiler trace.',
    'engine.parent_loader.native_kernel_dispatch_scope':
        'Defines the scope of the newly recorded native-callable metadata.',
    'engine.parent_loader.hub_kernels_requested':
        'Records whether the loader requested Hub kernels.',
}
_MISSING = object()


def required(mapping, path):
    value = mapping
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Required identity field is absent: ' + path)
        value = value[key]
    if value is None:
        raise ValueError('Required identity field is null: ' + path)
    return value


def validate_identities(reference, candidate):
    """Return every permitted difference, rejecting any other identity change."""
    mandatory = (
        'batch_size', 'calibration_file_sha256', 'input_file_sha256',
        'input_record_ids_sha256', 'input_rows_sha256', 'n_input',
        'option_ids.A', 'option_ids.B', 'readout', 'mode', 'schema',
        'format_record_ids', 'r2_record_ids', 'engine.code_sha256',
        'engine.native_loader_sha256', 'engine.remedy', 'engine.model_kind',
        'engine.precision', 'engine.template_policy',
        'engine.parent_loader.code_sha256', 'engine.parent_loader.repo_id',
        'engine.parent_loader.revision', 'engine.parent_loader.dtype',
        'engine.parent_loader.template_sha256', 'engine.parent_loader.max_length',
    )
    for run in (reference, candidate):
        for path in mandatory:
            required(run, path)
        if run['batch_size'] != 16 or run['readout'] != 'R0_R1':
            raise ValueError('Native-backend pilot requires batch 16 and R0_R1 in both runs')
    for name in ('optional_kernel_distributions', 'native_kernel_dispatch',
                 'native_kernel_dispatch_scope', 'hub_kernels_requested'):
        required(candidate, 'engine.parent_loader.' + name)

    differences = []

    def walk(a, b, path):
        if a is not _MISSING and b is not _MISSING and a == b:
            return
        if path in ALLOWED_DIFFERENCES:
            differences.append({
                'path': path,
                'reference_present': a is not _MISSING,
                'candidate_present': b is not _MISSING,
                'reference': None if a is _MISSING else a,
                'candidate': None if b is _MISSING else b,
                'reason': ALLOWED_DIFFERENCES[path],
            })
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, _MISSING), b.get(key, _MISSING),
                     path + '.' + key if path else key)
            return
        raise ValueError('Unapproved native-backend identity difference: ' + path)

    walk(reference, candidate, '')
    return differences


def validate_rows(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError('Native-backend pilot row count changed')
    if [v2.row_id(r) for r in reference] != [v2.row_id(r) for r in candidate]:
        raise ValueError('Native-backend pilot record IDs or order changed')
    fields = (
        'actual_prompt_sha256', 'actual_prompt_ids_sha256',
        'R1_actual_prompt_sha256', 'R1_actual_prompt_ids_sha256',
        'batch_size', 'actual_batch_size_R0',
    )
    for x, y in zip(reference, candidate):
        for key in fields:
            if key not in x or key not in y or x[key] is None or y[key] is None:
                raise ValueError('Missing actual row identity: ' + key)
            if x[key] != y[key]:
                raise ValueError('Actual row identity changed: ' + key + ' at ' + str(v2.row_id(x)))
        for candidate_key in ('R1_candidate_A', 'R1_candidate_B'):
            for key in ('text', 'token_ids', 'scored_leading_separator', 'actual_batch_size'):
                if key not in x.get(candidate_key, {}) or key not in y.get(candidate_key, {}):
                    raise ValueError('Missing R1 candidate identity: ' + candidate_key + '.' + key)
                if x[candidate_key][key] != y[candidate_key][key]:
                    raise ValueError('R1 candidate identity changed: ' + candidate_key + '.' + key)
        if x['batch_size'] != 16:
            raise ValueError('Raw pilot row does not record fixed batch 16')


def compare(inputs, reference, candidate):
    runs = [read_json(str(p) + '.run.json') for p in (reference, candidate)]
    differences = validate_identities(*runs)
    source = v2.read_rows(inputs)
    a, b = [list(read_closed(inputs, p)) for p in (reference, candidate)]
    validate_rows(a, b)
    result = {
        'comparison': 'native_backend_migration_fixed_batch16',
        'n_rows': len(a), 'batch_sizes': [r['batch_size'] for r in runs],
        'same_inference_model_template_options_and_actual_prompt_tokens': True,
        'run_identities': {'reference': runs[0], 'candidate': runs[1]},
        'identity_differences': differences,
        'permitted_identity_differences': ALLOWED_DIFFERENCES,
        'comparison_is_descriptive': True,
        'no_v2_point05_automatic_rejection': True,
        'readouts': {}, 'quality': [],
    }
    for rows in (a, b):
        checks = v2.generated_checks({'physical_rows': rows})
        result['quality'].append({
            'q_median': float(np.median([r['q'] for r in rows])),
            'q_below_point5_count': sum(r['q'] < .5 for r in rows),
            'generated_checks': checks,
        })
    for readout in ('R0', 'R1'):
        panels = [v2.prepare(source, rows, readout) for rows in (a, b)]
        if any(p['status'] != 'complete' for p in panels):
            raise ValueError('Incomplete numerical pilot')
        comps = [v2.components(p) for p in panels]
        points = [v2.primary_point(c[2], c[3], c[4]) for c in comps]
        denominators = [float(np.abs(c[2]).mean()) for c in comps]
        x = np.array([r[readout + '_y_AB'] for r in a])
        y = np.array([r[readout + '_y_AB'] for r in b])
        diff = np.abs(x - y)
        scale = denominators[0]
        metrics = ('w_same', 'w_diff', 'ASI')
        changes = {k: abs(points[1][k] - points[0][k])
                   if all(p[k] is not None for p in points) else None for k in metrics}
        result['readouts'][readout] = {
            'reference_mean_absolute_singleton': scale,
            'candidate_mean_absolute_singleton': denominators[1],
            'mean_absolute_y_difference': float(diff.mean()),
            'mean_absolute_y_difference_over_reference_abs_singleton':
                float(diff.mean() / scale) if scale > 0 else None,
            'max_absolute_y_difference': float(diff.max()),
            'p95_absolute_y_difference': float(np.percentile(diff, 95)),
            'sign_agreement_including_equal_ties': float((np.sign(x) == np.sign(y)).mean()),
            'strict_same_nonzero_direction_fraction': float(((x * y) > 0).mean()),
            'reference_ties': int((x == 0).sum()), 'candidate_ties': int((y == 0).sum()),
            'reference_primary': {k: points[0][k] for k in metrics},
            'candidate_primary': {k: points[1][k] for k in metrics},
            'absolute_primary_changes': changes,
            'mean_absolute_label_slot_averaged_y_difference':
                float(np.abs(comps[0][1] - comps[1][1]).mean()),
        }
    summaries = [read_json(str(p) + '.summary.json') for p in (reference, candidate)]
    seconds = [s['seconds_this_invocation'] for s in summaries]
    result['runtime'] = {
        'seconds': seconds, 'reference_over_candidate_speedup': seconds[0] / seconds[1],
        'scope': 'measured score invocation, not download or model-load wall time unless engine includes it',
    }
    result['within_backend_R0_R1_primary_absolute_differences'] = [
        {k: abs(result['readouts']['R0'][which][k] - result['readouts']['R1'][which][k])
         if result['readouts']['R0'][which][k] is not None and
         result['readouts']['R1'][which][k] is not None else None
         for k in ('w_same', 'w_diff', 'ASI')}
        for which in ('reference_primary', 'candidate_primary')
    ]
    from analyze_v3 import __file__ as analyzer_path
    paths = [inputs, reference, candidate, __file__, v2.__file__, analyzer_path,
             Path(__file__).with_name('compare_numeric_batches_v3.py')]
    paths += [str(p) + suffix for p in (reference, candidate)
              for suffix in ('.run.json', '.summary.json')]
    result['file_sha256'] = {str(Path(p).resolve()): sha(p) for p in paths}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Refusing to overwrite: ' + args.output)
    result = compare(args.inputs, args.reference, args.candidate)
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({
        'batch_sizes': result['batch_sizes'], 'runtime': result['runtime'],
        'identity_difference_paths': [d['path'] for d in result['identity_differences']],
        'readouts': result['readouts'],
        'quality': [{'q': q['q_median'],
                     'format': q['generated_checks']['first_token_format']['value'],
                     'R2': q['generated_checks']['R2_concordance']['value']}
                    for q in result['quality']],
    }))
