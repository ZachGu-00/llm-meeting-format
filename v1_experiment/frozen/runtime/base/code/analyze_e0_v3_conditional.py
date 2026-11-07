"""Apply the documented conditional E0 quality rule to a closed original analysis.

No inference, bootstrap rerun, effect-based selection or raw-file mutation.
Original analyses remain authoritative for measured values and are hash-bound.
"""
import argparse
import copy
import json
from pathlib import Path
import analyze_e0_v3 as original
from run_e0_v3_taskpilot import validate_existing_analysis
from inference_v3 import sha

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / 'provenance/E0_conditional_mass_amendment_20261031.txt'
SCHEMA = 'voices_v3_E0_conditional_v1'
NORMAL = {'option_mass_median': .70, 'first_token_format': .90, 'R2_concordance': .85}
CONDITIONAL = {'option_mass_median': .50, 'first_token_format': .95, 'R2_concordance': .90}


def apply_policy(source):
    """Pure policy transform; all measured values/effect diagnostics stay unchanged."""
    if source.get('schema') != 'voices_v3_E0_v1':
        raise ValueError('Conditional policy requires an original V3 E0 analysis')
    result = copy.deepcopy(source)
    branches = {}
    for name, thresholds in [('normal', NORMAL), ('conditional', CONDITIONAL)]:
        gates = {key: original.flag(source['operational_gates'][key].get('value'), value)
                 for key, value in thresholds.items()}
        branches[name] = {'gates': gates, 'status': original.ref.overall(gates)}
    selected = next((name for name in ('normal', 'conditional') if branches[name]['status'] == 'pass'), None)
    status = 'pass' if selected else 'incomplete' if any(b['status'] == 'incomplete' for b in branches.values()) else 'failed'
    result['schema'] = SCHEMA
    result['operational_policy'] = {**branches, 'union_status': status,
        'selected_branch': selected, 'conditional_only': selected == 'conditional',
        'calibration_quality_used_for_admission': False,
        'adaptive_pre_E1_development_amendment': True,
        'scientific_effects_used_for_remedy_selection': False}
    ds = source['applicability_diagnostics']['weight_directions']
    gaps = source['strict_original_gates']['R0_R1_primary_agreement'].get('value')
    if not isinstance(gaps, dict):
        gaps = None
    active = branches[selected or 'normal']['gates']
    result['qualification'] = original.admission(active,
        source['qualification']['R1_candidate_scoring_valid'], ds, gaps, source['n_items'])
    result['qualification'].update(conditional_option_mass_admission=selected == 'conditional',
        operational_admission_branch=selected, prior_relaxed_R0_operational=branches['normal']['status'])
    if source.get('bootstrap_draws') != 10000:
        result['qualification']['final_qualification'] = 'nonstandard_draw_count_diagnostic_only'
    result['original_qualification_before_conditional_policy'] = copy.deepcopy(source['qualification'])
    result['conditional_option_mass_caveat'] = (
        'Conditional-only admission needs explicit disclosure and normally-admitted-subset sensitivity. '
        'This operational rule does not prove calibrated behavioral probabilities. '
        'Low-q rows are retained and reported; no E1/E4 effect determines admission.')
    return result


def from_closed(source_path, input_path, result_path, lockpath, amendment_path=AMENDMENT):
    source = validate_existing_analysis(source_path, input_path, result_path, lockpath)
    result = apply_policy(source)
    result['conditional_policy_provenance'] = {
        'original_analysis': str(Path(source_path).resolve()), 'original_analysis_sha256': sha(source_path),
        'amendment_file': str(Path(amendment_path).resolve()), 'amendment_sha256': sha(amendment_path),
        'policy_code_file': str(Path(__file__).resolve()), 'policy_code_sha256': sha(__file__),
        'measurements_reused_without_recomputation': True}
    result['file_sha256'].update({str(Path(p).resolve()): sha(p)
        for p in (source_path, amendment_path, __file__)})
    return result


def ensure_analysis(source_path, input_path, result_path, lockpath, output_path, amendment_path=AMENDMENT):
    """Create a separate file once, or verify a byte-independent identical reuse."""
    result = from_closed(source_path, input_path, result_path, lockpath, amendment_path)
    output_path = Path(output_path)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding='utf-8'))
        if existing != result:
            raise ValueError('Existing conditional analysis has different bindings or policy; preserve it')
        return existing
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-analysis', 'input', 'result', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--amendment-file', type=Path, default=AMENDMENT)
    args = parser.parse_args()
    result = ensure_analysis(args.source_analysis, args.input, args.result, args.calibration,
                             args.output, args.amendment_file)
    print(json.dumps({'n_items': result['n_items'], 'qualification': result['qualification'],
                      'operational_policy': result['operational_policy']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
