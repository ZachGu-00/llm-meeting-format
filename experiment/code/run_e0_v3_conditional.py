"""Symmetric E0 conditional-mass reassessment, preserving all previous evidence."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import analyze_e0_v3_conditional as conditional
from run_e0_v3_taskpilot import call, write_state
from inference_v3 import canonical_hash, sha

HERE = Path(__file__).resolve().parent
MODES = ('original', 't2', 'prefill')
SCHEMA = 'voices_v3_E0_conditional_driver_v1'
AMENDMENT = conditional.AMENDMENT


def archive_previous_state(statepath):
    path = Path(statepath)
    digest = sha(path)
    archive = path.parent / 'provenance' / ('e0_driver_state_before_conditional_' + digest + '.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if sha(archive) != digest:
            raise ValueError('Existing prior-state snapshot changed')
    else:
        with archive.open('xb') as stream:
            stream.write(path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    return archive, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model-dir', 'repo-id', 'revision', 'calibration-input', 'pilot-input', 'development-input', 'output-dir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--model-kind', choices=['auto', 'chat', 'base'], default='auto')
    parser.add_argument('--attn-implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default='eager')
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--stop-after', choices=['pilot', 'development'], default='development')
    parser.add_argument('--amendment-file', type=Path, default=AMENDMENT)
    parser.add_argument('--audit-only', action='store_true', help='Print reassessment scope without writing or executing models')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive; this driver does not tune batching')
    out = Path(args.output_dir); statepath = out / 'e0_driver_state.json'
    if not statepath.is_file():
        raise ValueError('Conditional reassessment requires an existing E0 driver state')
    previous = json.loads(statepath.read_text(encoding='utf-8'))
    if args.audit_only:
        print(json.dumps({'repo_id': args.repo_id, 'eligible': True, 'audit_only': True,
            'policy': 'all_models_symmetrically_fixed_original_t2_prefill',
            'previous_schema': previous.get('schema'), 'previous_remedy': previous.get('selected_remedy')})); return
    if not os.environ.get('STY'):
        raise RuntimeError('E0 driver and its GPU subprocesses must run in screen')
    if any(previous.get(key) != value for key, value in [('repo_id', args.repo_id), ('revision', args.revision),
        ('batch_size', args.batch_size), ('attn_implementation', args.attn_implementation)]):
        raise ValueError('Reassessment must retain prior model/revision/batch/attention identity')
    amendment_hash = sha(args.amendment_file)
    archive, old_hash = archive_previous_state(statepath)
    skips = previous.get('original_calibration_only_skips', []) or [a['remedy'] for a in previous.get('attempts', [])
        if a.get('decision') == 'advance_due_to_calibration_format_or_q']
    runbase = [sys.executable, str(HERE / 'inference_v3.py'), '--model-dir', args.model_dir,
        '--repo-id', args.repo_id, '--revision', args.revision, '--batch-size', str(args.batch_size),
        '--model-kind', args.model_kind, '--attn-implementation', args.attn_implementation,
        '--max-length', str(args.max_length)]
    state = {'schema': SCHEMA, 'repo_id': args.repo_id, 'revision': args.revision,
        'batch_size': args.batch_size, 'attn_implementation': args.attn_implementation,
        'selection_rule': 'fixed_original_t2_prefill_actual_pilot_then_first_full100_operational_union_pass',
        'amendment_reason': 'Pre-E1 adaptive E0: normal .70/.90/.85 OR conditional .50/.95/.90 actual-task q/format/R2; all initial models reassessed symmetrically.',
        'amendment_file': str(args.amendment_file.resolve()), 'amendment_sha256': amendment_hash,
        'previous_state_snapshot': str(archive.resolve()), 'previous_state_sha256': old_hash,
        'original_calibration_only_skips': skips,
        'supersedes_selection': {k: previous.get(k) for k in ('selected_remedy', 'selected_analysis', 'final_qualification')},
        'calibration_used_for_admission': False, 'calibration_screen_is_proxy_not_formal_format_diagnostic': True,
        'effect_directions_not_used_to_choose_template': True,
        'execution_protocol_sha256': sha(HERE.parent / 'execution_protocol.txt'),
        'driver_sha256': sha(__file__), 'conditional_analyzer_sha256': sha(conditional.__file__),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'attempts': [], 'status': 'running'}
    write_state(statepath, state)
    for remedy in MODES:
        folder = out / remedy; folder.mkdir(exist_ok=True)
        lockpath = folder / 'token_calibration.json'
        if not lockpath.exists():
            call(runbase + ['--remedy', remedy, '--mode', 'calibrate', '--input', args.calibration_input,
                '--output', str(lockpath)], folder / 'calibration.log')
        lock = json.loads(lockpath.read_text(encoding='utf-8'))
        if (lock['input_file_sha256'] != sha(args.calibration_input) or
            lock['lock_content_sha256'] != canonical_hash({k: v for k, v in lock.items() if k != 'lock_content_sha256'}) or
            lock['engine']['code_sha256'] != sha(HERE / 'inference_v3.py') or
            lock['engine']['native_loader_sha256'] != sha(HERE / 'loader_v3.py') or
            lock['engine']['parent_loader']['repo_id'] != args.repo_id or
            lock['engine']['parent_loader']['revision'] != args.revision or
            lock['engine']['parent_loader']['attn_implementation'] != args.attn_implementation or
            lock['engine']['remedy'] != remedy or lock['batch_size'] != args.batch_size):
            raise ValueError('Existing calibration identity differs; do not reuse another run')
        attempt = {'remedy': remedy, 'calibration_file': str(lockpath.resolve()),
            'calibration_q_median': lock['median_q'], 'calibration_argmax_format_proxy': lock['greedy_any_variant_fraction'],
            'calibration_screen_pass': lock['median_q'] >= .70 and lock['greedy_any_variant_fraction'] >= .90,
            'calibration_gate_used': False}
        state['attempts'].append(attempt); write_state(statepath, state)
        for stage, input_path in [('pilot', args.pilot_input), ('development', args.development_input)]:
            result = folder / (stage + '_R0_R1.jsonl')
            if not Path(str(result) + '.summary.json').exists():
                call(runbase + ['--remedy', remedy, '--mode', 'score', '--readouts', 'R0_R1',
                    '--input', input_path, '--output', str(result), '--calibration', str(lockpath)], folder / (stage + '.log'))
            source_path = folder / (stage + '_E0_analysis.json')
            if not source_path.exists():
                call([sys.executable, str(HERE / 'analyze_e0_v3.py'), '--inputs', input_path,
                    '--r0', str(result), '--output', str(source_path)], folder / (stage + '_analysis.log'))
            path = folder / (stage + '_E0_conditional_analysis.json')
            metrics = conditional.ensure_analysis(source_path, input_path, result, lockpath, path, args.amendment_file)
            if metrics['qualification']['R0_operational'] == 'incomplete' or not metrics['qualification']['R1_candidate_scoring_valid']:
                raise ValueError('Missing operational/R1 measurements; do not classify as instruction failure')
            attempt[stage] = {'analysis': str(path.resolve()), 'original_analysis': str(source_path.resolve()),
                'operational_gates': metrics['operational_gates'], 'operational_policy': metrics['operational_policy'],
                'qualification': metrics['qualification']}
            write_state(statepath, state)
            passed = metrics['qualification']['R0_operational'] == 'pass'
            if not passed and remedy != MODES[-1]:
                attempt['decision'] = 'advance_due_to_' + stage + '_operational_union'; write_state(statepath, state); break
            if args.stop_after == stage or stage == 'development':
                state.update(status='complete' if stage == 'development' else 'pilot_selection_complete',
                    selected_remedy=remedy, selected_calibration=str(lockpath.resolve()),
                    final_qualification=metrics['qualification']['final_qualification'],
                    qualification=metrics['qualification'], selected_analysis=str(path.resolve()))
                attempt['decision'] = 'retain_earliest_operational_union_variant' if passed else 'retain_R1_only_last_rescue'
                break
        if state['status'] != 'running':
            break
    state['finished_utc'] = datetime.now(timezone.utc).isoformat(); write_state(statepath, state)
    print(json.dumps({k: state.get(k) for k in ('status', 'repo_id', 'selected_remedy', 'final_qualification')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
