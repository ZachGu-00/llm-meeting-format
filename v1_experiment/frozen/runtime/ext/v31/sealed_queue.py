"""Single GPU lease successor: G0 closure to native G1 collection, no effect analysis."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def audit_rows(job):
    """Content audit restricted to identity and numerical quality, never effects."""
    import hashlib
    run_sha = hashlib.sha256(Path(job['output'] + '.run.json').read_bytes()).hexdigest()
    seen = set(); low = 0; count = 0; diagnostic = 0; diagnostic_valid = 0
    with Path(job['input']).open() as inputs, Path(job['output']).open() as outputs:
        for source, raw in zip(inputs, outputs, strict=True):
            source, row = json.loads(source), json.loads(raw)
            rid = source.get('record_id', source['id'])
            if rid in seen or row.get('record_id', row.get('id')) != rid or row.get('status') != 'ok':
                raise ValueError('Sealed native row identity/status differs')
            seen.add(rid)
            if row.get('run_file_sha256') != run_sha: raise ValueError('Native row/run binding differs')
            for key, value in source.items():
                if row.get(key) != value: raise ValueError('Sealed input metadata differs: ' + key)
            if not math.isfinite(float(row[job['readout'] + '_y_AB'])): raise ValueError('Nonfinite native score')
            if job['readout'] == 'R0':
                q = float(row['q'])
                if not math.isfinite(q) or not -1e-6 <= q <= 1 + 1e-6: raise ValueError('Invalid option mass')
                low += q < .5
            # Format diagnostic availability is recorded; scores are not effect-screened.
            if row.get('diagnostic_format_selected'):
                if not row.get('diagnostic_is_actual_generate_call'): raise ValueError('Missing native format diagnostic')
                diagnostic += 1
                diagnostic_valid += row.get('generated_first_token_selected_letter') in ('A', 'B')
            count += 1
    if count != job['rows']: raise ValueError('Sealed row count differs')
    return {'rows': count, 'all_native_scores_finite': True, 'all_row_metadata_verified': True,
            'option_mass_below_0_5': low, 'format_rows': diagnostic, 'format_valid_rows': diagnostic_valid,
            'effect_analysis_executed': False, 'no_outcome_based_exclusions': True}


def retire_g0(root, plan, save):
    from common import load, sha, verify_bindings, write_new
    from g0_boundary_guard import checked, process
    regpath = Path(plan['G0_boundary_guard']); reg = load(regpath)
    receipt = root / 'G0_retired_at_closed_boundary.json'
    if receipt.exists():
        prior = load(receipt); verify_bindings(prior['source_sha256'])
        for key in ('manager', 'screen'):
            p = process(reg[key]['pid'])
            if p and p['start_ticks'] == reg[key]['start_ticks'] and p['state'] != 'Z':
                raise RuntimeError('Retired old controller is still live')
        return prior
    save(status='waiting_G0_closed_boundary', GPU_work_started=False)
    while True:
        s = load(reg['guard_status'])
        if s['status'] == 'G0_boundary_reached': break
        if s['status'] not in ('armed', 'armed_G0_running'): raise RuntimeError('G0 guard needs inspection')
        if not process(s['guard_pid']): raise RuntimeError('G0 guard disappeared')
        time.sleep(10)
    if s['plan_sha256'] != sha(regpath): raise ValueError('G0 boundary registration changed')
    completion = load(reg['completion']); verify_bindings(completion['source_sha256'])
    if completion['task_plan_sha256'] != reg['task_sha256'] or completion['model'] != reg['model']:
        raise ValueError('G0 completion differs')
    before = {key: checked(reg[key]) for key in ('screen', 'manager')}
    if any(p['state'] != 'T' for p in before.values()): raise RuntimeError('G0 boundary is not stopped')
    if any(p['pid'] not in {reg['screen']['pid'], reg['manager']['pid']} for p in s['stopped']):
        raise RuntimeError('Unexpected boundary descendant; inspect rather than terminate it')
    lease = Path(plan['base']) / 'setup/e1_e2_queue.lock'
    if load(lease).get('pid') != reg['manager']['pid']: raise ValueError('G0 lease ownership changed')
    lease_bytes = lease.read_bytes()
    save(status='retiring_closed_G0_dispatcher')
    # Both registered processes are suspended after every Gemma job has closed.
    # Default SIGTERM is delivered before CONT; no further Python queue work runs.
    for key in ('manager', 'screen'):
        p = checked(reg[key]); os.kill(p['pid'], signal.SIGTERM); os.kill(p['pid'], signal.SIGCONT)
        for _ in range(100):
            live = process(p['pid'])
            if not live or live['state'] == 'Z': break
            time.sleep(.05)
        else: raise RuntimeError('Closed G0 dispatcher did not terminate')
    # Preserve the retired lease itself. Never remove an unrelated or live lease.
    if lease.exists():
        if lease.read_bytes() != lease_bytes: raise ValueError('Lease changed while retiring G0')
        lease.replace(root / 'G0_retired_GPU_lease.json')
    result = {'status': 'G0_closed_old_dispatcher_retired_for_v31', 'utc': datetime.now(timezone.utc).isoformat(),
              'before': before, 'no_unfinished_inference_terminated': True,
              'source_sha256': {str(regpath): sha(regpath), reg['completion']: sha(reg['completion']),
                                reg['guard_status']: sha(reg['guard_status'])},
              'all_other_old_controllers_remain_paused': True}
    write_new(receipt, result); return result


def execute(path):
    root = Path(path).resolve().parent; ext = root.parent
    sys.path[:0] = [str(ext), str(root)]
    from common import load, sha, verify_bindings, write_new, atomic_status
    from dialogue_provenance import closure, VerifiedSources
    from evidence_delta import create
    from checkpoint import prepare_checkpoint, verify_checkpoint
    from panel_checkpoint import prepare, rotate, validate_ack
    from panel_archive import archive
    plan = load(path); base = Path(plan['base']); sys.path.insert(0, str(base / 'setup'))
    from run_e1_e2_queue import score_closure
    if not os.environ.get('STY'): raise RuntimeError('G1 must run in persistent screen')
    lock = (root / 'G1_dispatcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'plan_sha256': sha(path), 'completed_jobs': [], 'CPU_effect_analysis_executed': False}
    def save(**changes):
        state.update(changes); state['utc'] = datetime.now(timezone.utc).isoformat(); atomic_status(root / 'G1_status.json', state)
    def archive_delta(artifact, repo, kind):
        with (root / 'archive_index.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return create(base, ext, artifact, repo, kind, closure({str(artifact): sha(artifact)}, base, ext), index_name='e7_index.json')
    def wait_ack(info):
        ack = ext / 'archives/off_server_ack' / (info['archive_sha256'] + '.json')
        while not ack.exists():
            save(status='waiting_independent_backup', archive_sha256=info['archive_sha256']); time.sleep(10)
        validate_ack(info, load(ack)); return load(ack)
    lease = base / 'setup/e1_e2_queue.lock'; acquired = False
    try:
        verify_bindings(plan['source_sha256'])
        if plan['schema'] != 'voices_v31_G1_sealed_registration_v1' or len(plan['jobs']) != 30:
            raise ValueError('Wrong sealed collection registration')
        prospective = archive_delta(Path(path), 'v31_G1', 'v31_G1_prospective')
        wait_ack(prospective)
        retire_g0(root, plan, save)
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout.strip()
        if gpu: raise RuntimeError('A different CUDA process exists at G1 handover')
        with lease.open('x') as handle:
            json.dump({'pid': os.getpid(), 'kind': 'v31_G1_sealed', 'history': str(root)}, handle)
        acquired = True
        reg = load(plan['G0_boundary_guard'])
        save(status='archiving_closed_G0')
        g0archive = archive(reg['frozen_task'], reg['completion']); wait_ack(g0archive)
        write_new(root / 'G0_verified_backup.json', {'archive': g0archive, 'status': 'G0_complete_independently_backed_up',
                  'source_sha256': {reg['completion']: sha(reg['completion'])}})
        verifier = VerifiedSources()
        for model in plan['models']:
            save(status='preparing_G1_checkpoint', current_model=model['repo_id'], GPU_work_started=False)
            if Path(model['model_dir']).is_relative_to(ext / 'expanded_models'):
                prepare(base, ext, model)
            else:
                prepare_checkpoint(model, ext)
            _, absent = verify_checkpoint(model['execution_model_dir'], model, full_weights=True)
            if absent: raise RuntimeError('G1 checkpoint is incomplete')
            closed = []
            for index, job in enumerate(plan['jobs']):
                if job['model'] != model['repo_id']: continue
                for source, digest in plan['source_sha256'].items(): verifier(source, digest)
                goal = load(base / 'operations/cloud_goal_20261031/goal.json')
                if not goal['status'].startswith('running_'): raise RuntimeError('Execution goal no longer authorizes running')
                save(status='scoring_sealed_G1', current_job={k: job[k] for k in ('name', 'rows', 'output')}, GPU_work_started=True)
                decision = {'batch_size': 16}; result = score_closure(job, model, decision)
                elapsed = None
                if result is None:
                    command = job['command']
                    if model['repo_id'] == 'HuggingFaceTB/SmolLM3-3B':
                        command = [command[0], str(root / 'score_entry.py'), command[1], '--plan', str(path),
                                   '--job-index', str(index), *command[2:]]
                    log = Path(job['log']); log.parent.mkdir(parents=True, exist_ok=True)
                    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    started = time.monotonic()
                    with log.open('a') as f:
                        status_code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
                    elapsed = time.monotonic() - started
                    if status_code: raise RuntimeError('Native G1 scoring failed; preserve partial output and inspect')
                    result = score_closure(job, model, decision)
                if result is None: raise ValueError('G1 native scoring did not close')
                quality = audit_rows(job)
                seal = Path(job['output'] + '.sealed.json')
                bindings = {str(path): sha(path)}
                bindings.update({job['output'] + s: sha(job['output'] + s) for s in ('', '.run.json', '.summary.json')})
                receipt = {'schema': 'voices_v31_sealed_native_score_v1', 'status': 'sealed_quality_audited',
                           'model': model['repo_id'], 'topic': job['topic_id'], 'rows': job['rows'],
                           'quality': quality, 'conditional_quality_caveat': model['conditional_quality_caveat'],
                           'E0_tier': model['tier'], 'source_sha256': bindings, 'closure': result}
                if not seal.exists(): write_new(seal, receipt)
                else: verify_bindings(load(seal)['source_sha256'])
                closed.append(seal)
                state['completed_jobs'].append({'model': model['repo_id'], 'topic': job['topic_id'],
                                               'sealed_receipt': str(seal), 'rows': job['rows'],
                                               'wall_seconds': elapsed, 'rows_per_wall_second': job['rows']/elapsed if elapsed else None})
                save(status='closed_sealed_G1_job')
            complete = root / 'sealed/G1' / model['slug'] / 'completion.json'
            bindings = {str(p): sha(p) for p in closed}
            bindings[str(root / 'G0_verified_backup.json')] = sha(root / 'G0_verified_backup.json')
            write_new(complete, {'schema': 'voices_v31_G1_model_seal_v1', 'status': 'six_topics_sealed_no_effect_analysis',
                      'repo_id': model['repo_id'], 'rows': 294120, 'source_sha256': bindings})
            save(status='archiving_sealed_G1_model')
            info = archive_delta(complete, model['repo_id'], 'v31_G1_sealed_model'); ack = wait_ack(info)
            save(status='rotating_backed_up_G1_weight_bytes')
            parent = Path(model['execution_model_dir']).parent
            rotate(model['execution_model_dir'], model, info, ack, root / 'rotation/G1', parent,
                   'G0 and current six G1 topics closed and independently backed up; only reproducible weights rotate, later G6 restores the exact pinned bytes')
        save(status='G1_complete_sealed', current_model=None, current_job=None, GPU_work_started=False,
             sealed_rows=sum(j['rows'] for j in plan['jobs']), full_v31_goal_complete=False)
        write_new(root / 'G1_complete.json', {**state, 'source_sha256': {str(path): sha(path)}})
    except BaseException as error:
        save(status='G1_execution_error_preserve_all_evidence', error_type=type(error).__name__, error=str(error), full_v31_goal_complete=False)
        raise
    finally:
        if acquired and lease.exists() and load(lease).get('pid') == os.getpid():
            lease.replace(root / ('G1_released_GPU_lease_' + str(os.getpid()) + '.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); execute(p.parse_args().plan)
