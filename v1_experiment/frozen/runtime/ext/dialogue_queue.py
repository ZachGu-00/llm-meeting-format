"""Single-GPU E6 successor with persistent native generation and verified raw retention."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_status, load, sha, verify_bindings, write_new
from checkpoint import prepare_checkpoint
from panel_checkpoint import rotate, validate_ack
from dialogue_plan import model_plan
from dialogue_bundle import release_work
from dialogue_analysis import analyze, report
from dialogue_archive import archive


def predecessor_closed(policy, deep=False):
    path = Path(policy['predecessor_plan'])
    if sha(path) != policy['predecessor_plan_sha256']: raise ValueError('E8 predecessor registration changed')
    terminal = path.parent / 'final_status.json'
    if not terminal.exists():
        status = path.parent / 'status.json'
        if status.exists() and load(status)['status'] == 'execution_error_not_model_rejection':
            raise RuntimeError('E8 needs inspection; do not bypass it')
        return False
    result = load(terminal)
    if result['status'] != 'complete_available_E8_scope' or not result['E8_available_complete'] or result['plan_sha256'] != sha(path):
        raise ValueError('E8 predecessor is not closed')
    if sha(result['report']) != result['report_sha256']: raise ValueError('E8 terminal report changed')
    if deep: verify_bindings(load(result['report'])['source_sha256'])
    proc = Path('/proc') / str(result['pid']) / 'cmdline'
    if proc.exists() and any(Path(x.decode(errors='replace')).name == 'e8_queue.py' for x in proc.read_bytes().split(b'\0') if x):
        return False
    return True


def execute(registration, preflight=False):
    registration = Path(registration).resolve(); policy = load(registration); verify_bindings(policy['frozen_file_sha256'])
    root, base = Path(policy['extension']), Path(policy['base']); history = registration.parent
    sys.path[:0] = [str(base / 'setup'), str(base / 'code')]
    from e9_queue import active_gpu_work
    if preflight: return {'status': 'preflight_pass', 'predecessor_closed': bool(predecessor_closed(policy)), 'GPU_work_started': False}
    if not os.environ.get('STY'): raise RuntimeError('Persistent screen required')
    lock = (history / 'dispatcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    terminal = history / 'final_status.json'
    if terminal.exists():
        prior = load(terminal)
        if prior['plan_sha256'] != sha(registration): raise ValueError('E6 terminal registration differs')
        verify_bindings(load(prior['report'])['source_sha256']); return prior
    state = {'status': 'waiting_for_E8', 'pid': os.getpid(), 'plan_sha256': sha(registration),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'models': [], 'GPU_work_started': False,
        'full_v3_goal_complete': False}
    def save(**changes):
        state.update(changes); state['updated_utc'] = datetime.now(timezone.utc).isoformat(); atomic_status(history / 'status.json', state)
    def wait_shared_backup(info):
        ack = root / 'archives/off_server_ack' / (info['archive_sha256'] + '.json')
        while not ack.exists(): save(status='waiting_for_verified_shared_sources_or_analysis_backup', archive_sha256=info['archive_sha256']); time.sleep(30)
        result = load(ack); validate_ack(info, result); return result
    lease = base / 'setup/e1_e2_queue.lock'
    try:
        save()
        while not predecessor_closed(policy) or lease.exists() or active_gpu_work():
            save(status='waiting_for_E8'); time.sleep(30)
        predecessor_closed(policy, deep=True); verify_bindings(policy['frozen_file_sha256'])
        with lease.open('x') as f: json.dump({'pid': os.getpid(), 'kind': 'v3_E6_E7', 'history': str(history)}, f)
        from advancement_gate import evaluate
        gate_path = root / 'plans/advancement/result.json'
        if not gate_path.exists(): evaluate(policy['advancement_registration'], gate_path)
        else: verify_bindings(load(gate_path)['source_sha256'])
        analyses = []
        for model in policy['models']:
            save(current_model=model['repo_id'], status='registering_PK_selected_dialogue_runs')
            path, plan = model_plan(registration, model)
            wait_shared_backup(archive(path))
            folder = root / 'results/dialogue' / model['slug'] / 'E6'; folder.mkdir(parents=True, exist_ok=True)
            complete = folder / 'native_completion.json'
            if not complete.exists():
                save(status='restoring_pinned_dialogue_checkpoint'); prepare_checkpoint(model, root)
                if active_gpu_work(): raise RuntimeError('Another GPU worker is active')
                save(status='running_native_dialogue', GPU_work_started=True, model_plan=str(path))
                with (folder / 'driver.log').open('a') as handle:
                    result = subprocess.run([sys.executable, str(root / 'dialogue_native.py'), '--plan', str(path)],
                        stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                if result.returncode: raise RuntimeError('Native dialogue stopped; retain partial artifacts and inspect ' + str(folder / 'driver.log'))
            native = load(complete)
            if native['model_plan_sha256'] != sha(path) or native['run_ids'] != [r['run_id'] for r in plan['runs']]:
                raise ValueError('Incomplete registered native dialogue run set')
            for info in native['bundles']:
                ack = root / 'archives/dialogue_off_server_ack' / (info['archive_sha256'] + '.json')
                while not ack.exists(): save(status='waiting_for_verified_native_run_backup', run_id=info['run_id']); time.sleep(30)
                release_work(info, load(ack), root)
            output = folder / 'analysis.json'; save(status='replaying_and_analyzing_native_dialogue')
            if not output.exists(): analyze(path, complete, output)
            else: verify_bindings(load(output)['source_sha256'])
            analyses.append(output); info = archive(path, output); ack = wait_shared_backup(info)
            released = rotate(model['execution_model_dir'], model, info, ack, root / 'provenance/E6_weight_rotation', root / 'model_cache',
                'All 600 native generated-dialogue run bundles independently replay-verified and backed up; full source/analysis archive verified')
            state['models'].append({'repo_id': model['repo_id'], 'analysis': str(output), 'analysis_sha256': sha(output),
                'native_runs': len(native['bundles']), 'archive_sha256': info['archive_sha256'], 'rotated_weight_bytes': released['bytes_released']})
            save(status='advancing')
        output = history / 'report.json'; report(registration, analyses, output)
        save(status='complete_verified_E6', E6_complete=True, report=str(output), report_sha256=sha(output),
            E7_advance=load(gate_path)['E7_advance'], finished_utc=datetime.now(timezone.utc).isoformat(), current_model=None)
        write_new(terminal, state); return state
    except BaseException as error:
        save(status='execution_error_not_model_rejection', error_type=type(error).__name__, error=str(error)); raise
    finally:
        if lease.exists() and load(lease).get('pid') == os.getpid(): lease.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--preflight', action='store_true')
    a = p.parse_args(); print(json.dumps(execute(a.plan, a.preflight), ensure_ascii=False))
