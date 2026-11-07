"""Serial native G4 successor; preserve every generated raw run and replay ACK."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def execute(manifest_path):
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent; ext = root.parent
    sys.path[:0] = [str(root), str(ext)]
    from common import load, sha, verify_bindings, write_new, atomic_status
    from checkpoint import prepare_checkpoint
    from panel_checkpoint import rotate, validate_ack
    from dialogue_provenance import closure
    from evidence_delta import create
    from dialogue_bundle import release_work
    from dialogue_analysis import analyze, report
    manifest = load(manifest_path); policy_path = Path(manifest['policy']); policy = load(policy_path)
    base = Path(policy['base']); sys.path[:0] = [str(base / 'setup'), str(base / 'code')]
    if not os.environ.get('STY'): raise RuntimeError('Persistent screen required')
    lock = (root / 'G4_dispatcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'plan_sha256': sha(policy_path), 'models': [], 'GPU_work_started': False,
             'planned_runs': 1800, 'planned_runs_per_model': 600, 'full_v31_goal_complete': False}
    def save(**changes):
        state.update(changes); state['utc'] = datetime.now(timezone.utc).isoformat()
        atomic_status(root / 'G4_status.json', state)
    def archived(path, repo, kind):
        with (root / 'archive_index.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            info = create(base, ext, path, repo, kind, closure({str(path): sha(path)}, base, ext), index_name='e7_index.json')
        ack_path = ext / 'archives/off_server_ack' / (info['archive_sha256'] + '.json')
        while not ack_path.exists():
            save(status='waiting_independent_backup', archive_sha256=info['archive_sha256']); time.sleep(10)
        ack = load(ack_path); validate_ack(info, ack); return info, ack
    lease = base / 'setup/e1_e2_queue.lock'; acquired = False
    try:
        verify_bindings(manifest['source_sha256']); verify_bindings(policy['source_sha256'])
        archived(manifest_path, 'v31_G4', 'v31_G4_prospective')
        save(status='waiting_G2_G3_sealed_completion')
        while not Path(policy['predecessor']).exists():
            status_path = root / 'G2_G3_status.json'
            if status_path.exists() and 'error' in load(status_path)['status']:
                raise RuntimeError('Predecessor error requires inspection')
            time.sleep(10)
        prior = load(policy['predecessor']); verify_bindings(prior['source_sha256'])
        if prior['status'] != 'G2_G3_complete_sealed' or len(prior['completed_jobs']) != 96:
            raise ValueError('G2/G3 sealed coverage differs')
        while lease.exists():
            if load(lease).get('pid') != prior['pid']: raise RuntimeError('Unexpected shared lease owner')
            time.sleep(1)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('Another CUDA process is active')
        with lease.open('x') as handle:
            json.dump({'pid': os.getpid(), 'kind': 'v3_E6_E7', 'stage': 'v31_G4', 'history': str(root)}, handle)
        acquired = True; analyses = []
        for plan_name in manifest['plans']:
            path = Path(plan_name); plan = load(path); verify_bindings(plan['source_sha256'])
            model = plan['model']; folder = ext / 'results/dialogue' / model['slug'] / 'E6_v31'
            folder.mkdir(parents=True, exist_ok=True); complete = folder / 'native_completion.json'
            if not load(base / 'operations/cloud_goal_20261031/goal.json')['status'].startswith('running_'):
                raise RuntimeError('Goal no longer authorizes execution')
            save(status='restoring_pinned_dialogue_checkpoint', current_model=model['repo_id'], GPU_work_started=False)
            if not complete.exists():
                prepare_checkpoint(model, ext)
                save(status='running_native_dialogue', GPU_work_started=True, model_plan=str(path), started_native_utc=datetime.now(timezone.utc).isoformat())
                started = time.monotonic()
                with (folder / 'driver.log').open('a') as handle:
                    subprocess.run([sys.executable, str(ext / 'dialogue_native.py'), '--plan', str(path)],
                        stdout=handle, stderr=subprocess.STDOUT, check=True,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                elapsed = time.monotonic() - started
            else: elapsed = None
            native = load(complete)
            if (native['status'] != 'complete_native_runs_backups_may_be_pending' or native['model_plan_sha256'] != sha(path)
                    or native['run_ids'] != [r['run_id'] for r in plan['runs']] or len(native['bundles']) != 600):
                raise ValueError('Native panel is not the exact registered 600-run set')
            save(status='verifying_native_raw_backups', GPU_work_started=False)
            for info in native['bundles']:
                if sha(info['archive']) != info['archive_sha256']: raise ValueError('Permanent native raw bundle changed')
                ack_path = ext / 'archives/dialogue_off_server_ack' / (info['archive_sha256'] + '.json')
                while not ack_path.exists():
                    save(status='waiting_independent_native_replay', run_id=info['run_id']); time.sleep(10)
                release_work(info, load(ack_path), ext)
            output = folder / 'analysis.json'; save(status='CPU_replaying_and_analyzing_dialogue')
            if not output.exists(): analyze(path, complete, output)
            else: verify_bindings(load(output)['source_sha256'])
            analyses.append(output); info, ack = archived(output, model['repo_id'], 'v31_G4_model_analysis')
            released = rotate(model['execution_model_dir'], model, info, ack, root / 'rotation/G4', ext / 'model_cache',
                'All 600 native run bundles independently replayed; model analysis and shared sources independently verified')
            state['models'].append({'repo_id': model['repo_id'], 'analysis': str(output), 'analysis_sha256': sha(output),
                'native_runs': 600, 'native_wall_seconds': elapsed, 'archive_sha256': info['archive_sha256'],
                'rotated_weight_bytes': released['bytes_released']}); save(status='advancing_G4_model')
        target = root / 'G4/report.json'; report(policy_path, analyses, target)
        archived(target, 'v31_G4', 'v31_G4_report')
        save(status='G4_complete_verified_1800_native_runs', current_model=None, GPU_work_started=False,
             report=str(target), report_sha256=sha(target))
        write_new(root / 'G4_complete.json', {**state, 'source_sha256': {str(target): sha(target), str(manifest_path): sha(manifest_path)}})
    except BaseException as error:
        save(status='G4_error_preserve_all_evidence', error_type=type(error).__name__, error=str(error)); raise
    finally:
        if acquired and lease.exists() and load(lease).get('pid') == os.getpid():
            lease.replace(root / ('G4_released_GPU_lease_' + str(os.getpid()) + '.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    execute(parser.parse_args().plan)
