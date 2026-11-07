"""Serial E9 dispatcher; wait for complete E4, then acquire the shared GPU lease."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import atomic_status, load, sha, verify_bindings, write_new
from checkpoint import prepare_checkpoint, RESERVE

def now(): return datetime.now(timezone.utc).isoformat()

def check_terminal_identity(terminal, expected):
    if terminal.get('status') != 'complete_E4_scope' or terminal.get('E4_complete') is not True:
        raise ValueError('E4 is not in its complete terminal state')
    for key in ('history', 'score_plan', 'score_plan_sha256', 'selection_sha256'):
        if terminal.get(key) != expected[key]:
            raise ValueError('E4 terminal identity mismatch: ' + key)
    if terminal.get('report_json') != expected['report']:
        raise ValueError('Unexpected E4 report path')

def e4_closed(plan, deep=False):
    expected = plan['e4_predecessor']; history = Path(expected['history'])
    path = history / 'final_status.json'
    if not path.exists():
        status = load(history / 'status.json')
        if status.get('status') == 'terminal_error_not_model_rejection':
            raise RuntimeError('E4 needs inspection; no new experiment will start')
        return False
    terminal = load(path); check_terminal_identity(terminal, expected)
    if sha(expected['report']) != terminal['report_sha256']:
        raise ValueError('E4 report digest differs from its terminal receipt')
    if deep:
        phase = importlib.import_module('run_e4_phase_v3')
        base = Path(plan['base']); selection = load(expected['selection'])
        phase.score_and_cpu_closed(Path(expected['score_plan']), expected['score_plan_sha256'])
        phase.dynamic_closed(Path(terminal['dynamics_plan']), terminal['dynamics_plan_sha256'],
            base, expected['selection_sha256'], Path(expected['score_plan']), terminal['benchmark_sha256'])
        phase.cross_model_closed(Path(terminal['cross_model_output']), expected['selection_sha256'], selection['selected_repos'])
        report = phase.report_closed(Path(expected['report']), expected['selection_sha256'], selection['selected_repos'])
        verify_bindings(report['verified_source_sha256'])
        for source in terminal['dynamic_aggregate_outputs']:
            result = load(source); verify_bindings(result['file_sha256'])
            artifact = result['bootstrap_artifact']
            if sha(Path(source).parent / artifact['filename']) != artifact['sha256']:
                raise ValueError('E4 aggregation sidecar changed')
    return {'terminal': str(path), 'terminal_sha256': sha(path),
            'report': expected['report'], 'report_sha256': terminal['report_sha256']}

def active_gpu_work():
    programs = {'inference_v3.py', 'run_e0_v3.py', 'run_e0_v3_taskpilot.py', 'run_e0_v3_conditional.py',
        'run_e0_queue.py', 'run_e0_taskpilot_recovery.py', 'run_e0_conditional_queue.py',
        'run_e1_e2_queue.py', 'run_e4_score_queue.py', 'run_e4_phase_v3.py',
        'run_e4_dynamics_queue_v3.py', 'run_e4_dynamics_v3.py', 'run_e9_affine_queue_v3.py',
        'benchmark_e4_gpu.py', 'continue_after_e0.py', 'e9_queue.py'}
    found = []
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name) == os.getpid(): continue
        try: args = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        args = [x.decode(errors='ignore') for x in args if x]
        if args and Path(args[0]).name.startswith('python') and any(Path(x).name in programs for x in args[1:]):
            found.append(int(process.name))
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True)
    found.extend(int(p) for p in gpu.stdout.splitlines() if p.strip().isdigit() and int(p) != os.getpid())
    return sorted(set(found))

def score_closed(job, model):
    old = importlib.import_module('run_e1_e2_queue')
    receipt = old.score_closure(job, model, {'batch_size': job['batch_size']})
    if receipt is None: return None
    run = load(job['output'] + '.run.json')
    for field in ('format_record_ids', 'r2_record_ids'):
        if len(run[field]) != job['rows'] or len(set(run[field])) != job['rows']:
            raise ValueError('Full E9 requires independent generation diagnostics on all inputs')
    analysis = importlib.import_module('analyze_v3')
    n = 0
    for row in analysis.read_closed(job['input'], job['output']):
        if not row.get('diagnostic_is_actual_generate_call') or not row.get('diagnostic_r2_selected'):
            raise ValueError('Closed output lacks required native R2')
        n += 1
    if n != job['rows']: raise ValueError('Row closure differs')
    return receipt

def execute(plan_path, wait=False, preflight=False):
    plan_path = Path(plan_path).resolve(); plan = load(plan_path)
    if plan.get('schema') != 'voices_v3_E9_full_plan_v1': raise ValueError('Unexpected plan schema')
    base, root = Path(plan['base']), Path(plan['extension'])
    sys.path[:0] = [str(base / 'setup'), str(base / 'code')]
    verify_bindings(plan['frozen_file_sha256'])
    if preflight:
        return {'preflight': 'pass', 'E4_closed': bool(e4_closed(plan)),
                'active_GPU_or_controller_pids': active_gpu_work(), 'GPU_work_started': False,
                'jobs': len(plan['jobs']), 'prompts': plan['total_physical_prompts']}
    if not os.environ.get('STY'): raise RuntimeError('Execute only inside a persistent screen session')
    history = plan_path.parent
    lock = (history / 'dispatcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (history / 'final_status.json').exists():
        terminal = load(history / 'final_status.json')
        if (terminal.get('status') != 'complete_E9_initial10' or not terminal.get('E9_initial10_complete')
            or terminal.get('plan_sha256') != sha(plan_path)):
            raise ValueError('Existing E9 terminal receipt differs from this exact plan')
        result = load(terminal['report']['json']); verify_bindings(result['source_sha256'])
        if sha(terminal['report']['json']) != terminal['report']['sha256']:
            raise ValueError('Closed E9 report changed')
        return terminal
    state = {'status': 'waiting_for_E4', 'pid': os.getpid(), 'plan': str(plan_path),
             'plan_sha256': sha(plan_path), 'started_utc': now(), 'jobs': [], 'models': [],
             'GPU_work_started': False, 'full_v3_goal_complete': False}
    def save(**changes):
        state.update(changes); state['updated_utc'] = now(); atomic_status(history / 'status.json', state)
    lease = base / 'setup/e1_e2_queue.lock'
    try:
        save()
        while not e4_closed(plan) or active_gpu_work() or lease.exists():
            save(status='waiting_for_E4', free_bytes=shutil.disk_usage(root).free)
            if not wait: raise RuntimeError('E4/controller/shared lease is still active')
            time.sleep(30)
        verify_bindings(plan['frozen_file_sha256'])
        predecessor = e4_closed(plan, deep=True)
        if active_gpu_work(): raise RuntimeError('Another controller appeared before lease acquisition')
        with lease.open('x') as f:
            json.dump({'pid': os.getpid(), 'kind': 'E9_full', 'history': str(history)}, f)
        save(status='running', E4_verified=predecessor)
        for model in plan['models']:
            model_jobs = [j for j in plan['jobs'] if j['model'] == model['repo_id']]
            needed = [j for j in model_jobs if score_closed(j, model) is None]
            if needed:
                if active_gpu_work(): raise RuntimeError('Another GPU worker appeared between models')
                save(status='preparing_pinned_checkpoint', current_model=model['repo_id'])
                checkpoint = prepare_checkpoint(model, root)
                save(checkpoint=checkpoint, status='running')
            for job in model_jobs:
                verify_bindings(plan['frozen_file_sha256'])
                save(current_job={k: job[k] for k in ('model', 'name', 'output', 'rows', 'batch_size')})
                receipt = score_closed(job, model)
                if receipt is None:
                    if active_gpu_work(): raise RuntimeError('Another GPU worker appeared between jobs')
                    if shutil.disk_usage(root).free < RESERVE + 512 * 1024 ** 2:
                        raise OSError('Disk reserve reached; preserve partial results and inspect storage')
                    Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
                    save(GPU_work_started=True)
                    with Path(job['log']).open('a') as log:
                        child = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT)
                        save(child_pid=child.pid)
                        code = child.wait()
                    if code: raise RuntimeError('Scoring child failed; exact partial rows/log retained: ' + job['name'])
                    receipt = score_closed(job, model)
                    if receipt is None: raise ValueError('Scoring returned without complete closure')
                state['jobs'].append({'model': model['repo_id'], 'name': job['name'], **receipt})
                save(child_pid=None)
            output = root / 'results/e9_full' / model['slug'] / 'analysis.json'
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, str(root / 'e9_analysis.py'), '--plan', str(plan_path),
                       '--model', model['repo_id'], '--output', str(output)]
            save(status='CPU_analysis', current_model=model['repo_id'], current_job=None)
            with output.with_suffix('.log').open('a') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
            record = load(output)
            if record.get('status') != 'complete_verified': raise ValueError('Model analysis incomplete')
            verify_bindings(record['source_sha256'])
            state['models'].append({'repo_id': model['repo_id'], 'analysis': str(output), 'sha256': sha(output)})
            save(status='running')
        from e9_report import report
        result = report(plan_path)
        save(status='complete_E9_initial10', E9_initial10_complete=True,
             report=result, current_job=None, finished_utc=now())
        write_new(history / 'final_status.json', state)
        return state
    except BaseException as error:
        save(status='execution_error_not_model_rejection', error_type=type(error).__name__,
             error=str(error), current_job=state.get('current_job'))
        raise
    finally:
        if lease.exists() and load(lease).get('pid') == os.getpid(): lease.unlink()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    p.add_argument('--wait-for-E4', action='store_true'); p.add_argument('--preflight', action='store_true')
    a = p.parse_args(); print(json.dumps(execute(a.plan, a.wait_for_E4, a.preflight), ensure_ascii=False))
