"""Exact E1 plan -> explicit six-model freeze -> E2 + CPU analyses -> short report.

Requires screen. Stops after the prerequisite report; never starts E4. Missing
live producers receive a bounded grace period, while actual inference/analysis
may continue for its natural runtime. Failures never classify a model as rejected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_analysis_watcher_v3 as watcher
import run_e1_e2_queue as queue


def live_lease(path):
    if not path.exists(): return False
    try: return watcher.pid_alive(queue.load(path)['pid'])
    except (OSError, ValueError, KeyError): return False


def wait_closed_plan(plan_path, stages, save, grace=90):
    """Require producer terminal receipts, then revalidate every closed score/analysis."""
    base, plan, decision, models, sources = watcher.prepare(plan_path)
    if plan['stages'] != stages or plan.get('pending_models'):
        raise ValueError('Plan has unexpected stages or unresolved model admissions')
    absent = None; signatures = {}
    while True:
        watcher.verify_frozen(sources, signatures)
        final_path = plan_path.parent/'final_status.json'
        analysis_path = plan_path.parent/'analysis_watcher_status.json'
        final = queue.load(final_path) if final_path.exists() else None
        analyses = queue.load(analysis_path) if analysis_path.exists() else None
        for receipt in (final, analyses):
            if receipt and not watcher.same_path(receipt.get('plan', ''), plan_path):
                raise ValueError('A terminal receipt belongs to a different explicit plan')
        if final and final.get('status') == 'finished' and not final.get('all_requested_scores_complete'):
            raise RuntimeError('Queue terminated with incomplete scores; execution error is not model rejection')
        if analyses and analyses.get('status') in ('finished_with_pending_or_errors', 'watcher_error_not_model_disqualification'):
            raise RuntimeError('CPU watcher terminated with pending/error analyses')
        if (final and final.get('status') == 'finished' and final.get('all_requested_scores_complete')
                and analyses and analyses.get('status') == 'complete' and analyses.get('all_requested_analyses_complete')):
            if analyses['plan_sha256'] != queue.sha(plan_path):
                raise ValueError('Watcher receipt plan SHA changed')
            for path, digest in analyses['frozen_file_sha256'].items():
                if queue.sha(path) != digest: raise ValueError('Watcher frozen source changed: '+path)
            for job in plan['jobs']:
                if queue.score_closure(job, models[job['model']], decision) is None:
                    raise ValueError('Terminal plan lacks a closed score: '+job['name'])
                for task in job['analyses']:
                    if watcher.analysis_closure(task) is None:
                        raise ValueError('Terminal plan lacks a closed analysis: '+task['name'])
            return plan
        active = watcher.queue_live(base, plan_path.parent) or live_lease(plan_path.parent/'analysis_watcher.lock')
        absent = None if active else time.monotonic() if absent is None else absent
        if absent is not None and time.monotonic()-absent >= grace:
            raise RuntimeError('Producer processes disappeared before closed receipts; awaiting root repair')
        save({'waiting_plan': str(plan_path), 'queue_status': final.get('status') if final else 'pending',
              'analysis_status': analyses.get('status') if analyses else 'pending', 'producer_active': active})
        time.sleep(15)


def new_child_plan(base, child_pid, previous):
    lease = base/'setup/e1_e2_queue.lock'
    if not lease.exists(): return None
    try: owner = queue.load(lease)
    except (ValueError, OSError): return None
    if owner.get('pid') != child_pid: return None
    history = Path(owner['history']).resolve()
    if history.parent != (base/'setup/e1_e2_queue_history').resolve() or history in previous:
        raise ValueError('E2 child lease must identify a new plan under this project')
    path = history/'plan.json'; plan = queue.load(path)
    if plan.get('stages') != ['E2'] or plan.get('dry_run') or plan.get('analysis_mode') != 'defer':
        raise ValueError('E2 child produced an unexpected plan')
    return path


def cpu_command(command, log):
    environment = {**os.environ, **watcher.CPU_ENV, 'CUDA_VISIBLE_DEVICES': ''}
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a', encoding='utf-8') as stream:
        stream.write('\nInvocation '+queue.now()+'\n'); stream.flush()
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False).returncode


def run_E2(base, models, history, save):
    previous = {path.resolve() for path in (base/'setup/e1_e2_queue_history').glob('*') if path.is_dir()}
    command = [sys.executable, str(base/'setup/run_e1_e2_queue.py'), '--root', str(base),
               '--stages', 'E2', '--analysis-mode', 'defer', '--models', *models]
    with (history/'E2_queue.log').open('a', encoding='utf-8') as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    save({'status': 'E2_running', 'E2_pid': child.pid, 'E2_command': command})
    plan = None
    while child.poll() is None:
        plan = new_child_plan(base, child.pid, previous)
        if plan is not None: break
        time.sleep(15)
    if plan is None: raise RuntimeError('E2 exited before an exact child PID/new-plan lease was observed')
    cpu = [sys.executable, str(base/'setup/run_analysis_watcher_v3.py'), '--plan', str(plan)]
    with (history/'E2_analysis_watcher.log').open('a', encoding='utf-8') as output:
        analysis = subprocess.Popen(cpu, stdout=output, stderr=subprocess.STDOUT)
    save({'E2_plan': str(plan), 'E2_plan_sha256': queue.sha(plan), 'E2_watcher_pid': analysis.pid, 'E2_watcher_command': cpu})
    while True:
        score_rc, analysis_rc = child.poll(), analysis.poll()
        save({'E2_exit_code': score_rc, 'E2_watcher_exit_code': analysis_rc})
        if score_rc not in (None, 0) or analysis_rc not in (None, 0):
            raise RuntimeError('E2 queue or CPU watcher exited with an execution error; live child PIDs retained in status')
        if score_rc is not None and analysis_rc is not None: break
        time.sleep(15)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--e1-plan', required=True, type=Path)
    parser.add_argument('--selection-prefix', required=True, type=Path)
    parser.add_argument('--report-prefix', required=True, type=Path)
    parser.add_argument('--missing-process-grace', type=float, default=90)
    args = parser.parse_args(); base = args.root.resolve(); e1_path = args.e1_plan.resolve()
    if not os.environ.get('STY'): raise RuntimeError('E1 continuation and GPU children must run inside screen')
    if args.missing_process_grace < 30: raise ValueError('Missing-process grace must allow at least two polls')
    if e1_path.parent.parent.parent.parent != base: raise ValueError('Explicit E1 plan must belong to --root')
    selection_prefix = args.selection_prefix.resolve(); report_prefix = args.report_prefix.resolve()
    for prefix, suffixes in [(selection_prefix, ('.json', '.md', '_E4_selection.json')), (report_prefix, ('.json', '.md'))]:
        if any(Path(str(prefix)+suffix).exists() for suffix in suffixes):
            raise FileExistsError('Use fresh selection and report prefixes; existing artifacts are preserved')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history = base/'setup/continue_after_e1_history'/stamp; history.mkdir(parents=True)
    lease = base/'setup/continue_after_e1.lock'
    if lease.exists():
        if live_lease(lease): raise RuntimeError('An E1 continuation coordinator is active')
        lease.replace(history/'stale_coordinator_lock.json')
    with lease.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'history': str(history)}, stream)
    state = {'status': 'waiting_E1', 'started_utc': queue.now(), 'pid': os.getpid(),
             'history': str(history), 'e1_plan': str(e1_path), 'e1_plan_sha256': queue.sha(e1_path),
             'selection_prefix': str(selection_prefix), 'report_prefix': str(report_prefix),
             'coordinator_sha256': queue.sha(__file__), 'automatic_scope': 'E1_close_then_freeze_then_E2_then_prerequisite_report',
             'E4_launched': False, 'execution_errors_are_not_model_rejections': True}
    def save(changes=None):
        if changes: state.update(changes)
        state['updated_utc'] = queue.now()
        queue.write(base/'setup/continue_after_e1_status.json', state); queue.write(history/'status.json', state)
    paired = []; frozen_path = Path(str(selection_prefix)+'_E4_selection.json'); requested = []
    def write_report():
        command = [sys.executable, str(base/'code/write_prerequisite_report_v3.py'), '--root', str(base),
                   '--output-prefix', str(report_prefix)]
        if requested: command += ['--models', *requested]
        if frozen_path.exists(): command += ['--selection', str(frozen_path)]
        for path in paired: command += ['--e2-report', str(path)]
        save({'report_command': command}); rc = cpu_command(command, history/'prerequisite_report.log')
        save({'report_exit_code': rc})
        if rc: raise RuntimeError('Prerequisite report generation failed; retain log')
        report = queue.load(str(report_prefix)+'.json')
        save({'report_json': str(report_prefix)+'.json', 'report_markdown': str(report_prefix)+'.md',
              'report_status': report['status'], 'report_counts': report['counts']})
        return report
    try:
        save(); plan = wait_closed_plan(e1_path, ['E1'], save, args.missing_process_grace)
        requested = [model['repo_id'] for model in plan['selected_models']]
        core = [model['repo_id'] for model in plan['selected_models'] if model['R0_operational'] and model['core_E2_panel']]
        if not core: raise RuntimeError('No operational core models are available for the requested E2 stage')
        freeze = [sys.executable, str(base/'code/summarize_select_e1_v3.py'), '--root', str(base),
                  '--models', *requested, '--output-prefix', str(selection_prefix), '--freeze']
        save({'status': 'freezing_E4_selection', 'selection_command': freeze})
        rc = cpu_command(freeze, history/'E1_selection.log'); save({'selection_exit_code': rc})
        if rc: raise RuntimeError('E1 selector could not freeze six eligible models')
        selected = queue.load(frozen_path)
        if selected.get('status') != 'frozen' or len(set(selected['selected_repos'])) != 6:
            raise ValueError('Selector returned without an explicit six-model freeze')
        save({'selection_file': str(frozen_path), 'selection_sha256': queue.sha(frozen_path), 'E2_operational_core': core})
        e2_path = run_E2(base, core, history, save)
        wait_closed_plan(e2_path, ['E2'], save, args.missing_process_grace)
        save({'status': 'E2_paired_comparisons', 'paired_jobs': []})
        for repo in core:
            slug = repo.replace('/', '--'); prefix = history/'paired'/slug
            command = [sys.executable, str(base/'code/compare_e2_v3.py'), '--root', str(base),
                       '--e1-semantic64', str(base/'results'/slug/'e1/semantic64_R0_analysis.json'),
                       '--e2-dir', str(base/'results'/slug/'e2'), '--output-prefix', str(prefix), '--draws', '10000']
            rc = cpu_command(command, history/'paired'/(slug+'.log'))
            state['paired_jobs'].append({'model': repo, 'command': command, 'exit_code': rc,
                                        'output': str(prefix)+'.json'}); save()
            if rc == 0: paired.append(Path(str(prefix)+'.json'))
        report = write_report()
        if any(job['exit_code'] for job in state['paired_jobs']) or report['status'] != 'complete_verified_for_requested_scope':
            raise RuntimeError('Prerequisite report remains intermediate; inspect explicit per-model pending/errors')
        save({'status': 'complete_prerequisite_scope', 'finished_utc': queue.now()})
    except BaseException as error:
        save({'status': 'terminal_error_not_model_rejection', 'error': str(error), 'finished_utc': queue.now()})
        if isinstance(error, Exception) and not Path(str(report_prefix)+'.json').exists():
            try: write_report()
            except Exception as report_error: save({'intermediate_report_error': str(report_error)})
        raise
    finally:
        if lease.exists() and queue.load(lease).get('pid') == os.getpid(): lease.unlink()


if __name__ == '__main__':
    main()
