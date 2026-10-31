"""Analyze closed E1/E2 scores concurrently with a deferred GPU queue.

An exact persisted --plan is required. This process only invokes the two CPU
analysis programs, sequentially; it never invokes a score command or loads a
checkpoint. Its status, event log and lease belong to that plan's history folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import run_e1_e2_queue as queue


ANALYSIS_SCRIPTS = ('analyze_v3.py', 'analyze_v3_counts.py')
CPU_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}


def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_key(job):
    return (job['model'], job['stage'], job['name'])


def fingerprint(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns)


def prepare(plan_path):
    """Verify the explicit plan and collect immutable CPU-side dependencies."""
    plan_path = Path(plan_path).resolve()
    history = plan_path.parent
    if (plan_path.name != 'plan.json' or history.parent.name != 'e1_e2_queue_history'
            or history.parent.parent.name != 'setup'):
        raise ValueError('--plan must name a persisted setup/e1_e2_queue_history/<run>/plan.json')
    base = history.parent.parent.parent
    plan = queue.load(plan_path)
    if plan.get('schema') != 'voices_v3_E1_E2_queue_v1' or plan.get('dry_run'):
        raise ValueError('An actual, compatible run plan is required')
    if plan.get('analysis_mode') != 'defer':
        raise ValueError('Only a --analysis-mode defer queue may have a concurrent CPU watcher')
    frozen = {
        str(plan_path): queue.sha(plan_path),
        str(base/'execution_protocol.txt'): plan['execution_protocol_sha256'],
        str(base/'setup/runtime_decision.json'): plan['runtime_decision_sha256'],
        str(base/'setup/download_queue.json'): plan['queue_sha256'],
        str(Path(queue.__file__).resolve()): plan['driver_sha256'],
    }
    for script in ANALYSIS_SCRIPTS:
        path = base/'code'/script
        frozen[str(path)] = queue.sha(path)
    # Counts imports analyze_v3, so both programs are bound even for counts-only plans.
    decision = queue.runtime(base)
    for key, name in [('inference_sha256', 'inference_v3.py'), ('loader_sha256', 'loader_v3.py')]:
        frozen[str(base/'code'/name)] = decision[key]
    models = {model['repo_id']: model for model in plan['selected_models']}
    if len(models) != len(plan['selected_models']):
        raise ValueError('Duplicate selected model identity')
    for model in models.values():
        frozen[model['calibration']] = model['calibration_sha256']
        e0 = base/'results'/model['slug']/'e0'
        frozen[str(e0/'e0_driver_state.json')] = model['E0_state_sha256']
        analysis_path = model.get('E0_analysis_path', str(e0/model['remedy']/'development_E0_analysis.json'))
        if (Path(analysis_path).resolve().parent != (e0/model['remedy']).resolve()
                or Path(analysis_path).name not in {'development_E0_analysis.json', 'development_E0_conditional_analysis.json'}):
            raise ValueError('Planned E0 analysis is outside the selected remedy development run')
        frozen[analysis_path] = model['E0_analysis_sha256']
        archive = model.get('E0_original_protocol_verified_archive')
        if archive:
            frozen[archive] = model['E0_execution_protocol_sha256']
    # Freeze selection files and verify their identities against the source split.
    # Their text is used only to select IDs, so validate IDs as well as file hashes.
    core_path = base/'data/core_data_manifest.json'
    core = queue.load(core_path)
    frozen[str(core_path)] = queue.sha(core_path)
    selections = {}
    for name in ('holdout200', 'semantic64', 'remaining180', 'semantic_remaining44'):
        path = base/'data'/('e1_'+name+'_items.json')
        items = queue.load(path)
        ids = {item['item_id'] for item in items}
        if len(ids) != len(items):
            raise ValueError('Duplicate item in frozen selection: '+str(path))
        selections[name] = ids
        frozen[str(path)] = queue.sha(path)
    old = set(core['reused20_ids'])
    if (len(selections['holdout200']) != 200 or len(selections['semantic64']) != 64
            or len(selections['remaining180']) != 180 or len(selections['semantic_remaining44']) != 44
            or not old <= selections['semantic64'] <= selections['holdout200']
            or selections['holdout200']-old != selections['remaining180']
            or selections['semantic64']-old != selections['semantic_remaining44']
            or selections['remaining180'] != set(core['Qwen4_and_LFM_confirmatory_item_ids'])):
        raise ValueError('Frozen 200/64 and fresh 180/44 selection identities differ')
    seen = set()
    for job in plan['jobs']:
        key = job_key(job)
        if key in seen:
            raise ValueError('Duplicate planned score job: '+str(key))
        seen.add(key)
        model = models[job['model']]
        expected = queue.analysis_jobs(base, model, job)
        if len(expected) != len(job['analyses']):
            raise ValueError('Planned analysis count differs from the frozen queue helpers')
        for actual, wanted in zip(job['analyses'], expected):
            # Python may have been invoked through an equivalent symlink. Everything
            # after the interpreter must be the exact prespecified CPU analysis CLI.
            if (not same_path(actual['command'][0], sys.executable)
                    or actual['command'][1:] != wanted['command'][1:]
                    or any(actual[field] != wanted[field] for field in ('name', 'output', 'log', 'scope'))):
                raise ValueError('Analysis command differs from the prespecified CPU job')
            if '--keep-items' in actual['command']:
                path = actual['command'][actual['command'].index('--keep-items')+1]
                frozen[path] = queue.sha(path)
        spec = plan['inputs'][job['name']]
        if (not same_path(spec['path'], job['input']) or spec['sha256'] != job['input_sha256']
                or spec['n_rows'] != job['rows']):
            raise ValueError('Score job differs from planned input identity')
        frozen[job['input']] = job['input_sha256']
    for path, digest in frozen.items():
        if queue.sha(path) != digest:
            raise ValueError('Frozen source changed: '+path)
    return base, plan, decision, models, frozen


def verify_frozen(frozen, signatures):
    """Rehash changed files; immutable large prompt files need not be reread each poll."""
    for path, digest in frozen.items():
        current = fingerprint(path)
        if signatures.get(path) != current:
            if queue.sha(path) != digest:
                raise ValueError('Frozen source changed while watcher was running: '+path)
            signatures[path] = current


def queue_state(base, plan_path):
    """Prefer this plan's terminal snapshot over the shared, replaceable queue state."""
    for path in (plan_path.parent/'final_status.json', base/'setup/e1_e2_queue_status.json'):
        if not path.exists():
            continue
        value = queue.load(path)
        if same_path(value.get('plan', ''), plan_path):
            return value, str(path)
        if path.name == 'final_status.json':
            raise ValueError('History terminal snapshot belongs to a different plan')
    return None, None


def queue_live(base, history):
    lease = base/'setup/e1_e2_queue.lock'
    if not lease.exists():
        return False
    value = queue.load(lease)
    return same_path(value.get('history', ''), history) and pid_alive(value['pid'])


def terminal_entries(plan_path, state):
    """History remains useful if another queue has replaced the shared status file."""
    entries = {}
    events = plan_path.parent/'events.jsonl'
    if events.exists():
        with events.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue  # An in-progress last append is not a closed event.
                if item.get('event') == 'job_finished':
                    entries[job_key(item)] = item
    if state:
        entries.update({job_key(item): item for item in state.get('jobs', [])})
    return entries


def analysis_closure(task):
    closure = queue.analysis_closed(task)
    if closure is None:
        return None
    result = queue.load(task['output'])
    command = task['command']
    argument = lambda flag: command[command.index(flag)+1] if flag in command else None
    required = [command[1], argument('--inputs'), argument('--results')]
    if argument('--keep-items'):
        required.append(argument('--keep-items'))
    bindings = {str(Path(path).resolve()): digest for path, digest in result['file_sha256'].items()}
    if any(str(Path(path).resolve()) not in bindings for path in required):
        raise ValueError('Existing analysis omits an expected source binding')
    count_program = Path(command[1]).name == 'analyze_v3_counts.py'
    # The counts program currently does not emit a readout field; its CLI here
    # is exactly reconstructed from the frozen R0-only count job above.
    if result.get('readout', argument('--readout') if count_program else None) != argument('--readout'):
        raise ValueError('Existing analysis has a different readout')
    if not count_program:
        expected = {'keep_items': argument('--keep-items'), 'exclude_items': None,
                    'filters': json.loads(argument('--filters') or '{}')}
        if result.get('selection') != expected:
            raise ValueError('Existing analysis has a different item or condition selection')
    return closure


def watch(plan_path, poll_seconds=15, missing_queue_grace=90):
    plan_path = Path(plan_path).resolve()
    base, plan, decision, models, frozen = prepare(plan_path)
    history = plan_path.parent
    lease = history/'analysis_watcher.lock'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if lease.exists():
        owner = queue.load(lease)
        if pid_alive(owner['pid']):
            raise RuntimeError('This plan already has a CPU watcher: '+str(owner['pid']))
        lease.replace(history/('analysis_watcher_stale_lock_'+stamp+'.json'))
    with lease.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'plan': str(plan_path), 'started_utc': queue.now()}, stream)
    status_path = history/'analysis_watcher_status.json'
    events_path = history/'analysis_watcher_events.jsonl'
    if status_path.exists():
        status_path.replace(history/('analysis_watcher_previous_'+stamp+'.json'))
    state = {
        'schema': 'voices_v3_CPU_analysis_watcher_v1', 'status': 'running',
        'started_utc': queue.now(), 'pid': os.getpid(), 'plan': str(plan_path),
        'plan_sha256': frozen[str(plan_path)], 'watcher_sha256': queue.sha(__file__),
        'frozen_file_sha256': frozen, 'cpu_workers': 1, 'cpu_thread_environment': CPU_ENV,
        'pending_models': plan['pending_models'], 'jobs': [], 'current_analysis': None,
        'no_model_disqualification_from_execution_errors': True,
    }
    records = {}
    for job in plan['jobs']:
        record = {field: job[field] for field in ('model', 'stage', 'name', 'output')}
        record.update(status='waiting_for_closed_score', analyses=[])
        records[job_key(job)] = record
        state['jobs'].append(record)
    signatures = {path: fingerprint(path) for path in frozen}
    absent_since = None
    os.environ.update(CPU_ENV)
    # Hide accelerators from analysis children, in addition to using CPU-only CLIs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        queue.write(status_path, state)
        queue.event(events_path, {'event': 'watcher_started', 'plan': str(plan_path), 'pid': os.getpid()})
        while True:
            verify_frozen(frozen, signatures)
            producer, producer_path = queue_state(base, plan_path)
            live = queue_live(base, history)
            if live:
                absent_since = None
            elif absent_since is None:
                absent_since = time.monotonic()
            terminal = bool(producer and producer.get('status') == 'finished')
            gone = not live and time.monotonic()-(absent_since or time.monotonic()) >= missing_queue_grace
            stopped = terminal or gone
            state['queue_state_source'] = producer_path
            state['queue_status'] = producer.get('status') if producer else 'not_observed'
            state['queue_process_live'] = live
            state['queue_stop_reason'] = ('normal_terminal_snapshot' if terminal else
                                          'queue_process_missing_or_superseded' if gone else None)
            producer_entries = terminal_entries(plan_path, producer)
            for job in plan['jobs']:
                record = records[job_key(job)]
                if record['status'] in ('complete', 'analysis_error_not_model_disqualification',
                                       'score_identity_error_not_model_disqualification',
                                       'unscored_pending_at_queue_end', 'unscored_execution_error'):
                    continue
                try:
                    closure = queue.score_closure(job, models[job['model']], decision)
                except Exception as ex:
                    record.update(status=('score_identity_error_not_model_disqualification' if stopped else
                                          'waiting_for_stable_score_closure'), error=str(ex))
                    continue
                if closure is None:
                    if stopped:
                        origin = producer_entries.get(job_key(job), {})
                        record.update(status=('unscored_execution_error' if 'error' in origin.get('status', '') else
                                              'unscored_pending_at_queue_end'), queue_job_status=origin.get('status'),
                                      error=origin.get('error'), score_summary_available=False)
                    continue
                record.update(closure)
                record['status'] = 'analyzing_closed_score'
                record.pop('error', None)
                record['analyses'] = []
                for task in job['analyses']:
                    verify_frozen(frozen, signatures)
                    result = {field: task[field] for field in ('name', 'output', 'scope', 'command', 'log')}
                    result['started_utc'] = queue.now()
                    record['analyses'].append(result)
                    state['current_analysis'] = {'model': job['model'], 'score': job['name'], 'analysis': task['name']}
                    queue.write(status_path, state)
                    try:
                        complete = analysis_closure(task)
                        if complete is None:
                            queue.event(events_path, {'event': 'analysis_start', **state['current_analysis'], 'command': task['command']})
                            print('ANALYZE '+job['model']+' '+task['name'], flush=True)
                            result['exit_code'] = queue.run_command(task['command'], task['log'])
                            if result['exit_code']:
                                raise RuntimeError('CPU analysis exited '+str(result['exit_code'])+'; output and log retained')
                            verify_frozen(frozen, signatures)
                            complete = analysis_closure(task)
                            if complete is None:
                                raise ValueError('CPU analysis returned without a bound result')
                        result.update(complete)
                    except Exception as ex:
                        result.update(status='analysis_error_not_model_disqualification', error=str(ex))
                    result['finished_utc'] = queue.now()
                    queue.event(events_path, {'event': 'analysis_finished', 'model': job['model'], **result})
                    queue.write(status_path, state)
                record['status'] = ('complete' if all(item['status'] == 'complete' for item in record['analyses']) else
                                    'analysis_error_not_model_disqualification')
                record['finished_utc'] = queue.now()
                state['current_analysis'] = None
                queue.write(status_path, state)
            state['last_poll_utc'] = queue.now()
            all_resolved = all(record['status'] in (
                'complete', 'analysis_error_not_model_disqualification',
                'score_identity_error_not_model_disqualification',
                'unscored_pending_at_queue_end', 'unscored_execution_error') for record in state['jobs'])
            if all_resolved:
                all_complete = not plan['pending_models'] and all(record['status'] == 'complete' for record in state['jobs'])
                state.update(status='complete' if all_complete else 'finished_with_pending_or_errors',
                             all_requested_analyses_complete=all_complete, finished_utc=queue.now())
                break
            queue.write(status_path, state)
            time.sleep(poll_seconds)
    except BaseException as ex:
        state.update(status='watcher_error_not_model_disqualification', error=str(ex), finished_utc=queue.now())
        raise
    finally:
        queue.write(status_path, state)
        queue.event(events_path, {'event': 'watcher_stopped', 'status': state['status'], 'plan': str(plan_path)})
        if lease.exists() and queue.load(lease).get('pid') == os.getpid():
            lease.unlink()
    print(json.dumps({'status': state['status'], 'state': str(status_path)}, ensure_ascii=False), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--missing-queue-grace', type=float, default=90)
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 20:
        parser.error('--poll-seconds must be between 10 and 20 seconds')
    if args.missing_queue_grace < 2*args.poll_seconds:
        parser.error('--missing-queue-grace must allow at least two polls')
    watch(args.plan, args.poll_seconds, args.missing_queue_grace)


if __name__ == '__main__':
    main()
