"""CPU-only E4 point-analysis watcher for one explicit persisted scoring plan.

No model scoring, CUDA work or dynamic MC is launched. --postprocess optionally
adds six-topic aggregation and the prespecified rule supplement, one CPU process
at a time. Missing producer processes receive a bounded grace period.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import run_analysis_watcher_v3 as shared
import run_e1_e2_queue as queue


def argument(command, flag):
    return command[command.index(flag)+1]


def point_path(job):
    return Path(job['deferred_analysis']['output_dir'])/(job['topic_id']+'_E4_analysis.json')


def prepare(plan_path, postprocess=False):
    plan_path = Path(plan_path).resolve(); history = plan_path.parent
    if (plan_path.name != 'plan.json' or history.parent.name != 'e4_score_queue_history'
            or history.parent.parent.name != 'setup'):
        raise ValueError('Use the exact setup/e4_score_queue_history/<run>/plan.json')
    base = history.parent.parent.parent; plan = queue.load(plan_path)
    if (plan.get('schema') != 'voices_v3_E4_score_queue_v1' or plan.get('dry_run')
            or plan.get('CPU_analysis_mode') != 'commands_only_deferred' or plan.get('MC_requested') is not False):
        raise ValueError('An actual E4 scoring plan with deferred non-MC analyses is required')
    frozen = dict(plan['frozen_file_sha256']); frozen[str(plan_path)] = queue.sha(plan_path)
    frozen[plan['selection']] = plan['selection_sha256']
    selection = queue.load(plan['selection'])
    models = {model['repo_id']: model for model in plan['selected_models']}
    if (len(models) != 6 or len(plan['selected_models']) != 6 or len(plan['selected_repos']) != 6
            or set(models) != set(plan['selected_repos']) or selection.get('status') != 'frozen'
            or selection.get('selected_repos') != plan['selected_repos']):
        raise ValueError('Plan does not match its explicit frozen six-model selection')
    decision = queue.runtime(base)
    if decision != plan['runtime_decision']:
        raise ValueError('Runtime decision differs from the scoring plan')
    manifest = queue.load(base/'data/e4_cache_manifest.json')
    topics = {topic['topic_id']: topic for topic in manifest['topics']}
    if len(topics) != 6 or len(plan['jobs']) != 36 or plan['n_score_jobs'] != 36 or plan['total_score_rows'] != 1764720:
        raise ValueError('Expected exactly six models by six complete topic score jobs')
    seen = set()
    for job in plan['jobs']:
        model = models[job['model']]; tid = job['topic_id']; spec = topics[tid]['output']; key = (job['model'], tid)
        if key in seen: raise ValueError('Duplicate model/topic score job')
        seen.add(key)
        if (job['readout'] != 'R0' or job['rows'] != 49020 or job['input_sha256'] != spec['sha256']
                or job['stage'] != 'E4' or job['name'] != tid
                or not shared.same_path(job['input'], base/spec['path'])
                or not shared.same_path(job['output'], base/'results'/model['slug']/'e4_cache'/(tid+'_R0.jsonl'))):
            raise ValueError('E4 job differs from the frozen R0 topic cache')
        target = job['deferred_analysis']; command = target['command']
        output_dir = base/'results'/model['slug']/'e4_analysis'/history.name/tid
        expected = [str(base/'code/run_e4_analysis.py'), '--inputs', job['input'], '--results', job['output'],
                    '--kernel-analysis', model['kernel_analysis'], '--cache-manifest', str(base/'data/e4_cache_manifest.json'),
                    '--sts', str(base/'data/e4_graphs/sts15_all80.json'), '--topic-id', tid,
                    '--output-dir', str(output_dir), '--mc-draws', '0']
        if (not shared.same_path(command[0], sys.executable) or command[1:] != expected
                or not shared.same_path(target['output_dir'], output_dir)
                or not shared.same_path(target['requires_closed_score'], job['output'])
                or target['kernel_analysis_sha256'] != model['kernel_analysis_sha256']):
            raise ValueError('Only the exact prespecified CPU single-round command may execute')
        frozen[job['input']] = job['input_sha256']
        frozen[model['kernel_analysis']] = model['kernel_analysis_sha256']
        frozen[model['calibration']] = model['calibration_sha256']
    if seen != {(repo, tid) for repo in models for tid in topics}:
        raise ValueError('E4 model/topic panel is incomplete')
    # The scoring plan already freezes the point analyzer and its simulation sources.
    for path in (Path(__file__).resolve(), Path(queue.__file__).resolve(), Path(shared.__file__).resolve()):
        frozen.setdefault(str(path), queue.sha(path))
    if postprocess:
        for name in ('code/aggregate_e4_v3.py', 'code/supplement_e4_rules_v3.py', 'code/e4_rule_baselines_v3.py',
                     'provenance/E4_pairwise_rule_supplement_pre_E4_20261031.txt'):
            frozen.setdefault(str(base/name), queue.sha(base/name))
    for path, digest in frozen.items():
        if queue.sha(path) != digest: raise ValueError('Frozen E4 source changed: '+path)
    return base, plan, models, decision, frozen


def verify_bindings(bindings, required=(), known=None):
    if not bindings: raise ValueError('Analysis lacks source hashes')
    known = known or {}; seen = set()
    for path, digest in bindings.items():
        resolved = str(Path(path).resolve()); seen.add(resolved)
        actual = known[resolved] if resolved in known else queue.sha(path)
        if actual != digest: raise ValueError('Analysis source changed: '+path)
    if any(str(Path(path).resolve()) not in seen for path in required):
        raise ValueError('Analysis omits a required source binding')


def topic_closure(job, model, decision, score_closure=None):
    """Verify point JSON/NPZ, source hashes and exact E1/E4 execution identities.

    The optional score_closure must be freshly verified by queue.score_closure;
    it avoids hashing the same closed raw file twice within one operation.
    """
    path = point_path(job)
    if not path.exists(): return None
    closure = score_closure if score_closure is not None else queue.score_closure(job, model, decision)
    if closure is None: raise ValueError('Point analysis exists without a closed planned score')
    result = queue.load(path); command = job['deferred_analysis']['command']
    kernel = queue.load(model['kernel_analysis']); manifest = queue.load(argument(command, '--cache-manifest'))
    meta = next(topic for topic in manifest['topics'] if topic['topic_id'] == job['topic_id'])
    if (kernel.get('readout') != 'R0' or result.get('topic_id') != job['topic_id'] or result.get('topic') != meta['topic']
            or result.get('MC_completed') is not False or result.get('E1_topic_disjoint') is not True
            or result['option_mass']['n_rows'] != job['rows'] or result['topic'] in kernel['topics']
            or result['E1_execution_identity'] != kernel['execution_identity']
            or result['execution_identity']['critical_identity'] != kernel['execution_identity']['critical_identity']):
        raise ValueError('Point analysis topic, scope or E1 identity differs from the plan')
    if (result['E4a']['n_conditions'] != 70 or len(result['E4a']['records']) != 70
            or result['E4b']['n_systems'] != 80 or result['E4b']['n_mappings'] != 20
            or len(result['E4b']['configurations']) != 6
            or result['E4c']['n_conditions'] != 192 or len(result['E4c']['records']) != 192):
        raise ValueError('Point analysis is not the complete G6/80-system/three-five-set design')
    runs = result['execution_identity']['runs']
    if (len(runs) != 1 or not shared.same_path(runs[0]['path'], job['output'])
            or runs[0]['run_sha256'] != closure['run_sha256'] or runs[0]['engine'] != model['engine']
            or runs[0]['batch_size'] != decision['batch_size'] or runs[0]['calibration_sha256'] != model['calibration_sha256']):
        raise ValueError('Point analysis raw run/engine/calibration identity differs')
    known = {str(Path(job['output']).resolve()): closure['output_sha256']}
    verify_bindings(result['input_hashes'], [job['input'], job['output'], model['kernel_analysis'], command[1],
                    argument(command, '--cache-manifest'), argument(command, '--sts')], known)
    artifact = result['cache_artifact']; cache = path.parent/artifact['filename']
    if artifact['filename'] != job['topic_id']+'_caches.npz' or queue.sha(cache) != artifact['sha256']:
        raise ValueError('Compiled point cache changed or has an unexpected path')
    return {'status': 'complete', 'analysis_path': str(path), 'analysis_sha256': queue.sha(path),
            'cache_path': str(cache), 'cache_sha256': artifact['sha256'], 'score_closure': closure,
            'model': job['model'], 'topic_id': job['topic_id'], 'MC_completed': False}


def model_tasks(base, history, model, jobs):
    directory = history/'cpu_model_summaries'/model['slug']
    topics = [str(point_path(job)) for job in sorted(jobs, key=lambda job: job['topic_id'])]
    topic_args = [value for path in topics for value in ('--topics', path)]
    common = ['--cache-manifest', str(base/'data/e4_cache_manifest.json'),
              '--sts', str(base/'data/e4_graphs/sts15_all80.json'), '--draws', '10000']
    aggregate = directory/'single_round_aggregate.json'; supplement = directory/'pairwise_rule_supplement'
    return [
        {'kind': 'aggregate', 'output': str(aggregate), 'topics': topics, 'log': str(aggregate)+'.log',
         'command': [sys.executable, str(base/'code/aggregate_e4_v3.py'), *topic_args,
                     '--kernel-analysis', model['kernel_analysis'], '--output', str(aggregate), *common]},
        {'kind': 'supplement', 'output': str(supplement)+'.json', 'topics': topics, 'log': str(supplement)+'.log',
         'command': [sys.executable, str(base/'code/supplement_e4_rules_v3.py'), *topic_args,
                     '--e1-analysis', model['kernel_analysis'], '--output-prefix', str(supplement), *common]}]


def model_closure(task, model):
    path = Path(task['output'])
    if not path.exists(): return None
    result = queue.load(path); kernel = queue.load(model['kernel_analysis']); command = task['command']
    ids = {queue.load(topic)['topic_id'] for topic in task['topics']}
    if (result.get('n_topics') != 6 or result.get('planned_six_topics_complete') is not True
            or len(result['topic_ids']) != 6 or set(result['topic_ids']) != ids
            or result['bootstrap']['draws'] != 10000 or result['bootstrap']['unit'] != 'topic'):
        raise ValueError('Model summary is not the complete planned six-topic bootstrap')
    verify_bindings(result['file_sha256'], [*task['topics'], model['kernel_analysis'], command[1],
                    argument(command, '--cache-manifest'), argument(command, '--sts')])
    if task['kind'] == 'aggregate':
        if (result['model_identity'] != kernel['execution_identity']['critical_identity']
                or result.get('single_round_E1_kernel_uncertainty_propagated') is not True
                or result['dynamic']['status'] != 'not_run'):
            raise ValueError('Aggregate model identity/single-round scope differs')
        artifact = result['bootstrap_artifact']; expected = path.stem+'_bootstrap.npz'
    else:
        if (result.get('schema') != 'voices_v3_E4_single_round_pairwise_rule_supplement_v1'
                or result['repo_id'] != model['repo_id'] or result['E1_analysis_sha256'] != model['kernel_analysis_sha256']
                or result.get('dynamic_rules_added') is not False or result.get('new_LLM_calls') != 0):
            raise ValueError('Supplement model identity/single-round scope differs')
        artifact = result['outcomes_and_bootstrap_artifact']; expected = path.stem+'_outcomes_and_bootstrap.npz'
        if not path.with_suffix('.md').is_file(): raise ValueError('Supplement Markdown output is missing')
    if artifact['filename'] != expected or queue.sha(path.parent/expected) != artifact['sha256']:
        raise ValueError('Model bootstrap artifact changed')
    return {'status': 'complete', 'output': str(path), 'output_sha256': queue.sha(path),
            'artifact': str(path.parent/expected), 'artifact_sha256': artifact['sha256'], 'n_topics': 6}


def producer_state(base, plan_path):
    for path in (plan_path.parent/'final_status.json', base/'setup/e4_score_queue_status.json'):
        if not path.exists(): continue
        state = queue.load(path)
        if shared.same_path(state.get('plan', ''), plan_path): return state, str(path)
        if path.name == 'final_status.json': raise ValueError('Score final receipt belongs to another plan')
    return None, None


def watch(plan_path, postprocess=False, poll_seconds=15, missing_producer_grace=90):
    plan_path = Path(plan_path).resolve(); base, plan, models, decision, frozen = prepare(plan_path, postprocess)
    history = plan_path.parent; lease = history/'analysis_watcher.lock'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if lease.exists():
        owner = queue.load(lease)
        if shared.pid_alive(owner['pid']): raise RuntimeError('An E4 CPU watcher is already active for this plan')
        lease.replace(history/('analysis_watcher_stale_lock_'+stamp+'.json'))
    with lease.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'plan': str(plan_path), 'started_utc': queue.now()}, stream)
    groups = {repo: [job for job in plan['jobs'] if job['model'] == repo] for repo in models}
    tasks = {repo: model_tasks(base, history, model, groups[repo]) for repo, model in models.items()}
    attempt = history/'analysis_watcher_runs'/stamp
    config = {'score_plan': str(plan_path), 'score_plan_sha256': queue.sha(plan_path), 'postprocess': postprocess,
              'model_tasks': tasks, 'frozen_file_sha256': frozen, 'cpu_workers': 1, 'cpu_environment': shared.CPU_ENV}
    queue.write(attempt/'plan.json', config)
    status_path = history/'analysis_watcher_status.json'; events = history/'analysis_watcher_events.jsonl'
    if status_path.exists(): status_path.replace(attempt/'previous_status.json')
    state = {'schema': 'voices_v3_E4_CPU_watcher_v1', 'status': 'running', 'started_utc': queue.now(), 'pid': os.getpid(),
             'plan': str(plan_path), 'plan_sha256': queue.sha(plan_path), 'watcher_plan': str(attempt/'plan.json'),
             'frozen_file_sha256': frozen, 'postprocess_requested': postprocess, 'CPU_workers': 1,
             'GPU_used': False, 'dynamic_MC_requested': False, 'current_analysis': None, 'jobs': [], 'model_summaries': []}
    records = {}
    for job in plan['jobs']:
        record = {'model': job['model'], 'topic_id': job['topic_id'], 'status': 'waiting_for_closed_score',
                  'analysis_path': str(point_path(job)), 'command': job['deferred_analysis']['command']}
        records[(job['model'], job['topic_id'])] = record; state['jobs'].append(record)
    summaries = {}
    for repo in models:
        record = {'model': repo, 'status': 'waiting_for_six_topics', 'tasks': tasks[repo]}
        summaries[repo] = record; state['model_summaries'].append(record)
    done = {'complete', 'analysis_error_not_model_disqualification', 'score_identity_error_not_model_disqualification',
            'unscored_pending_at_queue_end', 'unscored_execution_error'}
    signatures = {}; absent = None
    os.environ.update(shared.CPU_ENV); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        while True:
            shared.verify_frozen(frozen, signatures)
            producer, source = producer_state(base, plan_path)
            live = shared.queue_live(base, history)
            absent = None if live else time.monotonic() if absent is None else absent
            terminal = bool(producer and producer.get('status') == 'finished_score_scope')
            gone = absent is not None and time.monotonic()-absent >= missing_producer_grace
            stopped = terminal or gone
            state.update(queue_state_source=source, queue_process_live=live,
                         queue_stop_reason='normal_terminal_snapshot' if terminal else 'producer_missing_or_superseded' if gone else None)
            failures = {(item['model'], item['topic_id']): item for item in (producer or {}).get('jobs', [])}
            for job in plan['jobs']:
                record = records[(job['model'], job['topic_id'])]
                if record['status'] in done: continue
                try: closure = queue.score_closure(job, models[job['model']], decision)
                except Exception as error:
                    record.update(status='score_identity_error_not_model_disqualification' if stopped else 'waiting_for_stable_score_closure', error=str(error)); continue
                if closure is None:
                    if stopped:
                        origin = failures.get((job['model'], job['topic_id']), {})
                        record.update(status='unscored_execution_error' if 'error' in origin.get('status', '') else 'unscored_pending_at_queue_end',
                                      queue_job_status=origin.get('status'), error=origin.get('error'))
                    continue
                state['current_analysis'] = {'model': job['model'], 'topic_id': job['topic_id']}
                record['started_utc'] = queue.now(); queue.write(status_path, state)
                try:
                    closed = topic_closure(job, models[job['model']], decision, closure)
                    if closed is None:
                        command = job['deferred_analysis']['command']; log = str(point_path(job))+'.log'
                        queue.event(events, {'event': 'topic_analysis_start', **state['current_analysis'], 'command': command})
                        record['exit_code'] = queue.run_command(command, log)
                        if record['exit_code']: raise RuntimeError('CPU point analysis exited '+str(record['exit_code'])+'; partial artifacts retained')
                        shared.verify_frozen(frozen, signatures)
                        closed = topic_closure(job, models[job['model']], decision, closure)
                        if closed is None: raise ValueError('CPU analysis returned without a closed point result')
                    record.update(closed)
                except Exception as error:
                    record.update(status='analysis_error_not_model_disqualification', error=str(error))
                record['finished_utc'] = queue.now(); state['current_analysis'] = None
                queue.event(events, {'event': 'topic_analysis_finished', **record}); queue.write(status_path, state)
            for repo, record in summaries.items():
                if record['status'] != 'waiting_for_six_topics': continue
                topic_records = [records[(repo, job['topic_id'])] for job in groups[repo]]
                if not all(item['status'] == 'complete' for item in topic_records):
                    if all(item['status'] in done for item in topic_records): record['status'] = 'pending_incomplete_topic_panel'
                    continue
                if not postprocess:
                    record['status'] = 'commands_saved_not_executed'; continue
                record['results'] = []
                for task in tasks[repo]:
                    shared.verify_frozen(frozen, signatures)
                    current = {'kind': task['kind'], 'output': task['output'], 'command': task['command']}
                    record['results'].append(current); state['current_analysis'] = {'model': repo, 'kind': task['kind']}
                    queue.write(status_path, state)
                    try:
                        # Recheck topic/cache closure before any six-topic aggregation.
                        for job in groups[repo]:
                            if topic_closure(job, models[repo], decision) is None:
                                raise ValueError('A previously completed point result is now missing')
                        closed = model_closure(task, models[repo])
                        if closed is None:
                            Path(task['output']).parent.mkdir(parents=True, exist_ok=True)
                            queue.event(events, {'event': 'model_analysis_start', **state['current_analysis'], 'command': task['command']})
                            current['exit_code'] = queue.run_command(task['command'], task['log'])
                            if current['exit_code']: raise RuntimeError('CPU model summary exited '+str(current['exit_code'])+'; artifacts retained')
                            shared.verify_frozen(frozen, signatures); closed = model_closure(task, models[repo])
                            if closed is None: raise ValueError('CPU summary returned without a bound result')
                        current.update(closed)
                    except Exception as error:
                        current.update(status='analysis_error_not_model_disqualification', error=str(error))
                    queue.event(events, {'event': 'model_analysis_finished', 'model': repo, **current})
                    state['current_analysis'] = None; queue.write(status_path, state)
                record['status'] = 'complete' if all(item['status'] == 'complete' for item in record['results']) else 'analysis_error_not_model_disqualification'
            state['last_poll_utc'] = queue.now()
            if all(item['status'] in done for item in state['jobs']):
                points_ok = all(item['status'] == 'complete' for item in state['jobs'])
                summaries_ok = not postprocess or all(item['status'] == 'complete' for item in state['model_summaries'])
                state.update(status='complete' if points_ok and summaries_ok else 'finished_with_pending_or_errors',
                             all_requested_point_analyses_complete=points_ok,
                             all_requested_postprocessing_complete=summaries_ok if postprocess else None,
                             finished_utc=queue.now()); break
            queue.write(status_path, state); time.sleep(poll_seconds)
    except BaseException as error:
        state.update(status='watcher_error_not_model_disqualification', error=str(error), finished_utc=queue.now()); raise
    finally:
        queue.write(status_path, state); queue.write(attempt/'final_status.json', state)
        queue.event(events, {'event': 'watcher_stopped', 'status': state['status'], 'plan': str(plan_path)})
        if lease.exists() and queue.load(lease).get('pid') == os.getpid(): lease.unlink()
    print(json.dumps({'status': state['status'], 'state': str(status_path)}), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--postprocess', action='store_true', help='Execute six-topic aggregate and rule supplement sequentially on CPU')
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--missing-producer-grace', type=float, default=90)
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 20 or args.missing_producer_grace < 2*args.poll_seconds:
        parser.error('Use 10–20-second polling and at least two polls of missing-producer grace')
    watch(args.plan, args.postprocess, args.poll_seconds, args.missing_producer_grace)


if __name__ == '__main__': main()
