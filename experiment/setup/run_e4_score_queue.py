"""Score E4 caches for an explicit frozen six-model selection; never select or run MC.

Actual scoring requires screen and the same exclusive GPU lease used by E1/E2.
CPU single-round analysis commands are saved, never executed. E2 readiness is an
operator prerequisite and is deliberately not inferred from the E1 selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import run_e1_e2_queue as queue

CODE = Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0, str(CODE))
import summarize_select_e1_v3 as selector


def source_path(value, base, roots=()):
    """Map an explicitly declared source project root, including Windows paths."""
    normalized = str(value).replace('\\', '/').rstrip('/')
    for root in [str(base), *roots, '/workspace/voices_nonadditivity_v3']:
        prefix = str(root).replace('\\', '/').rstrip('/')
        if normalized == prefix or normalized.startswith(prefix+'/'):
            return base/normalized[len(prefix):].lstrip('/')
    path = Path(value)
    if path.is_absolute():
        return path
    if ':' in normalized.split('/')[0]:
        raise ValueError('Unmapped source path; provide its project --source-root: '+str(value))
    return base/path


class Verifier(selector.Verifier):
    def __init__(self, root, source_roots):
        super().__init__(root, '/workspace/voices_nonadditivity_v3')
        self.source_roots = source_roots

    def path(self, value):
        return source_path(value, self.root, self.source_roots)


def prepare(base, selection_path, source_roots=()):
    frozen = queue.load(selection_path)
    if (frozen.get('schema') != 'voices_v3_frozen_E4_model_selection_v1'
            or frozen.get('status') != 'frozen' or frozen.get('performed') is not True
            or frozen.get('all_requested_E1_complete') is not True):
        raise ValueError('An explicitly frozen, complete E1 six-model selection is required')
    repos = frozen['selected_repos']
    if len(repos) != 6 or len(set(repos)) != 6:
        raise ValueError('Frozen E4 selection must contain exactly six unique models')
    verifier = Verifier(base, source_roots)
    panel_path = verifier.path(frozen['panel_path'])
    verifier.check(selection_path, queue.sha(selection_path))
    verifier.check(panel_path, frozen['panel_sha256'])
    verifier.check(base/'code/summarize_select_e1_v3.py', frozen['selector_sha256'])
    verifier.check(Path(selector.__file__).resolve(), frozen['selector_sha256'])
    verifier.check(base/'execution_protocol.txt', frozen['protocol_sha256'])
    panel = queue.load(panel_path)
    if (panel.get('schema') != 'voices_v3_E1_panel_and_E4_selection_v1'
            or panel.get('all_requested_E1_complete') is not True
            or panel['script_sha256'] != frozen['selector_sha256']
            or panel['protocol_sha256'] != frozen['protocol_sha256']
            or any(frozen.get(key) != value for key, value in panel['E4_selection'].items())):
        raise ValueError('Frozen selection differs from its bound E1 panel')
    verifier.check(base/'setup/download_queue.json', panel['queue_sha256'])
    verifier.check(base/'code/summarize_e0_panel_v3.py', panel['E0_summarizer_sha256'])
    verifier.check(Path(selector.e0.__file__).resolve(), panel['E0_summarizer_sha256'])
    if not panel.get('verified_source_sha256'):
        raise ValueError('Frozen E1 panel lacks verified source bindings')
    for path, digest in panel['verified_source_sha256'].items():
        verifier.check(verifier.path(path), digest)
    decision = queue.runtime(base)
    for path in (base/'setup/runtime_decision.json', base/'code/inference_v3.py',
                 base/'code/loader_v3.py', Path(queue.__file__).resolve(), Path(__file__).resolve()):
        verifier.check(path, queue.sha(path))
    rows = {row['repo_id']: row for row in panel['models']}
    checkpoints = {item['repo_id']: item for item in queue.load(base/'setup/download_queue.json')}
    models = []
    for repo in repos:
        row = rows[repo]
        if not row.get('E1_complete') or not row.get('E4_eligible') or not row.get('R0_operational'):
            raise ValueError('Selected model lacks verified R0 E4 eligibility: '+repo)
        model, reason = queue.selected_model(base, checkpoints[repo], decision)
        if reason:
            raise ValueError(repo+': '+reason+'; no model download or substitution is performed')
        if (not model['R0_operational'] or row['revision'] != model['revision']
                or row['calibration_sha256'] != model['calibration_sha256']
                or row['E0_state_sha256'] != model['E0_state_sha256']
                or row['E0_analysis_sha256'] != model['E0_analysis_sha256']):
            raise ValueError('Selected E0 identity differs from frozen E1 panel: '+repo)
        e0_dir = base/'results'/model['slug']/'e0'
        verifier.check(Path(model['calibration']), model['calibration_sha256'])
        verifier.check(e0_dir/'e0_driver_state.json', model['E0_state_sha256'])
        verifier.check(Path(model['E0_analysis_path']), model['E0_analysis_sha256'])
        if model.get('E0_original_protocol_verified_archive'):
            verifier.check(Path(model['E0_original_protocol_verified_archive']), model['E0_execution_protocol_sha256'])
        primary_job = next(job for job in selector.expected_jobs(base, repo, True) if job['role'] == 'primary')
        primary = row['primary']; kernel = verifier.path(primary['path'])
        expected = base/'results'/model['slug']/'e1'/(primary_job['name']+'.json')
        if kernel.resolve() != expected.resolve():
            raise ValueError('Wrong E1 primary kernel source: '+repo)
        verifier.check(kernel, primary['sha256'])
        calibration = {'value': queue.load(model['calibration']), 'sha256': model['calibration_sha256']}
        analysis = selector.verify_e1(kernel, primary_job, repo, calibration, verifier)
        count = 180 if repo in queue.PREVIOUS20_MODELS else 200
        if (analysis['n_items'] != count or primary['n_items'] != count or primary['readout'] != 'R0'
                or primary['C'] != analysis['point']['C']
                or not selector.finite(analysis['point']['C'])
                or primary['execution_identity'] != analysis['execution_identity']['critical_identity']):
            raise ValueError('Primary fresh180/full200 R0 kernel binding differs: '+repo)
        model.update(kernel_analysis=str(kernel), kernel_analysis_sha256=primary['sha256'],
                     kernel_items=count, E1_topics=analysis['topics'])
        models.append(model)
    manifest_path = base/'data/e4_cache_manifest.json'; manifest = queue.load(manifest_path)
    verifier.check(manifest_path, queue.sha(manifest_path))
    for path, key in [('data/e4_agents.json', 'agents_sha256'),
                      ('data/e4_agent_manifest.json', 'source_agent_manifest_sha256'),
                      ('code/build_v3_data.py', 'builder_sha256')]:
        verifier.check(base/path, manifest[key])
    topics = manifest['topics']
    if (manifest['n_topics'] != 6 or len(topics) != 6 or len({topic['topic_id'] for topic in topics}) != 6
            or manifest['n_rows'] != 294120):
        raise ValueError('E4 manifest does not contain the frozen six-topic cache')
    for topic in topics:
        spec = topic['output']; validation = topic['validation']
        if spec['n_rows'] != 49020 or validation['status'] != 'pass' or validation['rows'] != 49020:
            raise ValueError('Incomplete E4 topic manifest: '+topic['topic_id'])
        path = base/spec['path']; verifier.check(path, spec['sha256'])
        if path.stat().st_size != spec['bytes']:
            raise ValueError('E4 input byte count differs: '+str(path))
        if any(topic['topic'] in model['E1_topics'] for model in models):
            raise ValueError('E1/E4 topic overlap: '+topic['topic'])
    for path in ['data/e4_graphs/sts15_all80.json', 'code/run_e4_analysis.py',
                 'code/e4_graphs.py', 'code/e4_simulation.py', 'code/analyze_v3.py']:
        verifier.check(base/path, queue.sha(base/path))
    return frozen, models, topics, decision, verifier.verified


def build_jobs(base, models, topics, decision, stamp):
    jobs = []
    for model in models:
        for topic in topics:
            tid = topic['topic_id']; spec = {**topic['output'], 'path': base/topic['output']['path']}
            job = queue.score_job(base, model, decision, spec, tid, 'E4')
            output = base/'results'/model['slug']/'e4_cache'/(tid+'_R0.jsonl')
            job['command'][job['command'].index('--output')+1] = str(output)
            job.update(output=str(output), log=str(output)+'.log', topic_id=tid)
            directory = base/'results'/model['slug']/'e4_analysis'/stamp/tid
            if directory.exists():
                raise ValueError('Deferred CPU analysis requires a new output directory: '+str(directory))
            command = [sys.executable, str(base/'code/run_e4_analysis.py'), '--inputs', job['input'],
                       '--results', str(output), '--kernel-analysis', model['kernel_analysis'],
                       '--cache-manifest', str(base/'data/e4_cache_manifest.json'),
                       '--sts', str(base/'data/e4_graphs/sts15_all80.json'), '--topic-id', tid,
                       '--output-dir', str(directory), '--mc-draws', '0']
            job['deferred_analysis'] = {'status': 'command_only_not_executed', 'command': command,
                'output_dir': str(directory), 'kernel_analysis_sha256': model['kernel_analysis_sha256'],
                'scope': 'E4a_b_c_single_round_only_MC_not_requested', 'requires_closed_score': str(output)}
            jobs.append(job)
    return jobs


def active_gpu_work(base):
    pids = queue.existing_model_gpu_work(base)
    names = {str(base/'setup'/name) for name in (
        'run_e1_e2_queue.py', 'run_e4_score_queue.py', 'benchmark_e4_gpu.py')}
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name) == os.getpid(): continue
        try: arguments = (process/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if any(arg.decode(errors='ignore') in names for arg in arguments): pids.append(int(process.name))
    cuda = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    pids.extend(int(line.strip()) for line in cuda.stdout.splitlines() if line.strip().isdigit())
    return sorted(set(pids))


def execute(base, history, plan, models, decision):
    lease = base/'setup/e1_e2_queue.lock'  # Shared with the already-frozen E1/E2 queue.
    if lease.exists():
        owner = queue.load(lease)
        try: os.kill(int(owner['pid']), 0)
        except ProcessLookupError: lease.replace(history/'recovered_stale_GPU_lease.json')
        else: raise RuntimeError('Another E1/E2/E4 GPU queue holds the lease')
    if active_gpu_work(base): raise RuntimeError('Another inference controller or CUDA process is active')
    with lease.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'history': str(history), 'kind': 'E4_score_only'}, stream)
    state = {'status': 'running', 'started_utc': queue.now(), 'plan': str(history/'plan.json'),
             'selection': plan['selection'], 'jobs': [], 'CPU_analysis_executed': False, 'MC_executed': False}
    statepath = base/'setup/e4_score_queue_status.json'; events = history/'events.jsonl'
    byrepo = {model['repo_id']: model for model in models}; failed = set()
    signatures = {}  # Recheck once after acquiring the GPU lease, then only changed files.
    try:
        queue.write(statepath, state)
        for job in plan['jobs']:
            entry = {key: job[key] for key in ('model', 'stage', 'name', 'topic_id', 'output')}
            entry.update(started_utc=queue.now(), deferred_analysis=job['deferred_analysis'])
            state['current_job'] = entry.copy(); queue.write(statepath, state)
            if job['model'] in failed:
                entry['status'] = 'deferred_after_execution_error_not_disqualified'
            else:
                try:
                    for path, digest in plan['frozen_file_sha256'].items():
                        current = (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                        if current != signatures.get(path):
                            if queue.sha(path) != digest: raise ValueError('Frozen source changed: '+path)
                            signatures[path] = current
                    closure = queue.score_closure(job, byrepo[job['model']], decision)
                    if closure is None:
                        raw = Path(job['output']); entry['resume_existing_bytes'] = raw.stat().st_size if raw.exists() else 0
                        queue.event(events, {'event': 'score_start', **entry, 'command': job['command']})
                        print('START '+job['model']+' '+job['topic_id'], flush=True)
                        entry['exit_code'] = queue.run_command(job['command'], job['log'])
                        if entry['exit_code']: raise RuntimeError('Scoring exited '+str(entry['exit_code'])+'; partial raw/log/run retained')
                        closure = queue.score_closure(job, byrepo[job['model']], decision)
                        if closure is None: raise ValueError('Scoring returned without a closed summary')
                    entry.update(closure)
                except Exception as error:
                    entry.update(status='execution_error_not_model_disqualification', error=str(error)); failed.add(job['model'])
            entry['finished_utc'] = queue.now(); state['jobs'].append(entry)
            queue.event(events, {'event': 'job_finished', **entry}); queue.write(statepath, state)
        state.update(status='finished_score_scope', finished_utc=queue.now(), current_job=None,
                     all_requested_scores_complete=all(item['status'] == 'complete' for item in state['jobs']),
                     no_model_disqualification_from_execution_errors=True)
        queue.write(statepath, state); queue.write(history/'final_status.json', state)
    finally:
        if lease.exists() and queue.load(lease).get('pid') == os.getpid(): lease.unlink()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--source-root', action='append', default=[], help='Original project root embedded in a transferred frozen report')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); base = args.root.resolve(); selection_path = args.selection.resolve()
    frozen, models, topics, decision, sources = prepare(base, selection_path, args.source_root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history = base/'setup/e4_score_queue_history'/stamp
    jobs = build_jobs(base, models, topics, decision, stamp)
    plan = {'schema': 'voices_v3_E4_score_queue_v1', 'created_utc': queue.now(), 'dry_run': args.dry_run,
            'selection': str(selection_path), 'selection_sha256': queue.sha(selection_path),
            'selected_repos': frozen['selected_repos'], 'selected_models': models,
            'runtime_decision': decision, 'source_root_mappings': args.source_root,
            'frozen_file_sha256': sources, 'jobs': jobs, 'n_score_jobs': len(jobs),
            'total_score_rows': sum(job['rows'] for job in jobs), 'CPU_analysis_mode': 'commands_only_deferred',
            'E2_stage_readiness_checked_here': False, 'E2_stage_readiness_owner': 'caller',
            'model_selection_performed_here': False,
            'MC_requested': False, 'positive_direction': 'global_support; tau reversal belongs to downstream analysis'}
    queue.write(history/'plan.json', plan)
    if args.dry_run:
        print(json.dumps({'status': 'dry_run_only', 'jobs': len(jobs), 'rows': plan['total_score_rows'], 'plan': str(history/'plan.json')})); return
    if not os.environ.get('STY'): raise RuntimeError('E4 scoring and GPU subprocesses must run inside screen')
    state = execute(base, history, plan, models, decision)
    print(json.dumps({'status': state['status'], 'all_scores_complete': state['all_requested_scores_complete'], 'plan': state['plan']}))


if __name__ == '__main__':
    main()
