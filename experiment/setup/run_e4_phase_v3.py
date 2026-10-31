"""Explicit post-prerequisite E4 scoring -> CPU summaries -> dynamics -> report.

This coordinator never downloads models, changes scientific gates, or starts
before an explicitly supplied verified prerequisite report. Child queues own
the shared GPU lease. Partial results survive failures; an explicit closed
--score-plan can be reused on a later invocation. No SSH or API operations.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_e1_e2_queue as queue
import run_analysis_watcher_v3 as shared
import run_e4_score_queue as scoring
import run_e4_analysis_watcher_v3 as watching
import run_e4_dynamics_queue_v3 as dynamics

ROOT=Path(__file__).resolve().parents[1]
HISTORIES={'score':'e4_score_queue_history','dynamics':'e4_dynamics_queue_history'}
SCHEMAS={'score':'voices_v3_E4_score_queue_v1','dynamics':'voices_v3_E4_dynamics_queue_v1'}
SOURCE_FILES=('setup/run_e4_phase_v3.py','setup/run_e4_score_queue.py',
    'setup/run_e4_analysis_watcher_v3.py','setup/run_e4_dynamics_queue_v3.py',
    'setup/run_analysis_watcher_v3.py','setup/run_e1_e2_queue.py','code/write_e4_report_v3.py',
    'code/aggregate_e4_v3.py','code/supplement_e4_rules_v3.py','code/e4_rule_baselines_v3.py',
    'code/run_e4_analysis.py','code/run_e4_dynamics_v3.py','code/e4_dynamic_stratified.py',
    'code/e4_simulation.py','code/e4_simulation_torch.py','code/analyze_v3.py',
    'code/analyze_e4_cross_model_v3.py','provenance/E4_H3_operationalization_20261031.txt')


def same(left,right):return Path(left).resolve()==Path(right).resolve()


def verify_hashes(bindings):
    for path,digest in bindings.items():
        if queue.sha(path)!=digest:raise ValueError('Bound source changed: '+str(path))


def prerequisite(base,path,selection_path,source_roots=()):
    """Validate the report's stated scope and hashes, not a guessed stage status."""
    report=queue.load(path);selected=queue.load(selection_path);receipt=report.get('E4_selection',{})
    repos=selected.get('selected_repos',[]);selection_sha=queue.sha(selection_path)
    if (report.get('schema')!='voices_v3_prerequisite_report_v1' or
        report.get('status')!='complete_verified_for_requested_scope' or report.get('input_errors')):
        raise ValueError('An explicitly complete verified prerequisite report is required')
    if (selected.get('schema')!='voices_v3_frozen_E4_model_selection_v1' or selected.get('status')!='frozen' or
        selected.get('performed') is not True or selected.get('all_requested_E1_complete') is not True or
        len(repos)!=6 or len(set(repos))!=6 or receipt.get('status')!='frozen_verified' or
        receipt.get('verified_frozen') is not True or receipt.get('sha256')!=selection_sha or
        receipt.get('selected_repos')!=repos):raise ValueError('Prerequisite report and explicit frozen selection differ')
    verifier=scoring.Verifier(base,source_roots)
    if not same(verifier.path(receipt['path']),selection_path):raise ValueError('Prerequisite selection path differs')
    requested=report.get('requested_models',[]);rows=report.get('models',[]);counts=report.get('counts',{})
    if (len(set(requested))!=len(requested) or len(rows)!=len(requested) or
        {r['repo_id'] for r in rows}!=set(requested) or not set(repos)<=set(requested) or
        counts.get('requested_models')!=len(requested) or counts.get('E0_verified')!=len(requested) or
        counts.get('E1_verified')!=len(requested) or
        counts.get('E2_verified',0)+counts.get('E2_not_applicable_R1_only',0)!=len(requested)):
        raise ValueError('Prerequisite scope/counts do not cover the frozen panel')
    byrepo={r['repo_id']:r for r in rows}
    for repo in repos:
        row=byrepo[repo]
        if (row.get('E0_verified') is not True or row['E1'].get('E1_complete') is not True or
            row['E1'].get('E4_eligible') is not True or row['E2'].get('complete_verified') is not True):
            raise ValueError('Selected model lacks closed E0/E1/E2 prerequisites: '+repo)
    bindings=report.get('verified_source_sha256')
    if not bindings:raise ValueError('Prerequisite report lacks verified source hashes')
    for source,digest in bindings.items():verifier.check(verifier.path(source),digest)
    if not any(same(verifier.path(source),selection_path) and digest==selection_sha for source,digest in bindings.items()):
        raise ValueError('Prerequisite source hashes omit the frozen selection')
    verifier.check(path,queue.sha(path));verifier.check(selection_path,selection_sha)
    return selected,verifier.verified


def validate_plan(path,base,kind,selection_sha,score_plan=None):
    path=Path(path).resolve()
    if path.name!='plan.json' or path.parent.parent!=(base/'setup'/HISTORIES[kind]).resolve():
        raise ValueError('Child plan is outside its exact project history')
    plan=queue.load(path)
    if plan.get('schema')!=SCHEMAS[kind] or plan.get('dry_run') is not False or plan.get('selection_sha256')!=selection_sha:
        raise ValueError('Child plan schema/dry-run/selection identity differs')
    if kind=='score' and (plan.get('CPU_analysis_mode')!='commands_only_deferred' or plan.get('MC_requested') is not False):
        raise ValueError('Score child must defer CPU analyses and not run dynamics')
    if kind=='dynamics' and (not same(plan.get('score_plan',''),score_plan) or
        plan.get('score_plan_sha256')!=queue.sha(score_plan) or plan.get('backend',{}).get('chunk_size')!=2048):
        raise ValueError('Dynamics child is not bound to this score plan and benchmark chunk2048')
    return plan


def discover_child_plan(base,pid,previous,kind,log,selection_sha,score_plan=None,exited=False):
    """PID-owned GPU lease, or this exact child's private stdout receipt if fast."""
    candidate=None;lease=base/'setup/e1_e2_queue.lock'
    if lease.exists():
        try:owner=queue.load(lease)
        except (ValueError,OSError):owner={}
        if owner.get('pid')==pid:candidate=Path(owner['history'])/'plan.json'
    if candidate is None and exited and Path(log).exists():
        # A private log was opened by this Popen invocation. Never choose newest history.
        for line in Path(log).read_text(encoding='utf-8').splitlines():
            try:receipt=json.loads(line)
            except ValueError:continue
            if isinstance(receipt,dict) and receipt.get('plan') and receipt.get('status') in ('finished_score_scope','finished_dynamics_scope'):
                candidate=Path(receipt['plan'])
    if candidate is None:return None
    candidate=candidate.resolve()
    if candidate.parent in previous:raise ValueError('Child receipt identifies a history that predates this invocation')
    validate_plan(candidate,base,kind,selection_sha,score_plan)
    return candidate


def terminal_jobs(receipt,plan_path,plan,kind):
    expected={(j['model'],j['topic_id']) for j in plan['jobs']};rows=receipt.get('jobs',[])
    observed=[(j['model'],j['topic_id']) for j in rows]
    status='finished_score_scope' if kind=='score' else 'finished_dynamics_scope'
    flag='all_requested_scores_complete' if kind=='score' else 'all_requested_dynamics_complete'
    if (receipt.get('status')!=status or receipt.get(flag) is not True or
        not same(receipt.get('plan',''),plan_path) or len(expected)!=36 or len(rows)!=36 or
        len(set(observed))!=36 or set(observed)!=expected or any(j.get('status')!='complete' for j in rows)):
        raise ValueError('Queue terminal receipt is incomplete or belongs to another plan')


def score_and_cpu_closed(plan_path,plan_sha):
    if queue.sha(plan_path)!=plan_sha:raise ValueError('Score plan changed after discovery')
    base,plan,models,decision,_=watching.prepare(plan_path,True)
    final=queue.load(plan_path.parent/'final_status.json');terminal_jobs(final,plan_path,plan,'score')
    status=queue.load(plan_path.parent/'analysis_watcher_status.json')
    if (status.get('status')!='complete' or status.get('all_requested_point_analyses_complete') is not True or
        status.get('all_requested_postprocessing_complete') is not True or status.get('postprocess_requested') is not True or
        not same(status.get('plan',''),plan_path) or status.get('plan_sha256')!=plan_sha or
        len(status.get('jobs',[]))!=36 or any(r.get('status')!='complete' for r in status['jobs']) or
        {(r['model'],r['topic_id']) for r in status['jobs']}!={(r['model'],r['topic_id']) for r in plan['jobs']}):
        raise ValueError('CPU watcher is pending/partial or belongs to another plan')
    summaries=status.get('model_summaries',[])
    if (len(summaries)!=6 or {r['model'] for r in summaries}!=set(models) or
        any(r.get('status')!='complete' or len(r.get('results',[]))!=2 or
            any(v.get('status')!='complete' for v in r['results']) for r in summaries)):
        raise ValueError('All six aggregate and rule-supplement summaries must be closed')
    verify_hashes(status['frozen_file_sha256']);outputs={'aggregate':[],'supplement':[]}
    for job in plan['jobs']:
        closure=queue.score_closure(job,models[job['model']],decision)
        if closure is None or watching.topic_closure(job,models[job['model']],decision,closure) is None:
            raise ValueError('Terminal receipts lack a verified score/point analysis')
    for repo,model in models.items():
        tasks=watching.model_tasks(base,plan_path.parent,model,[j for j in plan['jobs'] if j['model']==repo])
        for task in tasks:
            if watching.model_closure(task,model) is None:raise ValueError('Terminal receipt lacks a verified model summary')
            outputs[task['kind']].append(task['output'])
    return plan,outputs


def dynamic_closed(plan_path,plan_sha,base,selection_sha,score_plan,benchmark_sha):
    if queue.sha(plan_path)!=plan_sha:raise ValueError('Dynamics plan changed after discovery')
    plan=validate_plan(plan_path,base,'dynamics',selection_sha,score_plan)
    if plan.get('benchmark_report_sha256')!=benchmark_sha:raise ValueError('Dynamics used another benchmark')
    terminal_jobs(queue.load(plan_path.parent/'final_status.json'),plan_path,plan,'dynamics')
    verify_hashes(plan['frozen_file_sha256'])
    for job in plan['jobs']:
        if dynamics.closed(job) is None:raise ValueError('Terminal receipt lacks a complete 190-part dynamics output')
    path=plan_path.parent/'optional_aggregation_commands.json';saved=queue.load(path)
    expected=dynamics.aggregation_commands(base,plan,plan_path.parent)
    if saved!=expected:raise ValueError('Saved optional aggregation commands differ from the closed dynamics plan')
    return plan,saved


def cross_model_closed(path,selection_sha,repos):
    result=queue.load(path);rows=result.get('models',[])
    if (result.get('schema')!='voices_v3_E4_cross_model_H3_v1' or
        result.get('status')!='complete_verified_for_selected_panel' or
        result.get('selection_sha256')!=selection_sha or result.get('selected_repos')!=repos or
        result.get('n_models')!=6 or len(rows)!=6 or {r.get('repo_id') for r in rows}!=set(repos) or
        len(result.get('topic_ids',[]))!=6 or len(set(result.get('topic_ids',[])))!=6 or
        result.get('bootstrap',{}).get('draws')!=10000 or
        result.get('primary',{}).get('n_permutations')!=720 or not result.get('file_sha256')):
        raise ValueError('Cross-model H3 is missing, partial or bound to another selected panel')
    artifact=result.get('bootstrap_artifact',{});name=artifact.get('filename')
    if not name:raise ValueError('Cross-model H3 lacks its paired-bootstrap artifact')
    sidecar=path.parent/name
    if sidecar.resolve().parent!=path.parent.resolve() or queue.sha(sidecar)!=artifact.get('sha256'):
        raise ValueError('Cross-model H3 bootstrap artifact is not locally bound')
    verify_hashes(result['file_sha256'])
    return result


def report_closed(path,selection_sha,repos):
    report=queue.load(path);counts=report.get('counts',{});rows=report.get('models',[])
    if (report.get('schema')!='voices_v3_E4_concise_report_v1' or report.get('status')!='complete_E4_main_and_supplement' or
        report.get('single_round_stage_complete') is not True or report.get('cross_model_stage_complete') is not True or
        report.get('cross_model_H3',{}).get('status')!='complete_verified' or
        report.get('input_errors') or report.get('inputs_outside_frozen_panel') or
        report.get('selection',{}).get('sha256')!=selection_sha or report['selection'].get('selected_repos')!=repos or
        any(counts.get(key)!=6 for key in ('planned_models','selected_models','single_round_complete_models','supplement_complete_models','dynamics_complete_models')) or
        len(rows)!=6 or {r['repo_id'] for r in rows}!=set(repos) or
        any(r.get('single_round_status')!='complete_verified' or r.get('supplement_status')!='complete_verified' or
            r.get('dynamics',{}).get('planned_complete') is not True for r in rows)):
        raise ValueError('E4 report remains partial/pending; no phase-complete claim is allowed')
    if not path.with_suffix('.md').is_file():raise ValueError('E4 Markdown report is missing')
    return report


def spawn(command,log,cpu=False):
    environment=dict(os.environ)
    if cpu:environment.update(shared.CPU_ENV);environment['CUDA_VISIBLE_DEVICES']=''
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('a',encoding='utf-8') as stream:
        stream.write('\nInvocation '+queue.now()+'\n');stream.flush()
        return subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)


def wait_processes(processes,save,poll=15):
    while True:
        exits={name:child.poll() for name,child in processes.items()};save({'child_exit_codes':exits})
        if any(code not in (None,0) for code in exits.values()):
            raise RuntimeError('Child execution failed; retained logs/PIDs/partial files require inspection')
        if all(code is not None for code in exits.values()):return
        time.sleep(poll)


def launch_gpu(kind,command,base,history,selection_sha,save,score_plan=None):
    previous={p.resolve() for p in (base/'setup'/HISTORIES[kind]).glob('*') if p.is_dir()}
    log=history/(kind+'_queue.log');child=spawn(command,log)
    save({kind+'_pid':child.pid,kind+'_command':command,kind+'_log':str(log)})
    while True:
        code=child.poll()
        path=discover_child_plan(base,child.pid,previous,kind,log,selection_sha,score_plan,code is not None)
        if path is not None:
            digest=queue.sha(path);save({kind+'_plan':str(path),kind+'_plan_sha256':digest})
            return child,path,digest
        if code is not None:raise RuntimeError(kind+' queue exited before an exact new child-plan receipt was available')
        time.sleep(1)


def cpu_run(command,log,save,label):
    child=spawn(command,log,True);save({'CPU_task':label,'CPU_pid':child.pid,'CPU_command':command,'CPU_log':str(log)})
    wait_processes({label:child},save);save({'CPU_task':None})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--prerequisite-report',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--source-root',action='append',default=[])
    parser.add_argument('--benchmark-report',type=Path);parser.add_argument('--score-plan',type=Path,help='Explicit previously closed score plan; rerun/reuse its CPU watcher and resumable dynamics')
    parser.add_argument('--report-prefix',type=Path);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();base=args.root.resolve();selection_path=args.selection.resolve();prereq=args.prerequisite_report.resolve()
    selected,sources=prerequisite(base,prereq,selection_path,args.source_root);selection_sha=queue.sha(selection_path)
    benchmark_path=args.benchmark_report.resolve() if args.benchmark_report else base/'provenance/e4_gpu_benchmark_20261031/benchmark_report.json'
    verifier=scoring.Verifier(base,args.source_root);benchmark=dynamics.benchmark(benchmark_path,base,verifier)
    if benchmark['selected_chunk_size']!=2048:raise ValueError('This phase requires the completed actual chunk2048 benchmark')
    sources.update(verifier.verified)
    for relative in SOURCE_FILES:sources[str(base/relative)]=queue.sha(base/relative)
    source_args=[value for root in args.source_root for value in ('--source-root',root)]
    score_command=[sys.executable,str(base/'setup/run_e4_score_queue.py'),'--root',str(base),'--selection',str(selection_path),*source_args]
    old_plan=args.score_plan.resolve() if args.score_plan else None
    if old_plan:
        plan=validate_plan(old_plan,base,'score',selection_sha)
        terminal_jobs(queue.load(old_plan.parent/'final_status.json'),old_plan,plan,'score')
        sources[str(old_plan)]=queue.sha(old_plan)
    if args.dry_run:
        print(json.dumps({'status':'readonly_validated_not_executed','selected_repos':selected['selected_repos'],
            'selection_sha256':selection_sha,'prerequisite_report_sha256':queue.sha(prereq),'chunk_size':2048,
            'score_plan_to_reuse':str(old_plan) if old_plan else None,'new_score_command':score_command if not old_plan else None,
            'postprocess':True,'no_model_downloads':True,'no_writes_or_subprocesses':True}));return
    if not os.environ.get('STY'):raise RuntimeError('E4 phase and GPU child queues must run inside screen')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history=base/'setup/e4_phase_history'/stamp
    history.mkdir(parents=True);lease=base/'setup/e4_phase.lock';shared_lease=base/'setup/e1_e2_queue.lock'
    if shared_lease.exists() and shared.pid_alive(queue.load(shared_lease)['pid']):raise RuntimeError('Shared GPU queue is active; do not overlap phases')
    if lease.exists():
        if shared.pid_alive(queue.load(lease)['pid']):raise RuntimeError('Another E4 phase coordinator is active')
        lease.replace(history/'stale_phase_lock.json')
    prefix=args.report_prefix.resolve() if args.report_prefix else history/'report/E4_brief'
    if any(Path(str(prefix)+suffix).exists() for suffix in ('.json','.md')):raise FileExistsError('Use a fresh report prefix; prior reports are preserved')
    with lease.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'history':str(history)},stream)
    state={'schema':'voices_v3_E4_phase_v1','status':'starting','pid':os.getpid(),'history':str(history),
        'started_utc':queue.now(),'prerequisite_report':str(prereq),'prerequisite_report_sha256':queue.sha(prereq),
        'selection':str(selection_path),'selection_sha256':selection_sha,'selected_repos':selected['selected_repos'],
        'benchmark':str(benchmark_path),'benchmark_sha256':queue.sha(benchmark_path),'chunk_size':2048,
        'frozen_file_sha256':sources,'report_prefix':str(prefix),'scientific_success_assumed':False,
        'automatic_downloads_or_deletions':False,'errors_are_not_model_rejections':True,'E4_complete':False}
    def save(changes=None):
        if changes:state.update(changes)
        state['updated_utc']=queue.now();queue.write(base/'setup/e4_phase_status.json',state);queue.write(history/'status.json',state)
    signatures={}
    def verify_sources():shared.verify_frozen(sources,signatures)
    try:
        save();queue.write(history/'plan.json',state);verify_sources()
        save({'status':'scoring_and_CPU_analysis'})
        if old_plan:score_child=None;score_path=old_plan;score_sha=queue.sha(old_plan);save({'score_plan':str(score_path),'score_plan_sha256':score_sha,'score_reused':True})
        else:score_child,score_path,score_sha=launch_gpu('score',score_command,base,history,selection_sha,save)
        watcher_command=[sys.executable,str(base/'setup/run_e4_analysis_watcher_v3.py'),'--plan',str(score_path),'--postprocess']
        watcher_child=spawn(watcher_command,history/'CPU_watcher.log',True)
        save({'watcher_pid':watcher_child.pid,'watcher_command':watcher_command})
        processes={'CPU_watcher':watcher_child}
        if score_child:processes['score']=score_child
        wait_processes(processes,save)
        _,single_outputs=score_and_cpu_closed(score_path,score_sha);save({'single_round_outputs':single_outputs})
        verify_sources();save({'status':'cross_model_H3'})
        cross_path=history/'cross_model/H3.json';cross_path.parent.mkdir(parents=True,exist_ok=True)
        cross_command=[sys.executable,str(base/'code/analyze_e4_cross_model_v3.py'),'--root',str(base),
            '--selection',str(selection_path),'--output',str(cross_path),*source_args]
        for path in single_outputs['aggregate']:cross_command+=['--aggregate',path]
        cpu_run(cross_command,history/'cross_model_H3.log',save,'cross_model_H3')
        cross_model_closed(cross_path,selection_sha,selected['selected_repos'])
        save({'cross_model_output':str(cross_path),'cross_model_sha256':queue.sha(cross_path)})
        verify_sources();save({'status':'dynamics'})
        dynamic_command=[sys.executable,str(base/'setup/run_e4_dynamics_queue_v3.py'),'--root',str(base),
            '--selection',str(selection_path),'--score-plan',str(score_path),'--benchmark-report',str(benchmark_path),*source_args]
        child,dynamic_path,dynamic_sha=launch_gpu('dynamics',dynamic_command,base,history,selection_sha,save,score_path)
        wait_processes({'dynamics':child},save)
        _,tasks=dynamic_closed(dynamic_path,dynamic_sha,base,selection_sha,score_path,queue.sha(benchmark_path))
        save({'status':'dynamic_topic_aggregation','aggregation_tasks':tasks,'aggregation_results':[]})
        outputs=[]
        for task in tasks:
            verify_sources();command=task['command'];output=Path(watching.argument(command,'--output'))
            Path(task['create_output_parent']).mkdir(parents=True,exist_ok=True)
            cpu_run(command,history/'aggregate_logs'/(task['model'].replace('/','--')+'.log'),save,'aggregate_'+task['model'])
            result=queue.load(output);artifact=result['bootstrap_artifact'];sidecar=output.parent/artifact['filename']
            if (result['n_topics']!=6 or not result.get('planned_six_topics_complete') or
                not result.get('single_round_E1_kernel_uncertainty_propagated') or result['bootstrap']['draws']!=10000 or
                result['dynamic']['status']!='complete_conditional_topic_aggregation' or queue.sha(sidecar)!=artifact['sha256']):
                raise ValueError('Dynamic aggregation is incomplete or lacks its bound bootstrap artifact')
            verify_hashes(result['file_sha256']);outputs.append(str(output))
            state['aggregation_results'].append({'model':task['model'],'output':str(output),'output_sha256':queue.sha(output),'artifact_sha256':artifact['sha256'],'status':'complete'});save()
        save({'status':'writing_report','dynamic_aggregate_outputs':outputs});verify_sources()
        command=[sys.executable,str(base/'code/write_e4_report_v3.py'),'--root',str(base),'--selection',str(selection_path),
            '--cross-model',str(cross_path),'--output-prefix',str(prefix)]
        for flag,paths in [('--aggregate',single_outputs['aggregate']),('--supplement',single_outputs['supplement']),('--dynamics-aggregate',outputs)]:
            for path in paths:command+=[flag,path]
        cpu_run(command,history/'E4_report.log',save,'E4_report')
        report_path=Path(str(prefix)+'.json');report=report_closed(report_path,selection_sha,selected['selected_repos'])
        save({'status':'complete_E4_scope','E4_complete':True,'report_json':str(report_path),'report_markdown':str(prefix)+'.md',
            'report_sha256':queue.sha(report_path),'report_counts':report['counts'],'finished_utc':queue.now()})
    except BaseException as error:
        save({'status':'terminal_error_not_model_rejection','error':str(error),'finished_utc':queue.now()});raise
    finally:
        queue.write(history/'final_status.json',state)
        if lease.exists() and queue.load(lease).get('pid')==os.getpid():lease.unlink()


if __name__=='__main__':main()
