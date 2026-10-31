"""Root-recorded local delivery -> all-ten E9 diagnostic -> existing E4 phase.

Only explicit receipts start work. Child queues own/release the GPU lease; this
coordinator never holds it across stages. Scientific programs are reused intact.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

import run_e1_e2_queue as queue
import run_analysis_watcher_v3 as shared
import run_e9_affine_queue_v3 as e9
import run_e4_phase_v3 as e4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import summarize_e9_affine_pairs_v3 as e9report

ROOT=Path(__file__).resolve().parents[1]
EXTRA_SOURCES=('setup/run_after_prerequisite_v3.py','setup/run_e9_affine_queue_v3.py',
    'code/analyze_e9_affine_pairs_v3.py','code/summarize_e9_affine_pairs_v3.py',
    'code/build_e9_affine_balance_v3.py')


def delivery_receipt(path,report,selection,markdown=None):
    """Root's delivery record is checked by hashes, not by pretending to read Windows remotely."""
    value=queue.load(path);markdown=Path(markdown) if markdown else Path(report).with_suffix('.md')
    if (value.get('schema')!='voices_v3_prerequisite_local_delivery_v1' or value.get('status')!='delivered_local' or
        value.get('report_sha256')!=queue.sha(report) or value.get('report_markdown_sha256')!=queue.sha(markdown) or
        value.get('selection_sha256')!=queue.sha(selection) or
        not all(isinstance(value.get(k),str) and value[k].strip() for k in ('local_report_path','local_markdown_path','delivered_utc'))):
        raise ValueError('Root local-delivery receipt is missing, incomplete, or bound to different report/selection files')
    return {'receipt':str(Path(path).resolve()),'receipt_sha256':queue.sha(path),
        'report_markdown':str(markdown.resolve()),'report_markdown_sha256':queue.sha(markdown),
        'local_report_path':value['local_report_path'],'local_markdown_path':value['local_markdown_path'],
        'delivered_utc':value['delivered_utc'],'scope':'Root-recorded delivery after local hash verification; not a remote observation of the Windows filesystem.'}


def validate_e9_plan(path,base,e1_path,prerequisite_path):
    path=Path(path).resolve();plan=queue.load(path);expected=set(queue.CORE_MODELS)
    if (path.name!='plan.json' or path.parent.parent!=(base/'setup/e9_affine_queue_history').resolve() or
        plan.get('schema')!=e9.SCHEMA or plan.get('dry_run') is not False or
        not e4.same(plan.get('e1_plan',''),e1_path) or plan.get('e1_plan_sha256')!=queue.sha(e1_path) or
        not e4.same(plan.get('prerequisite_report',''),prerequisite_path) or plan.get('prerequisite_report_sha256')!=queue.sha(prerequisite_path) or
        plan.get('effect_based_selection') is not False or plan.get('CPU_analysis_mode')!='commands_only_deferred' or
        plan.get('required_models')!=10 or plan.get('total_new_physical_rows')!=40000 or
        len(plan.get('initial_panel',[]))!=10 or set(plan['initial_panel'])!=expected or
        len(plan.get('jobs',[]))!=10 or {j['model'] for j in plan['jobs']}!=expected):
        raise ValueError('E9 plan does not match the exact ten-model prerequisite/E1 supplement')
    for job in plan['jobs']:
        if (job['readout']!=('R1' if job['model']==e9.R1_ONLY else 'R0_R1') or job['rows']!=4000 or
            set(job['original_results'])!=({'R1'} if job['model']==e9.R1_ONLY else {'R0','R1'})):
            raise ValueError('E9 readout panel differs from fixed nine dual-readout plus one R1-only models')
    return plan


def find_e9_child_plan(base,pid,previous,log,e1_path,prerequisite_path,exited=False):
    candidate=None;lease=base/'setup/e1_e2_queue.lock'
    if lease.exists():
        try:owner=queue.load(lease)
        except (ValueError,OSError):owner={}
        if owner.get('pid')==pid:candidate=Path(owner['history'])/'plan.json'
    if candidate is None and exited and Path(log).is_file():
        for line in Path(log).read_text(encoding='utf-8').splitlines():
            try:value=json.loads(line)
            except ValueError:continue
            if isinstance(value,dict) and value.get('status')=='finished_E9_affine_score_scope' and value.get('plan'):
                candidate=Path(value['plan'])
    if candidate is None:return None
    candidate=candidate.resolve()
    if candidate.parent in previous:raise ValueError('E9 child identified an old history')
    validate_e9_plan(candidate,base,e1_path,prerequisite_path);return candidate


def closed_e9_plan(path,digest,base,e1_path,prerequisite_path):
    if queue.sha(path)!=digest:raise ValueError('Explicit E9 plan changed')
    plan=validate_e9_plan(path,base,e1_path,prerequisite_path);final=queue.load(path.parent/'final_status.json')
    if (final.get('status')!='finished_E9_affine_score_scope' or final.get('all_initial10_scores_complete') is not True or
        not e4.same(final.get('plan',''),path) or final.get('plan_sha256')!=digest or len(final.get('jobs',[]))!=10 or
        {r['model'] for r in final['jobs']}!=set(queue.CORE_MODELS) or any(r.get('status')!='complete' for r in final['jobs'])):
        raise ValueError('E9 queue has partial/error scores, so E4 cannot start')
    e4.verify_hashes(plan['frozen_file_sha256']);models={m['repo_id']:m for m in plan['selected_models']}
    for job in plan['jobs']:
        if queue.score_closure(job,models[job['model']],plan['runtime_decision']) is None:
            raise ValueError('E9 terminal receipt lacks a verified raw closure')
    tasks=queue.load(path.parent/'deferred_analysis_commands.json')
    if tasks!=e9.analysis_commands(base,plan,path.parent):raise ValueError('Saved E9 CPU commands differ from the exact score plan')
    return plan,tasks


def analysis_closed(task,job,base):
    path=Path(task['output_prefix']+'.json')
    if not path.exists():return None
    report=e9report.build([path],base,[job['model']])
    if report['status']!='complete_requested_diagnostic_panel':raise ValueError('E9 per-model diagnostic is partial or source verification failed: '+str(report['input_errors']))
    if not path.with_suffix('.md').is_file():raise ValueError('E9 per-model Markdown output missing')
    value=queue.load(path);expected=['R1'] if job['readout']=='R1' else ['R0','R1']
    if value['readouts']!=expected or value['execution_identity']['critical_identity']['repo']!=job['model']:
        raise ValueError('E9 diagnostic omitted a requested readout or changed model identity')
    bindings={str(Path(source).resolve()):digest for source,digest in value['file_sha256'].items()}
    for raw in [job['output'],*job['original_results'].values()]:
        if bindings.get(str(Path(raw).resolve()))!=queue.sha(raw):raise ValueError('E9 diagnostic is bound to another old/new raw source')
    return {'model':job['model'],'status':'complete_verified','output':str(path),'sha256':queue.sha(path)}


def e9_report_closed(path,analyses,base):
    value=queue.load(path);expected=list(queue.CORE_MODELS)
    if (value.get('schema')!='voices_v3_E9_affine_panel_report_v1' or value.get('status')!='complete_requested_diagnostic_panel' or
        value.get('requested_models')!=expected or value.get('n_verified_models')!=10 or value.get('input_errors') or
        len(value.get('models',[]))!=10 or {r['repo_id'] for r in value['models']}!=set(expected) or
        value.get('main_E1_E2_selection_and_PK_unchanged') is not True or value.get('fresh_confirmation') is not False or
        value.get('reporter_sha256')!=queue.sha(e9report.__file__) or not path.with_suffix('.md').is_file()):
        raise ValueError('All-ten E9 report is not completely verified; E4 remains blocked')
    supplied={r['model']:r for r in analyses}
    for row in value['models']:
        if (row.get('status')!='complete_verified' or row.get('sha256')!=supplied[row['repo_id']]['sha256'] or
            not e4.same(row['path'],supplied[row['repo_id']]['output'])):raise ValueError('E9 panel report refers to different per-model diagnostics')
    e4.verify_hashes(value['verified_source_sha256']);return value


def find_e4_child_history(base,pid,previous):
    history=None;lease=base/'setup/e4_phase.lock';status=base/'setup/e4_phase_status.json'
    if lease.exists():
        try:owner=queue.load(lease)
        except (ValueError,OSError):owner={}
        if owner.get('pid')==pid:history=Path(owner['history']).resolve()
    if history is None and status.exists():
        value=queue.load(status)
        if value.get('pid')==pid:history=Path(value['history']).resolve()
    if history is None:return None
    if history.parent!=(base/'setup/e4_phase_history').resolve() or history in previous:
        raise ValueError('E4 child PID does not identify a new exact phase history')
    return history


def e4_child_closed(history,pid,prerequisite_path,selection_path):
    status=queue.load(history/'final_status.json');selection=queue.load(selection_path)
    if (status.get('pid')!=pid or status.get('status')!='complete_E4_scope' or status.get('E4_complete') is not True or
        not e4.same(status.get('history',''),history) or status.get('report_sha256')!=queue.sha(status['report_json']) or
        status.get('selection_sha256')!=queue.sha(selection_path) or
        status.get('prerequisite_report_sha256')!=queue.sha(prerequisite_path)):
        raise ValueError('E4 child phase is not complete for these exact prerequisites/selection')
    e4.report_closed(Path(status['report_json']),queue.sha(selection_path),selection['selected_repos'])
    return status


def commands(base,args,e9_plan=None):
    roots=[v for root in args.source_root for v in ('--source-root',root)]
    first=[sys.executable,str(base/'setup/run_e9_affine_queue_v3.py'),'--root',str(base),
        '--e1-plan',str(args.e1_plan.resolve()),'--prerequisite-report',str(args.prerequisite_report.resolve()),*roots]
    last=[sys.executable,str(base/'setup/run_e4_phase_v3.py'),'--root',str(base),
        '--prerequisite-report',str(args.prerequisite_report.resolve()),'--selection',str(args.selection.resolve()),*roots]
    if args.benchmark_report:last+=['--benchmark-report',str(args.benchmark_report.resolve())]
    if args.e4_report_prefix:last+=['--report-prefix',str(args.e4_report_prefix.resolve())]
    return first,last


def summary_command(base,prefix,paths):
    command=[sys.executable,str(base/'code/summarize_e9_affine_pairs_v3.py'),'--root',str(base),
        '--expected-models',*queue.CORE_MODELS,'--output-prefix',str(prefix)]
    for path in paths:command+=['--analysis',str(path)]
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--e1-plan',type=Path,required=True);parser.add_argument('--prerequisite-report',type=Path,required=True)
    parser.add_argument('--prerequisite-markdown',type=Path);parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--local-delivery-receipt',type=Path,required=True);parser.add_argument('--source-root',action='append',default=[])
    parser.add_argument('--e9-plan',type=Path,help='Explicit already closed E9 score plan for CPU/phase recovery')
    parser.add_argument('--e9-report-prefix',type=Path);parser.add_argument('--e4-report-prefix',type=Path)
    parser.add_argument('--benchmark-report',type=Path);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();base=args.root.resolve();e1=args.e1_plan.resolve();prereq=args.prerequisite_report.resolve();selection=args.selection.resolve()
    delivery=delivery_receipt(args.local_delivery_receipt,prereq,selection,args.prerequisite_markdown)
    frozen,sources=e4.prerequisite(base,prereq,selection,args.source_root)
    sources[delivery['receipt']]=delivery['receipt_sha256'];sources[delivery['report_markdown']]=delivery['report_markdown_sha256']
    manifest=base/'data/e9_affine_balance/manifest.json';amendment=base/'provenance/E9_affine_pair_balance_20261031.txt'
    old=args.e9_plan.resolve() if args.e9_plan else None
    if old:plan,tasks=closed_e9_plan(old,queue.sha(old),base,e1,prereq);sources[str(old)]=queue.sha(old)
    else:plan=e9.prepare(base,e1,prereq,manifest,amendment,args.source_root);tasks=None
    sources.update(plan['frozen_file_sha256'])
    verifier=e9.Verifier(base,args.source_root)
    benchmark_path=args.benchmark_report.resolve() if args.benchmark_report else base/'provenance/e4_gpu_benchmark_20261031/benchmark_report.json'
    benchmark=e4.dynamics.benchmark(benchmark_path,base,verifier)
    if benchmark['selected_chunk_size']!=2048:raise ValueError('Expected the completed actual chunk2048 E4 benchmark')
    sources.update(verifier.verified)
    for relative in (*e4.SOURCE_FILES,*EXTRA_SOURCES):sources[str(base/relative)]=queue.sha(base/relative)
    first,last=commands(base,args);prefix=args.e9_report_prefix.resolve() if args.e9_report_prefix else base/'reports/E9_affine_all10_20261031'
    prior=[Path(str(prefix)+suffix).exists() for suffix in ('.json','.md')]
    if any(prior) and (not old or not all(prior)):
        raise FileExistsError('Preserve the prior E9 report: supply its closed --e9-plan with both report files or choose a fresh report prefix')
    if args.e4_report_prefix and any(Path(str(args.e4_report_prefix.resolve())+suffix).exists() for suffix in ('.json','.md')):
        raise FileExistsError('The existing E4 phase requires a fresh explicit report prefix')
    if args.dry_run:
        future_history=old.parent if old else base/'setup/e9_affine_queue_history/EXACT_CHILD_HISTORY_NOT_YET_CREATED'
        future_tasks=tasks if tasks is not None else e9.analysis_commands(base,plan,future_history)
        print(json.dumps({'status':'read_only_validated_not_executed','local_delivery':delivery,'E9_before_E4':True,
            'E9_score_command':first if not old else None,'reuse_closed_E9_plan':str(old) if old else None,
            'E9_CPU_commands':future_tasks,'E9_report_prefix':str(prefix),'E4_phase_command_after_verified_E9':last,
            'E9_summary_command':summary_command(base,prefix,[task['output_prefix']+'.json' for task in future_tasks]),
            'no_writes_or_subprocesses':True}));return
    if not os.environ.get('STY'):raise RuntimeError('The post-prerequisite coordinator and GPU children must run inside screen')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history=base/'setup/after_prerequisite_history'/stamp
    history.mkdir(parents=True);lease=base/'setup/after_prerequisite.lock';gpu_lease=base/'setup/e1_e2_queue.lock'
    if gpu_lease.exists() and shared.pid_alive(queue.load(gpu_lease)['pid']):raise RuntimeError('A GPU queue is active; prerequisites must finish before this invocation')
    if lease.exists():
        if shared.pid_alive(queue.load(lease)['pid']):raise RuntimeError('A post-prerequisite coordinator is already active')
        lease.replace(history/'stale_coordinator_lock.json')
    with lease.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'history':str(history)},stream)
    state={'schema':'voices_v3_after_prerequisite_v1','status':'starting','pid':os.getpid(),'history':str(history),
        'started_utc':queue.now(),'local_delivery':delivery,'prerequisite_report':str(prereq),'prerequisite_report_sha256':queue.sha(prereq),
        'selection':str(selection),'selection_sha256':queue.sha(selection),'selected_repos':frozen['selected_repos'],
        'frozen_file_sha256':sources,'E9_complete':False,'E4_launched':False,'E4_complete':False,
        'scientific_success_assumed':False,'no_model_download_or_delete':True,'CPU_threads':1}
    def save(changes=None):
        if changes:state.update(changes)
        state['updated_utc']=queue.now();queue.write(base/'setup/after_prerequisite_status.json',state);queue.write(history/'status.json',state)
    signatures={}
    def verify_sources():shared.verify_frozen(sources,signatures)
    try:
        save();queue.write(history/'plan.json',state);verify_sources()
        if old:score_path=old;score_sha=queue.sha(old);save({'E9_plan':str(old),'E9_plan_sha256':score_sha,'E9_score_reused':True})
        else:
            previous={p.resolve() for p in (base/'setup/e9_affine_queue_history').glob('*') if p.is_dir()}
            log=history/'E9_queue.log';child=e4.spawn(first,log)
            save({'status':'E9_scoring','E9_pid':child.pid,'E9_command':first,'E9_log':str(log)})
            while True:
                rc=child.poll();score_path=find_e9_child_plan(base,child.pid,previous,log,e1,prereq,rc is not None)
                if score_path is not None:break
                if rc is not None:raise RuntimeError('E9 exited before an exact child-plan receipt')
                time.sleep(1)
            score_sha=queue.sha(score_path);save({'E9_plan':str(score_path),'E9_plan_sha256':score_sha})
            e4.wait_processes({'E9_queue':child},save)
        plan,tasks=closed_e9_plan(score_path,score_sha,base,e1,prereq)
        save({'status':'E9_CPU_analyses','E9_CPU_tasks':tasks,'E9_analysis_results':[]});jobs={j['model']:j for j in plan['jobs']}
        for task in tasks:
            verify_sources();closed=analysis_closed(task,jobs[task['model']],base)
            if closed is None:
                Path(task['create_output_parent']).mkdir(parents=True,exist_ok=True)
                e4.cpu_run(task['command'],history/'E9_CPU_logs'/(task['model'].replace('/','--')+'.log'),save,'E9_'+task['model'])
                closed=analysis_closed(task,jobs[task['model']],base)
                if closed is None:raise ValueError('E9 CPU command returned without a verified diagnostic')
            state['E9_analysis_results'].append(closed);save()
        verify_sources();save({'status':'E9_panel_report'});report_path=Path(str(prefix)+'.json')
        if not report_path.exists():
            report_command=summary_command(base,prefix,[result['output'] for result in state['E9_analysis_results']])
            e4.cpu_run(report_command,history/'E9_panel_report.log',save,'E9_panel_report')
        e9_report_closed(report_path,state['E9_analysis_results'],base)
        save({'E9_complete':True,'E9_report_json':str(report_path),'E9_report_markdown':str(prefix)+'.md','E9_report_sha256':queue.sha(report_path)})
        verify_sources();save({'status':'E4_phase','E4_command':last})
        previous={p.resolve() for p in (base/'setup/e4_phase_history').glob('*') if p.is_dir()}
        child=e4.spawn(last,history/'E4_phase.log');save({'E4_pid':child.pid,'E4_launched':True})
        while True:
            rc=child.poll();child_history=find_e4_child_history(base,child.pid,previous)
            if child_history is not None:break
            if rc is not None:raise RuntimeError('E4 child exited before an exact PID-bound phase history')
            time.sleep(1)
        save({'E4_history':str(child_history)});e4.wait_processes({'E4_phase':child},save)
        completed=e4_child_closed(child_history,child.pid,prereq,selection)
        save({'status':'complete_E9_then_E4','E4_complete':True,'E4_report_json':completed['report_json'],
            'E4_report_markdown':completed['report_markdown'],'E4_report_sha256':completed['report_sha256'],'finished_utc':queue.now()})
    except BaseException as error:
        save({'status':'terminal_error_not_model_rejection','error':str(error),'finished_utc':queue.now()});raise
    finally:
        queue.write(history/'final_status.json',state)
        if lease.exists() and queue.load(lease).get('pid')==os.getpid():lease.unlink()


if __name__=='__main__':main()
