"""Run the fixed E4 dynamics panel from an explicit closed scoring plan.

No model inference, selection, threshold changes or waiting for unfinished work.
The existing dynamics driver owns resumable per-condition/mapping parts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import run_e1_e2_queue as queue
from run_e4_score_queue import Verifier

CODE=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(CODE))
from e4_dynamic_stratified import mapping_seed, combine_mapping_runs

DYNAMIC_CODE=('run_e4_dynamics_v3.py','e4_dynamic_stratified.py','e4_simulation.py','e4_simulation_torch.py')
SECTIONS=('outcomes','paired_difference_to_design0','paired_residual_U_measured_minus_other_methods')


def benchmark(path,base,verifier):
    value=queue.load(path);verifier.check(path,queue.sha(path))
    if (value.get('schema')!='voices_v3_synthetic_E4_GPU_benchmark_v1' or value.get('status')!='complete'
        or value.get('synthetic_only') is not True or value.get('selected_rng_backend')!='torch'
        or value.get('CPU_numpy_vs_GPU_same_stream',{}).get('passed') is not True
        or value.get('inference_runtime_decision_unchanged') is not True
        or value['configuration']['device']!='cuda:0' or value['configuration']['draws']!=5000
        or value['configuration']['rounds']!=6):
        raise ValueError('A completed, matching CUDA benchmark with passed same-stream check is required')
    for source,digest in value['source_sha256'].items():verifier.check(base/source,digest)
    required={'code/e4_simulation_torch.py','code/e4_simulation.py','code/e4_graphs.py',
              'data/e4_graphs/sts15_all80.json','setup/benchmark_e4_gpu.py'}
    if not required<=set(value['source_sha256']):raise ValueError('Benchmark lacks required source identities')
    verifier.check(path.parent/'synthetic_inputs.npz',value['synthetic_inputs_sha256'])
    if value['inference_runtime_decision_sha256_after']!=queue.sha(base/'setup/runtime_decision.json'):
        raise ValueError('Inference runtime changed since the benchmark')
    stable=[c for c in value['chunks'] if c['status']=='stable']
    chosen=min(stable,key=lambda c:(c['median_seconds'],c['chunk_size'])) if stable else None
    if chosen is None or chosen['chunk_size']!=value['selected_chunk_size'] or not chosen['exact_repeated_results']:
        raise ValueError('Benchmark did not choose its fastest stable chunk')
    for trial in chosen['trials']:verifier.check(path.parent/trial['artifact'],trial['artifact_sha256'])
    return value


def point_closed(path,score_job,model,base,verifier):
    """Bind closed point estimates to exactly this cache score and E1 checkpoint."""
    point=queue.load(path);kernel=queue.load(model['kernel_analysis'])
    if (point.get('topic_id')!=score_job['topic_id'] or point.get('MC_completed') is not False
        or point.get('E1_topic_disjoint') is not True or point['option_mass']['n_rows']!=49020
        or point['E1_execution_identity']!=kernel['execution_identity']
        or kernel.get('readout')!='R0' or point['topic'] in kernel['topics']):
        raise ValueError('Point analysis is not the complete planned held-out R0 topic')
    hashes={str(verifier.path(p).resolve()):h for p,h in point['input_hashes'].items()}
    required=[score_job['input'],score_job['output'],model['kernel_analysis'],
        base/'data/e4_cache_manifest.json',base/'data/e4_graphs/sts15_all80.json',base/'code/run_e4_analysis.py']
    if not all(str(Path(p).resolve()) in hashes for p in required):raise ValueError('Point analysis omits a required input binding')
    for p,h in hashes.items():verifier.check(p,h)
    raw=Path(score_job['output']);run=queue.load(str(raw)+'.run.json');identity=point['execution_identity']
    receipts=identity['runs']
    if (len(receipts)!=1 or verifier.path(receipts[0]['path']).resolve()!=raw.resolve()
        or receipts[0]['run_sha256']!=queue.sha(str(raw)+'.run.json')
        or receipts[0]['engine']!=model['engine'] or receipts[0]['calibration_sha256']!=model['calibration_sha256']
        or receipts[0]['batch_size']!=run['batch_size'] or identity['same_execution_policy'] is not True
        or identity['mixed_execution_explicitly_allowed'] is not False
        or identity['critical_identity']!=kernel['execution_identity']['critical_identity']):
        raise ValueError('Point analysis execution identity differs from R0 score/E1 kernel')
    mappings=point['E4b']['point_to_agent_mappings']
    if point['E4b']['n_systems']!=80 or len(mappings)!=20 or any(sorted(m)!=list(range(15)) for m in mappings):
        raise ValueError('Point analysis lacks the fixed 80 STS / 20 mappings')
    cache=path.parent/point['cache_artifact']['filename'];verifier.check(cache,point['cache_artifact']['sha256'])
    verifier.check(path,queue.sha(path))
    return point,cache


def prepare(base,selection_path,score_plan_path,benchmark_path,source_roots=()):
    verifier=Verifier(base,source_roots)
    selection=queue.load(selection_path);scoreplan=queue.load(score_plan_path)
    repos=selection.get('selected_repos',[])
    if (selection.get('schema')!='voices_v3_frozen_E4_model_selection_v1' or selection.get('status')!='frozen'
        or selection.get('performed') is not True or selection.get('all_requested_E1_complete') is not True
        or len(repos)!=6 or len(set(repos))!=6 or scoreplan.get('schema')!='voices_v3_E4_score_queue_v1'
        or scoreplan.get('dry_run') is not False or scoreplan['selected_repos']!=repos
        or scoreplan['selection_sha256']!=queue.sha(selection_path)):
        raise ValueError('Require the explicit frozen six-model selection and its executed E4 score plan')
    verifier.check(selection_path,queue.sha(selection_path));verifier.check(score_plan_path,queue.sha(score_plan_path))
    for path,digest in scoreplan['frozen_file_sha256'].items():verifier.check(verifier.path(path),digest)
    panel_path=verifier.path(selection['panel_path']);verifier.check(panel_path,selection['panel_sha256'])
    panel=queue.load(panel_path);rows={r['repo_id']:r for r in panel['models']}
    if not panel.get('all_requested_E1_complete') or any(selection.get(k)!=v for k,v in panel['E4_selection'].items()):
        raise ValueError('Frozen selection differs from its completed E1 panel')
    report=benchmark(benchmark_path,base,verifier);decision=scoreplan['runtime_decision']
    if decision!=queue.runtime(base):raise ValueError('Score plan uses another frozen inference runtime')
    models={m['repo_id']:dict(m) for m in scoreplan['selected_models']}
    if set(models)!=set(repos):raise ValueError('Score plan model/checkpoint panel differs')
    for repo,model in models.items():
        for key in ('calibration','kernel_analysis','E0_analysis_path'):model[key]=str(verifier.path(model[key]))
        row=rows[repo]
        if (not row['E1_complete'] or not row['E4_eligible'] or not model['R0_operational']
            or model['revision']!=row['revision'] or model['calibration_sha256']!=row['calibration_sha256']
            or model['kernel_analysis_sha256']!=row['primary']['sha256']):
            raise ValueError('Frozen E1 checkpoint/readout binding differs: '+repo)
        for key,digest in [('calibration','calibration_sha256'),('kernel_analysis','kernel_analysis_sha256'),('E0_analysis_path','E0_analysis_sha256')]:
            verifier.check(model[key],model[digest])
        lock=queue.load(model['calibration']);engine=lock['engine'];parent=engine['parent_loader']
        if (parent['repo_id']!=repo or parent['revision']!=model['revision'] or engine!=model['engine']
            or lock['option_ids']!=model['option_ids'] or lock['batch_size']!=decision['batch_size']):
            raise ValueError('Checkpoint/calibration does not match the completed R0 scoring plan')
    manifest=queue.load(base/'data/e4_cache_manifest.json');topics={t['topic_id']:t for t in manifest['topics']}
    if len(topics)!=6:raise ValueError('Exactly six fixed E4 topics required')
    jobs=[];seen=set();codes={name:queue.sha(base/'code'/name) for name in DYNAMIC_CODE}
    for name,digest in codes.items():verifier.check(base/'code'/name,digest)
    for extra in [Path(__file__),base/'setup/run_e1_e2_queue.py',base/'setup/run_e4_score_queue.py',
                  base/'code/aggregate_e4_v3.py',base/'code/run_e4_analysis.py',base/'code/analyze_v3.py']:
        verifier.check(extra,queue.sha(extra))
    versions={key:report['runtime'][key] for key in ('python','numpy','torch','cuda')}
    for saved in scoreplan['jobs']:
        job=dict(saved);repo=job['model'];tid=job['topic_id'];pair=(repo,tid)
        if repo not in models or tid not in topics or pair in seen or job['readout']!='R0' or job['rows']!=49020:
            raise ValueError('E4 score plan has duplicate/unexpected topics or non-R0 jobs')
        seen.add(pair)
        for key in ('input','output'):job[key]=str(verifier.path(job[key]))
        spec=topics[tid]['output']
        if Path(job['input']).resolve()!=(base/spec['path']).resolve() or job['input_sha256']!=spec['sha256']:
            raise ValueError('Score input does not match its fixed topic manifest')
        closure=queue.score_closure(job,models[repo],decision)
        if closure is None:raise ValueError('E4 score is not closed; rerun after scoring finishes: '+job['output'])
        for suffix,key in [('', 'output_sha256'),('.run.json','run_sha256'),('.summary.json','summary_sha256')]:
            verifier.check(job['output']+suffix,closure[key])
        directory=verifier.path(job['deferred_analysis']['output_dir']);path=directory/(tid+'_E4_analysis.json')
        point,cache=point_closed(path,job,models[repo],base,verifier)
        if point['topic']!=topics[tid]['topic']:raise ValueError('Point topic label differs from its manifest')
        output=directory/(tid+'_E4_dynamics.json')
        identity={'source_E4_analysis_sha256':queue.sha(path),'E1_analysis_sha256':models[repo]['kernel_analysis_sha256'],
            'cache_sha256':queue.sha(cache),'backend':'torch','device':'cuda:0','rng_backend':'torch',
            'runtime_versions':versions,'code_sha256':codes}
        command=[sys.executable,str(base/'code/run_e4_dynamics_v3.py'),'--topic-analysis',str(path),
            '--kernel-analysis',models[repo]['kernel_analysis'],'--output',str(output),'--stages','a,b',
            '--cache-manifest',str(base/'data/e4_cache_manifest.json'),'--sts',str(base/'data/e4_graphs/sts15_all80.json'),
            '--total-draws-per-design','100000','--pasch-draws','100000','--rounds','6',
            '--backend','torch','--device','cuda:0','--rng-backend','torch','--chunk-size',str(report['selected_chunk_size'])]
        jobs.append({'model':repo,'topic_id':tid,'topic_analysis':str(path),'kernel_analysis':models[repo]['kernel_analysis'],
            'output':str(output),'log':str(output)+'.log','expected_identity':identity,'command':command,
            'chunk_size':report['selected_chunk_size'],'g6_agent_ids':topics[tid]['g6_agent_ids']})
    if seen!=set(itertools.product(repos,topics)):raise ValueError('Require all six models x six topics before dynamics')
    return {'schema':'voices_v3_E4_dynamics_queue_v1','created_utc':queue.now(),
        'selection':str(selection_path),'selection_sha256':queue.sha(selection_path),
        'score_plan':str(score_plan_path),'score_plan_sha256':queue.sha(score_plan_path),
        'benchmark_report':str(benchmark_path),'benchmark_report_sha256':queue.sha(benchmark_path),
        'selected_repos':repos,'jobs':jobs,'n_jobs':len(jobs),'frozen_file_sha256':verifier.verified,
        'allocation':{'STS_total_per_design_condition_topic_model':100000,'independent_mapping_seeds':20,
            'STS_draws_per_mapping':5000,'Pasch_per_condition':100000,'rounds':6},
        'backend':{'implementation':'torch','device':'cuda:0','rng':'torch','chunk_size':report['selected_chunk_size'],
            'runtime_versions':versions,'benchmark_gpu_query':report['runtime']['nvidia_gpu_query']},
        'model_inference_requested':False,'weights_required':False,'selection_or_threshold_tuning':False,
        'dynamic_E1_kernel_uncertainty_propagated':False}


def simulation_valid(result,job,draws,designs,agents,seed,tau):
    expected={'draws':draws,'n_designs':designs,'n_agents':agents,'rounds':6,'chunk_size':job['chunk_size'],
              'seed':seed,'tau':tau,'methods':['measured','P0','PK']}
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete or mismatched simulation dimensions/seed')
    backend=result['simulation_backend'];versions=job['expected_identity']['runtime_versions']
    if (backend['device']!='cuda:0' or backend['rng']!='torch_private_device_generator'
        or backend['torch_version']!=versions['torch'] or backend['numpy_version']!=versions['numpy']
        or backend['code_sha256']!=job['expected_identity']['code_sha256']['e4_simulation_torch.py']):
        raise ValueError('Simulation backend differs from the benchmarked frozen implementation')
    for section in SECTIONS:
        shape=(2 if section==SECTIONS[-1] else 3,designs,6)
        means=np.asarray(result[section]['means']);se=np.asarray(result[section]['Monte_Carlo_standard_errors'])
        if means.shape!=shape or se.shape!=shape or not np.isfinite(means).all() or not np.isfinite(se).all() or (se<0).any():
            raise ValueError('Incomplete/nonfinite dynamic outcomes or paired MCSE')


def closed(job):
    path=Path(job['output'])
    if not path.exists():return None
    result=queue.load(path)
    if (result.get('identity')!=job['expected_identity'] or result.get('topic_id')!=job['topic_id']
        or result.get('MC_completed') is not True or result.get('requested_stages')!='a,b'
        or result.get('E1_kernel_parameter_uncertainty_propagated') is not False):
        raise ValueError('Existing dynamics output has changed source/runtime or incomplete stages')
    parts=path.with_name(path.stem+'_parts');hashes={}
    def part(name,sim,draws,designs,agents,seed,tau):
        simulation_valid(sim,job,draws,designs,agents,seed,tau)
        pp=parts/(name+'.json');saved=queue.load(pp)
        kwargs={'draws':draws,'rounds':6,'chunk_size':job['chunk_size'],'seed':seed,'tau':tau}
        if saved['binding']!={'identity':job['expected_identity'],'simulation_arguments':kwargs} or saved['result']!=sim:
            raise ValueError('Completed part differs from final dynamics output: '+str(pp))
        hashes[pp.name]=queue.sha(pp)
    a=result['E4a_MC'];expected_a={}
    for n,tau in itertools.product((2,3),(-1,1)):
        for agents in itertools.combinations(sorted(job['g6_agent_ids']),n):
            key=f'm{n}_tau{tau}_agents'+('_'.join(map(str,agents)));expected_a[key]=(n,tau,list(agents))
    records={r['condition_id']:r for r in a['records']}
    if (len(a['records'])!=70 or set(records)!=set(expected_a) or a['n_conditions']!=70 or a['outcome_denominator']!=6
        or a['irrelevant_frozen_agents_included'] is not False or a['draws_per_design_per_condition']!=100000):
        raise ValueError('Incomplete G6 Pasch conditions or wrong agent denominator')
    for name,(n,tau,agents) in expected_a.items():
        record=records[name]
        if (record['n_target'],record['tau'],record['target_agent_ids'])!=(n,tau,agents):raise ValueError('Pasch condition identity differs')
        part('E4a_'+name,record['simulation'],100000,2,6,mapping_seed(job['topic_id'],n,tau,name),tau)
    expected_b={f'm{n}_tau{tau}':(n,tau) for n,tau in itertools.product((3,5,7),(-1,1))}
    if set(result['E4b_MC'])!=set(expected_b):raise ValueError('Missing STS conditions')
    for name,(n,tau) in expected_b.items():
        value=result['E4b_MC'][name];seeds=[mapping_seed(job['topic_id'],n,tau,mi) for mi in range(20)]
        if (value['n_systems']!=80 or value['n_mappings']!=20 or value['total_draws_per_design']!=100000
            or value['draws']!=5000 or value['mapping_seeds']!=seeds or len(value['mapping_runs'])!=20):
            raise ValueError('STS budget must be 20 independent mappings x5000 draws')
        for mi,sim in enumerate(value['mapping_runs']):part(f'E4b_{name}_map{mi:02d}',sim,5000,80,15,seeds[mi],tau)
        if value!=combine_mapping_runs(value['mapping_runs'],seeds,100000):
            raise ValueError('STS pooled means or paired Monte Carlo SE differ from their completed mapping parts')
    return {'status':'complete','output_sha256':queue.sha(path),'n_closed_parts':len(hashes),'part_sha256':hashes}


def active_gpu_work(base):
    names={'inference_v3.py','run_e0_v3.py','run_e0_v3_taskpilot.py','run_e0_v3_conditional.py',
        'run_e0_queue.py','run_e0_taskpilot_recovery.py','run_e0_conditional_queue.py','run_e1_e2_queue.py',
        'run_e4_score_queue.py','run_e4_dynamics_queue_v3.py','run_e4_dynamics_v3.py','benchmark_e4_gpu.py'}
    pids=[]
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name)==os.getpid():continue
        try:args=[a.decode(errors='ignore') for a in (process/'cmdline').read_bytes().split(b'\0') if a]
        except OSError:continue
        if args and Path(args[0]).name.startswith('python') and any(Path(a).name in names for a in args[1:]):pids.append(int(process.name))
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids.extend(int(s) for s in query.stdout.splitlines() if s.strip().isdigit() and int(s)!=os.getpid())
    return sorted(set(pids))


def runtime_matches(plan):
    import torch  # No model or GPU allocation; called only by the actual screened queue.
    actual={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cuda':torch.version.cuda}
    if actual!=plan['backend']['runtime_versions']:raise ValueError('Simulation runtime differs from completed benchmark')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip()!=plan['backend']['benchmark_gpu_query']:raise ValueError('GPU/driver differs from completed benchmark')


def execute(base,history,plan):
    if not os.environ.get('STY'):raise RuntimeError('Dynamics GPU queue must run in GNU screen')
    lease=base/'setup/e1_e2_queue.lock'
    if lease.exists():
        owner=queue.load(lease)
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:lease.replace(history/'recovered_stale_GPU_lease.json')
        else:raise RuntimeError('Another GPU queue holds the shared lease')
    if active_gpu_work(base):raise RuntimeError('Inference/another CUDA process is active; no overlap permitted')
    with lease.open('x',encoding='utf-8') as stream:
        __import__('json').dump({'pid':os.getpid(),'history':str(history),'kind':'E4_dynamics'},stream)
    state={'status':'running','started_utc':queue.now(),'plan':str(history/'plan.json'),
        'plan_sha256':queue.sha(history/'plan.json'),'jobs':[],'model_inference_executed':False}
    statepath=base/'setup/e4_dynamics_queue_status.json';events=history/'events.jsonl';stamps={}
    try:
        runtime_matches(plan);queue.write(statepath,state)
        for job in plan['jobs']:
            entry={k:job[k] for k in ('model','topic_id','output','topic_analysis')};entry['started_utc']=queue.now()
            state['current_job']=entry.copy();queue.write(statepath,state)
            try:
                for path,digest in plan['frozen_file_sha256'].items():
                    p=Path(path);stamp=(p.stat().st_size,p.stat().st_mtime_ns)
                    if stamps.get(path)!=stamp:
                        if queue.sha(p)!=digest:raise ValueError('Frozen source changed: '+path)
                        stamps[path]=stamp
                receipt=closed(job)
                if receipt is None:
                    if active_gpu_work(base):raise RuntimeError('Another GPU/inference process appeared between jobs')
                    parts=Path(job['output']).with_name(Path(job['output']).stem+'_parts')
                    entry['existing_part_files']=len(list(parts.glob('*.json'))) if parts.exists() else 0
                    queue.event(events,{'event':'dynamics_start',**entry,'command':job['command']})
                    print('START '+job['model']+' '+job['topic_id'],flush=True)
                    entry['exit_code']=queue.run_command(job['command'],job['log'])
                    if entry['exit_code']:raise RuntimeError('Dynamics exited '+str(entry['exit_code'])+'; all partial files retained')
                    receipt=closed(job)
                    if receipt is None:raise ValueError('Dynamics returned without a closed output')
                entry.update(receipt)
            except Exception as error:entry.update(status='execution_error_not_model_disqualification',error=str(error))
            entry['finished_utc']=queue.now();state['jobs'].append(entry)
            queue.event(events,{'event':'job_finished',**entry});queue.write(statepath,state)
        state.update(status='finished_dynamics_scope',finished_utc=queue.now(),current_job=None,
            all_requested_dynamics_complete=len(state['jobs'])==36 and all(j['status']=='complete' for j in state['jobs']),
            errors_preserved_without_model_disqualification=True)
        queue.write(statepath,state);queue.write(history/'final_status.json',state)
    finally:
        if lease.exists() and queue.load(lease).get('pid')==os.getpid():lease.unlink()
    return state


def aggregation_commands(base,plan,history):
    records=[]
    for repo in plan['selected_repos']:
        jobs=[j for j in plan['jobs'] if j['model']==repo]
        output=history/'optional_aggregates'/(repo.replace('/','--')+'_E4_with_dynamics.json')
        cmd=[sys.executable,str(base/'code/aggregate_e4_v3.py'),'--kernel-analysis',jobs[0]['kernel_analysis'],
            '--output',str(output),'--cache-manifest',str(base/'data/e4_cache_manifest.json'),
            '--sts',str(base/'data/e4_graphs/sts15_all80.json'),'--draws','10000']
        for job in jobs:cmd+=['--topics',job['topic_analysis'],'--dynamics',job['output']]
        records.append({'model':repo,'command':cmd,'status':'command_only_not_executed',
            'requires_all_six_topic_dynamics_closed':True,'create_output_parent':str(output.parent),
            'dynamic_intervals_conditional_on_E1_point_kernels':True})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--score-plan',type=Path,required=True)
    parser.add_argument('--benchmark-report',type=Path)
    parser.add_argument('--source-root',action='append',default=[]);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();base=args.root.resolve()
    bp=args.benchmark_report.resolve() if args.benchmark_report else base/'provenance/e4_gpu_benchmark_20261031/benchmark_report.json'
    plan=prepare(base,args.selection.resolve(),args.score_plan.resolve(),bp,args.source_root)
    history=base/'setup/e4_dynamics_queue_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    plan['dry_run']=args.dry_run;queue.write(history/'plan.json',plan)
    queue.write(history/'optional_aggregation_commands.json',aggregation_commands(base,plan,history))
    if args.dry_run:
        print(__import__('json').dumps({'status':'dry_run_only','jobs':len(plan['jobs']),'plan':str(history/'plan.json')}));return
    state=execute(base,history,plan)
    print(__import__('json').dumps({'status':state['status'],'all_dynamics_complete':state['all_requested_dynamics_complete'],'plan':state['plan']}))


if __name__=='__main__':main()
