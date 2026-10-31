"""Separate E9 pair-position supplement after closed E0/E1/E2, before E4.

All initial ten models, fixed inputs/runtime/readouts; no effect-based selection.
The unchanged inference engine owns prefix-resume and raw/run/summary closure.
CPU analysis commands are saved for the caller, never executed under this lease.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import run_e1_e2_queue as common
from run_e4_score_queue import Verifier

ROOT=Path(__file__).resolve().parents[1]
R1_ONLY='Qwen/Qwen3.5-0.8B'
SCHEMA='voices_v3_E9_affine_pair_queue_v1'


def prerequisite(path,base,verifier):
    report=common.load(path);rows=report.get('models',[]);expected=set(common.CORE_MODELS)
    if (report.get('schema')!='voices_v3_prerequisite_report_v1'
        or report.get('status')!='complete_verified_for_requested_scope' or report.get('input_errors')
        or set(report.get('requested_models',[]))!=expected or len(rows)!=10
        or {r['repo_id'] for r in rows}!=expected):
        raise ValueError('All ten models require a completed E0/E1/E2 prerequisite report before E9')
    byrepo={r['repo_id']:r for r in rows}
    for repo,row in byrepo.items():
        if not row.get('E0_verified') or not row['E1'].get('E1_complete'):
            raise ValueError('E0/E1 evidence is not closed for '+repo)
        if repo==R1_ONLY:
            if row['E2'].get('status')!='not_applicable_R1_only' or row['E1'].get('R0_operational') is not False:
                raise ValueError('Qwen0.8 must retain its established R1-only scope')
        elif not row['E2'].get('complete_verified') or row['E1'].get('R0_operational') is not True:
            raise ValueError('Operational R0 model has unfinished E2: '+repo)
    if not report.get('verified_source_sha256'):raise ValueError('Prerequisite report lacks source bindings')
    for source,digest in report['verified_source_sha256'].items():verifier.check(verifier.path(source),digest)
    verifier.check(path,common.sha(path))
    return byrepo


def materials(path,base,verifier):
    manifest=common.load(path)
    if manifest.get('schema')!='voices_v3_E9_affine_pair_balance_materials_v1':raise ValueError('Unexpected E9 material manifest')
    ids=manifest['item_ids'];old=common.load(base/'data/core_data_manifest.json')['reused20_ids']
    if len(ids)!=20 or len(set(ids))!=20 or set(ids)!=set(old):raise ValueError('E9 must use exactly the frozen prior20 items')
    if manifest['old_permutations']!=list(range(10)) or manifest['new_permutations']!=list(range(10,20)):
        raise ValueError('E9 old/new permutation indices differ from the fixed supplement')
    for spec in [*manifest['original_sources'].values(),manifest['additional_prompts'],manifest['items'],manifest['builder'],manifest['validation']]:
        target=base/spec['path'];verifier.check(target,spec['sha256'])
        if target.stat().st_size!=spec['bytes']:raise ValueError('E9 material byte count differs')
    spec=manifest['additional_prompts'];source=base/spec['path']
    if spec['n_rows']!=4000:raise ValueError('Expected exactly 4000 additional pair prompts')
    validation=common.load(base/manifest['validation']['path'])
    if (validation.get('status')!='pass' or not manifest['proofs']['baseline_singleton_message_multisets']['all_equal']
        or not manifest['proofs']['pair_slot_coverage']['every_ordered_role_pair_covers_20_ordered_slot_pairs_once']
        or manifest.get('source_only_no_outcomes_read') is not True or manifest.get('main_E1_E2_selection_and_PK_unchanged') is not True):
        raise ValueError('Missing fixed baseline-reuse or ordered-pair-balance proof')
    seen=set();cells=set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);rid=row.get('record_id',row['id']);mask=row['subset_mask'];p=row['permutation'];label=row['label_map']
            cell=(row['item_id'],mask,p,label)
            if (row['item_id'] not in ids or rid in seen or cell in cells or mask.bit_count()!=2 or not 0<=mask<32
                or p not in range(10,20) or label not in (0,1) or row['positive_letter'] not in ('A','B')
                or row['affine_multiplier'] not in (2,3) or not 0<=row['affine_offset']<5
                or p!=10+(row['affine_multiplier']-2)*5+row['affine_offset']
                or row['slot_role_order']!=[manifest['roles'][(row['affine_multiplier']*s+row['affine_offset'])%5] for s in range(5)]):
                raise ValueError('Additional input cell/affine identity differs')
            seen.add(rid);cells.add(cell)
    if len(cells)!=4000:raise ValueError('Additional input factorial panel is incomplete')
    verifier.check(path,common.sha(path))
    return manifest,source


def prepare(base,e1_plan_path,prerequisite_path,manifest_path,amendment_path,source_roots=()):
    if any((base/'results').glob('*/e4_cache/*.jsonl')):
        raise ValueError('This pre-E4 supplement cannot start after E4 scoring artifacts exist')
    verifier=Verifier(base,source_roots);prereq=prerequisite(prerequisite_path,base,verifier)
    manifest,source=materials(manifest_path,base,verifier);decision=common.runtime(base)
    if (base/manifest['amendment']['path']).resolve()!=amendment_path.resolve() or manifest['amendment']['sha256']!=common.sha(amendment_path):
        raise ValueError('Material and queue amendment bindings differ')
    e1=common.load(e1_plan_path);final_path=e1_plan_path.parent/'final_status.json';final=common.load(final_path)
    if (e1.get('schema')!='voices_v3_E1_E2_queue_v1' or e1.get('dry_run') is not False or 'E1' not in e1['stages']
        or e1.get('pending_models') or len(e1['selected_models'])!=10
        or {m['repo_id'] for m in e1['selected_models']}!=set(common.CORE_MODELS)
        or final.get('status')!='finished' or final.get('all_requested_scores_complete') is not True
        or verifier.path(final.get('plan','')).resolve()!=e1_plan_path.resolve()
        or e1['runtime_decision_sha256']!=common.sha(base/'setup/runtime_decision.json')
        or e1['execution_protocol_sha256']!=common.sha(base/'execution_protocol.txt')):
        raise ValueError('Use the exact complete main E1 plan with all original ten frozen models')
    models={m['repo_id']:dict(m) for m in e1['selected_models']};original_jobs={}
    for old in e1['jobs']:
        if old['stage']!='E1':continue
        key=(old['model'],old['name'])
        if key in original_jobs:raise ValueError('Duplicate original E1 score job')
        original_jobs[key]=dict(old)
    expected={(r,n) for r in common.CORE_MODELS for n in (['E1_R1'] if r==R1_ONLY else ['E1_R0','E1_R1'])}
    if set(original_jobs)!=expected:raise ValueError('Original E1 readout panel differs from fixed 9 R0 + 10 R1')
    complete={(j['model'],j['name']) for j in final['jobs'] if j.get('status')=='complete' and j.get('stage')=='E1'}
    if complete!=expected:raise ValueError('E1 final receipt does not close every required original readout')
    for path in (e1_plan_path,final_path,prerequisite_path,amendment_path,base/'setup/runtime_decision.json',
        base/'execution_protocol.txt',base/'setup/download_queue.json',base/'setup/run_e1_e2_queue.py',
        base/'setup/run_e4_score_queue.py',Path(__file__),base/'code/inference_v3.py',base/'code/loader_v3.py'):
        verifier.check(path,common.sha(path))
    jobs=[]
    for repo in common.CORE_MODELS:
        model=models[repo];slug=repo.replace('/','--');statepath=base/'results'/slug/'e0/e0_driver_state.json'
        for key in ('calibration','E0_analysis_path','model_dir'):model[key]=str(verifier.path(model[key]))
        state=common.load(statepath);lock=common.load(model['calibration']);row=prereq[repo]['E1']
        if (model['R0_operational']!=(repo!=R1_ONLY) or state.get('status')!='complete'
            or state['revision']!=model['revision'] or state['selected_remedy']!=model['remedy']
            or common.sha(statepath)!=model['E0_state_sha256'] or lock['engine']!=model['engine']
            or lock['option_ids']!=model['option_ids'] or lock['batch_size']!=decision['batch_size']
            or row['revision']!=model['revision'] or row['calibration_sha256']!=model['calibration_sha256']):
            raise ValueError('Frozen E0/model/runtime differs from original E1/prerequisite report: '+repo)
        verifier.check(statepath,model['E0_state_sha256']);verifier.check(model['calibration'],model['calibration_sha256'])
        verifier.check(model['E0_analysis_path'],model['E0_analysis_sha256'])
        originals={};receipts={}
        for name in (['E1_R1'] if repo==R1_ONLY else ['E1_R0','E1_R1']):
            old=original_jobs[repo,name]
            for key in ('input','output'):old[key]=str(verifier.path(old[key]))
            readout=name[-2:];filename='e1_holdout200_prompts.jsonl' if readout=='R0' else 'e1_semantic64_prompts.jsonl'
            expected_raw=base/'results'/slug/'e1'/('holdout200_R0.jsonl' if readout=='R0' else 'semantic64_R1.jsonl')
            if (Path(old['input']).resolve()!=(base/'data'/filename).resolve() or Path(old['output']).resolve()!=expected_raw.resolve()
                or old['readout']!=readout or old['input_sha256']!=common.sha(old['input'])
                or old['rows']!=(124400 if readout=='R0' else 39808)):
                raise ValueError('Original E1 input/output/readout source differs: '+repo)
            closure=common.score_closure(old,model,decision)
            if closure is None:raise ValueError('Original E1 raw not closed: '+old['output'])
            originals[readout]=old['output'];receipts[readout]=closure
            for suffix,key in [('', 'output_sha256'),('.run.json','run_sha256'),('.summary.json','summary_sha256')]:
                verifier.check(old['output']+suffix,closure[key])
        output=base/'results'/slug/'e9_affinepair/e9_affinepair.jsonl';readout='R1' if repo==R1_ONLY else 'R0_R1'
        job=common.score_job(base,model,decision,{'path':source,'sha256':manifest['additional_prompts']['sha256'],'n_rows':4000},'E9_affine_pair','E9')
        command=job['command'];command[command.index('--output')+1]=str(output);command[command.index('--readouts')+1]=readout
        checkpoint=Path(model['model_dir'])/'snapshot_manifest.json'
        job.update(name='E9_affine_pair',readout=readout,output=str(output),log=str(output)+'.log',
            original_results=originals,original_closures=receipts,checkpoint_available=checkpoint.is_file(),
            conditional_option_mass_admission=state['qualification'].get('conditional_option_mass_admission',False))
        if checkpoint.is_file():verifier.check(checkpoint,common.sha(checkpoint))
        jobs.append(job)
    return {'schema':SCHEMA,'created_utc':common.now(),'initial_panel':list(common.CORE_MODELS),'jobs':jobs,
        'selected_models':list(models.values()),'runtime_decision':decision,'required_models':10,
        'readout_counts':{'R0_R1':9,'R1_only':1},'total_new_physical_rows':40000,
        'e1_plan':str(e1_plan_path),'e1_plan_sha256':common.sha(e1_plan_path),
        'prerequisite_report':str(prerequisite_path),'prerequisite_report_sha256':common.sha(prerequisite_path),
        'manifest':str(manifest_path),'manifest_sha256':common.sha(manifest_path),
        'amendment':str(amendment_path),'amendment_sha256':common.sha(amendment_path),
        'frozen_file_sha256':verifier.verified,'effect_based_selection':False,
        'scope':'Adaptive pair-position E9 supplement on fixed old20; no change to primary E1/E2/E4 or model selection',
        'original_stage_files_modified':False,'CPU_analysis_mode':'commands_only_deferred'}


def analysis_commands(base,plan,history):
    commands=[]
    for job in plan['jobs']:
        prefix=history/'model_analysis'/job['model'].replace('/','--')/'E9_affine_pairs'
        cmd=[sys.executable,str(base/'code/analyze_e9_affine_pairs_v3.py'),'--original-r1',job['original_results']['R1'],
            '--additional',job['output'],'--manifest',plan['manifest'],'--output-prefix',str(prefix),'--draws','10000']
        if 'R0' in job['original_results']:cmd+=['--original-r0',job['original_results']['R0']]
        commands.append({'model':job['model'],'command':cmd,'output_prefix':str(prefix),
            'create_output_parent':str(prefix.parent),'status':'command_only_not_executed'})
    return commands


def active_gpu_work():
    if sys.platform!='linux':raise RuntimeError('Actual E9 scoring requires the Linux GPU server')
    programs={'inference_v3.py','run_e0_v3.py','run_e0_v3_taskpilot.py','run_e0_v3_conditional.py',
        'run_e0_queue.py','run_e0_taskpilot_recovery.py','run_e0_conditional_queue.py','run_e1_e2_queue.py',
        'run_e4_score_queue.py','run_e4_phase_v3.py','run_e4_dynamics_queue_v3.py','run_e4_dynamics_v3.py',
        'run_e9_affine_queue_v3.py','benchmark_e4_gpu.py','continue_after_e0.py'}
    pids=[]
    for process in Path('/proc').glob('[0-9]*'):
        if int(process.name)==os.getpid():continue
        try:args=[x.decode(errors='ignore') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:continue
        if args and Path(args[0]).name.startswith('python') and any(Path(x).name in programs for x in args[1:]):pids.append(int(process.name))
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids.extend(int(s) for s in result.stdout.splitlines() if s.strip().isdigit() and int(s)!=os.getpid())
    return sorted(set(pids))


def execute(base,history,plan):
    if not os.environ.get('STY'):raise RuntimeError('E9 supplemental queue and GPU children must run inside screen')
    lease=base/'setup/e1_e2_queue.lock'
    if lease.exists():
        owner=common.load(lease)
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:lease.replace(history/'recovered_stale_GPU_lease.json')
        else:raise RuntimeError('Another E1/E2/E4 GPU queue owns the shared lease')
    if active_gpu_work():raise RuntimeError('Main inference or another CUDA/controller process is active')
    with lease.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'kind':'E9_affine_pair','history':str(history)},stream)
    models={m['repo_id']:m for m in plan['selected_models']};stamps={};events=history/'events.jsonl'
    state={'status':'running','started_utc':common.now(),'plan':str(history/'plan.json'),
        'plan_sha256':common.sha(history/'plan.json'),'jobs':[],'CPU_analysis_executed':False,'original_stage_files_modified':False}
    path=base/'setup/e9_affine_queue_status.json'
    try:
        common.write(path,state)
        for job in plan['jobs']:
            entry={k:job[k] for k in ('model','name','readout','output')};entry['started_utc']=common.now()
            state['current_job']=entry.copy();common.write(path,state)
            try:
                for source,digest in plan['frozen_file_sha256'].items():
                    current=(Path(source).stat().st_size,Path(source).stat().st_mtime_ns)
                    if stamps.get(source)!=current:
                        if common.sha(source)!=digest:raise ValueError('Frozen supplement source changed: '+source)
                        stamps[source]=current
                closure=common.score_closure(job,models[job['model']],plan['runtime_decision'])
                if closure is None:
                    if not job['checkpoint_available']:raise FileNotFoundError('Checkpoint unavailable; no automatic download or model replacement')
                    if active_gpu_work():raise RuntimeError('Another controller/CUDA process appeared between jobs')
                    raw=Path(job['output']);entry['resume_existing_bytes']=raw.stat().st_size if raw.exists() else 0
                    common.event(events,{'event':'score_start',**entry,'command':job['command']})
                    print('E9_AFFINE '+job['model'],flush=True)
                    entry['exit_code']=common.run_command(job['command'],job['log'])
                    if entry['exit_code']:raise RuntimeError('Additional scoring failed; partial raw/run/log retained')
                    closure=common.score_closure(job,models[job['model']],plan['runtime_decision'])
                    if closure is None:raise ValueError('Scoring returned without a closed 4000-row summary')
                entry.update(closure)
            except Exception as error:entry.update(status='execution_error_not_model_disqualification',error=str(error))
            entry['finished_utc']=common.now();state['jobs'].append(entry)
            common.event(events,{'event':'job_finished',**entry});common.write(path,state)
        state.update(status='finished_E9_affine_score_scope',finished_utc=common.now(),current_job=None,
            all_initial10_scores_complete=len(state['jobs'])==10 and all(r['status']=='complete' for r in state['jobs']),
            no_model_disqualification_from_execution_errors=True)
        common.write(path,state);common.write(history/'final_status.json',state)
    finally:
        if lease.exists() and common.load(lease).get('pid')==os.getpid():lease.unlink()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--e1-plan',type=Path,required=True);p.add_argument('--prerequisite-report',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--amendment-file',type=Path)
    p.add_argument('--source-root',action='append',default=[]);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();base=a.root.resolve();manifest=(a.manifest or base/'data/e9_affine_balance/manifest.json').resolve()
    amendment=(a.amendment_file or base/'provenance/E9_affine_pair_balance_20261031.txt').resolve()
    plan=prepare(base,a.e1_plan.resolve(),a.prerequisite_report.resolve(),manifest,amendment,a.source_root)
    if a.dry_run:
        print(json.dumps({'status':'read_only_plan','models':10,'rows':40000,'jobs':[
            {k:j[k] for k in ('model','readout','rows','output','checkpoint_available')} for j in plan['jobs']]}));return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history=base/'setup/e9_affine_queue_history'/stamp
    plan['dry_run']=False;common.write(history/'plan.json',plan)
    common.write(history/'deferred_analysis_commands.json',analysis_commands(base,plan,history))
    state=execute(base,history,plan)
    print(json.dumps({'status':state['status'],'all_initial10_scores_complete':state['all_initial10_scores_complete'],'plan':state['plan']}))
    if not state['all_initial10_scores_complete']:raise SystemExit(2)


if __name__=='__main__':main()
