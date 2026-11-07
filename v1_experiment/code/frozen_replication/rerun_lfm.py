"""Repeat the frozen LFM dialogue panel on the new GPU with a fresh calibration lock."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/workspace/voices_backup_recheck_20261106')
RUNROOT=ROOT/'LFM_rerun_3090'
sys.path[:0]=[str(ROOT/'runtime/ext'),str(ROOT/'runtime/base/code')]
from common import load,sha,write_new,atomic_status
import inference_v3 as inference
from dialogue_native import execute_run
from dialogue_bundle import archive_run,verify_bundle
from dialogue_analysis import summarize_run
def now():return datetime.now(timezone.utc).isoformat()
def initialize():
    import torch
    torch.set_num_threads(2)
    original=load(ROOT/'original_LFM_plan.json');m=original['model']
    engine=inference.Engine(str(ROOT/'model'),m['repo_id'],m['revision'],m['max_length'],m['remedy'],m['model_kind'],'eager')
    calibration=RUNROOT/'calibration.json';RUNROOT.mkdir(exist_ok=True,parents=True)
    data=ROOT/'runtime/calibration_prompts.jsonl';items=inference.normalized_items(data)
    if calibration.exists():
        lock=load(calibration);engine.validate_calibration(lock)
    else:
        lock=engine.calibrate(items,m['batch_size']);lock['input_file_sha256']=sha(data)
        lock['lock_content_sha256']=inference.canonical_hash(lock);write_new(calibration,lock)
    if lock['option_ids']!=m['option_ids']:raise ValueError('Selected output tokens changed')
    greedy=engine.greedy(items,lock['option_ids'],m['batch_size'])
    agree=sum(g['generated_first_token_id']==o['greedy_argmax_id'] for g,o in zip(greedy,lock['observations']))/len(items)
    formatting=sum(g['generated_first_token_selected_letter'] is not None for g in greedy)/len(items)
    if lock['median_q']<.70 or formatting<.90 or agree<.85:
        raise ValueError(f'New platform readout gate failed: {lock["median_q"]}, {formatting}, {agree}')
    old=load(ROOT/'runtime/original_token_calibration.json')
    selected=lock['selection']
    differences=[]
    for oldrow,newrow in zip(old['observations'],lock['observations']):
        if oldrow['record_id']!=newrow['record_id']:raise ValueError('Calibration items reordered')
        def value(r):
            return r['variant_logits']['A'][selected['A']['selected_index']]-r['variant_logits']['B'][selected['B']['selected_index']]
        differences.append(value(newrow)-value(oldrow))
    quality={'utc':now(),'q_median':lock['median_q'],'format_fraction':formatting,'independent_generate_vs_logits_agreement':agree,
             'n':len(items),'old_new_calibration_logit_differences':differences,'new_engine':engine.identity,
             'original_engine':m['engine'],'old_panel_intact':True,'new_rerun_preserves_prompts_seeds_and_state_update_rule':True}
    qualitypath=RUNROOT/'platform_validation.json'
    if not qualitypath.exists():write_new(qualitypath,quality)
    planpath=RUNROOT/'plan.json'
    if not planpath.exists():
        plan=deepcopy(original);plan['planned_runs']=288;plan['runs']=[deepcopy(r) for r in original['runs'] if 0<=r['run_number']<24]
        assert len(plan['runs'])==288
        plan['base']=str(ROOT/'runtime/base');plan['extension']=str(RUNROOT/'extension')
        plan['replication']={'utc':now(),'original_model_plan':str(ROOT/'original_LFM_plan.json'),'original_model_plan_sha256':sha(ROOT/'original_LFM_plan.json'),
                             'purpose':'new-platform full-panel replication after archival loss; original archives remain untouched',
                             'original_frozen_run_ids':[r['run_id'] for r in plan['runs']]}
        for r in plan['runs']:r['work_dir']=str(RUNROOT/'work'/r['run_id'])
        model=plan['model'];model['execution_model_dir']=str(ROOT/'model');model['engine']=engine.identity
        model['calibration']=str(calibration);model['calibration_sha256']=sha(calibration);model['option_ids']=lock['option_ids']
        plan['source_sha256']={str(p):sha(p) for p in list((ROOT/'runtime/ext').glob('*.py'))+list((ROOT/'runtime/base/code').glob('*.py'))}
        for p in [calibration,ROOT/'original_LFM_plan.json',data,ROOT/'model/snapshot_manifest.json',Path(__file__).resolve()]:plan['source_sha256'][str(p)]=sha(p)
        write_new(planpath,plan)
    plan=load(planpath)
    if plan['model']['engine']!=engine.identity:raise ValueError('Rerun engine changed after lock')
    print(json.dumps({'phase':'ready','q':lock['median_q'],'format':formatting,'agreement':agree,'plan_sha256':sha(planpath)}),flush=True)
    return engine,plan,planpath
def execute(engine,plan,planpath,item):
    runid=item['run_id'];status=RUNROOT/'run_status'/f'{runid}.json';status.parent.mkdir(exist_ok=True)
    if status.exists() and load(status).get('status')=='verified':
        info=load(status)['bundle'];verify_bundle(info['archive'],info);return load(status)
    cfg={**item,'model_plan':str(planpath),'model_plan_sha256':sha(planpath)}
    start=time.monotonic()
    execute_run(plan,cfg,engine,inference)
    # The legacy archive index is shared across workers. Serialize its update.
    with (RUNROOT/'archive.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        info=archive_run(planpath,runid)
        fcntl.flock(guard,fcntl.LOCK_UN)
    _,replayed=verify_bundle(info['archive'],info,return_events=True)
    row=summarize_run(replayed)
    result={'status':'verified','utc':now(),'run_id':runid,'seconds':time.monotonic()-start,'bundle':info,'summary':row}
    write_new(status,result)
    print(json.dumps({'phase':'run_complete','run_id':runid,'seconds':result['seconds'],'family':item['family'],'q_mean':row['q_mean']}),flush=True)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','worker'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not os.environ.get('STY'):raise RuntimeError('Run all model inference inside screen')
    engine,plan,planpath=initialize()
    if a.mode=='pilot':
        wanted=['E6v31_G6_d00_r00','E6v31_STS15_d00_r00'];items=[next(r for r in plan['runs'] if r['run_id']==n) for n in wanted]
    else:
        items=[r for r in plan['runs'] if r['run_number']%a.workers==a.worker]
    allrows=[]
    for item in items:
        r=execute(engine,plan,planpath,item);allrows.append(r)
        atomic_status(RUNROOT/f'worker_{a.mode}_{a.worker}.json',{'utc':now(),'completed_in_worker':len(allrows),'planned_in_worker':len(items),'last_run_id':item['run_id'],'status':'running'})
    atomic_status(RUNROOT/f'worker_{a.mode}_{a.worker}.json',{'utc':now(),'completed_in_worker':len(allrows),'planned_in_worker':len(items),'status':'complete'})
    if a.mode=='pilot':
        atomic_status(RUNROOT/'pilot.json',{'utc':now(),'runs':[{'run_id':r['run_id'],'seconds':r['seconds'],'summary':r['summary']} for r in allrows]})
if __name__=='__main__':main()

