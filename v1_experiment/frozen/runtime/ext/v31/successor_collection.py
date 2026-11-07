"""Serial sealed G2 then G3, after source-verified G1 completion and lease release."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def execute(path):
    root=Path(path).resolve().parent;ext=root.parent;sys.path[:0]=[str(root),str(ext)]
    from common import load,sha,verify_bindings,write_new,atomic_status
    from checkpoint import prepare_checkpoint,verify_checkpoint
    from dialogue_provenance import closure,VerifiedSources
    from evidence_delta import create
    from panel_checkpoint import rotate,validate_ack
    from sealed_queue import audit_rows
    plan=load(path);base=Path(plan['base']);sys.path.insert(0,str(base/'setup'))
    from run_e1_e2_queue import score_closure
    if not os.environ.get('STY'):raise RuntimeError('Persistent screen required')
    lock=(root/'G2_G3_dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'plan_sha256':sha(path),'completed_jobs':[],'effect_analysis_executed':False}
    def save(**changes):
        state.update(changes);state['utc']=datetime.now(timezone.utc).isoformat();atomic_status(root/'G2_G3_status.json',state)
    def archived(artifact,repo,kind):
        with (root/'archive_index.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            info=create(base,ext,artifact,repo,kind,closure({str(artifact):sha(artifact)},base,ext),index_name='e7_index.json')
        ack=ext/'archives/off_server_ack'/(info['archive_sha256']+'.json')
        while not ack.exists():
            save(status='waiting_independent_backup',archive_sha256=info['archive_sha256']);time.sleep(10)
        validate_ack(info,load(ack));return info,load(ack)
    lease=base/'setup/e1_e2_queue.lock';acquired=False
    try:
        verify_bindings(plan['source_sha256'])
        archived(Path(path),'v31_G2_G3','v31_G2_G3_prospective')
        save(status='waiting_G1_sealed_completion',GPU_work_started=False)
        while not Path(plan['predecessor']).exists():
            status_path=root/'G1_status.json'
            if status_path.exists() and load(status_path)['status']=='G1_execution_error_preserve_all_evidence':
                raise RuntimeError('G1 error must be inspected; successor will not bypass it')
            time.sleep(10)
        prior=load(plan['predecessor'])
        if prior['status']!='G1_complete_sealed' or prior['sealed_rows']!=1470600 or len(prior['completed_jobs'])!=30:
            raise ValueError('G1 terminal coverage differs')
        bindings={plan['predecessor']:sha(plan['predecessor'])}
        for job in prior['completed_jobs']:
            p=Path(job['sealed_receipt']);verify_bindings(load(p)['source_sha256']);bindings[str(p)]=sha(p)
        handoff=root/'G1_to_G2_verified_handoff.json'
        write_new(handoff,{'status':'all_thirty_sealed_G1_tasks_reverified','source_sha256':bindings})
        while lease.exists():
            owner=load(lease)
            if owner.get('pid')!=prior['pid']:raise RuntimeError('Unexpected GPU lease owner')
            time.sleep(1)
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        if gpu:raise RuntimeError('Another CUDA process is active')
        with lease.open('x') as f:json.dump({'pid':os.getpid(),'kind':'v31_G2_G3_sealed','history':str(root)},f)
        acquired=True;verifier=VerifiedSources()
        for stage in plan['stage_order']:
            stage_receipts=[]
            for model in plan['models']:
                save(status='preparing_pinned_checkpoint',stage=stage,current_model=model['repo_id'],GPU_work_started=False)
                prepare_checkpoint(model,ext)
                _,missing=verify_checkpoint(model['execution_model_dir'],model,full_weights=True)
                if missing:raise RuntimeError('Pinned checkpoint remains incomplete')
                seals=[]
                for job in plan['jobs']:
                    if job['stage']!=stage or job['model']!=model['repo_id']:continue
                    for source,digest in plan['source_sha256'].items():verifier(source,digest)
                    if not load(base/'operations/cloud_goal_20261031/goal.json')['status'].startswith('running_'):
                        raise RuntimeError('Execution goal no longer authorizes running')
                    if shutil.disk_usage(root).free<12*1024**3+512*1024**2:raise OSError('Evidence storage reserve reached')
                    save(status='scoring_sealed_'+stage,current_job={k:job[k] for k in ('name','rows','output')},GPU_work_started=True)
                    result=score_closure(job,model,{'batch_size':16});elapsed=None
                    if result is None:
                        log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
                        with log.open('a') as f:
                            code=subprocess.run(job['command'],stdout=f,stderr=subprocess.STDOUT,
                                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')).returncode
                        elapsed=time.monotonic()-start
                        if code:raise RuntimeError('Native collection failed; preserve partial output')
                        result=score_closure(job,model,{'batch_size':16})
                    if result is None:raise ValueError('Native collection did not close')
                    quality=audit_rows(job);seal=Path(job['output']+'.sealed.json')
                    sources={str(path):sha(path),str(handoff):sha(handoff)}
                    sources.update({job['output']+s:sha(job['output']+s) for s in ('','.run.json','.summary.json')})
                    if not seal.exists():write_new(seal,{'schema':'voices_v31_sealed_native_score_v1','stage':stage,
                        'status':'sealed_quality_audited','model':model['repo_id'],'topic':job['topic_id'],
                        'rows':job['rows'],'quality':quality,'closure':result,'source_sha256':sources})
                    else:verify_bindings(load(seal)['source_sha256'])
                    seals.append(seal);state['completed_jobs'].append({'stage':stage,'model':model['repo_id'],'topic':job['topic_id'],
                        'sealed_receipt':str(seal),'rows':job['rows'],'wall_seconds':elapsed,
                        'rows_per_wall_second':job['rows']/elapsed if elapsed else None})
                    save(status='closed_sealed_'+stage+'_job')
                complete=root/'sealed'/stage/model['slug']/'completion.json'
                write_new(complete,{'status':'stage_model_sealed_no_effect_analysis','stage':stage,'repo_id':model['repo_id'],
                    'rows':len(seals)*49020,'source_sha256':{str(p):sha(p) for p in seals}})
                info,ack=archived(complete,model['repo_id'],'v31_'+stage+'_sealed_model');stage_receipts.append(complete)
                save(status='rotating_verified_weight_bytes')
                rotate(model['execution_model_dir'],model,info,ack,root/'rotation'/stage,ext/'model_cache',
                       stage+' current model scores independently backed up; subsequent stages restore identical pinned weights')
            write_new(root/(stage+'_complete.json'),{'status':stage+'_complete_sealed','rows':sum(load(p)['rows'] for p in stage_receipts),
                'source_sha256':{str(p):sha(p) for p in stage_receipts},'effect_analysis_executed':False})
        save(status='G2_G3_complete_sealed',current_model=None,current_job=None,GPU_work_started=False,full_v31_goal_complete=False)
        write_new(root/'G2_G3_complete.json',{**state,'source_sha256':{str(root/(s+'_complete.json')):sha(root/(s+'_complete.json')) for s in plan['stage_order']}})
    except BaseException as error:
        save(status='G2_G3_execution_error_preserve_all_evidence',error_type=type(error).__name__,error=str(error));raise
    finally:
        if acquired and lease.exists() and load(lease).get('pid')==os.getpid():
            lease.replace(root/('G2_G3_released_GPU_lease_'+str(os.getpid())+'.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);execute(p.parse_args().plan)
