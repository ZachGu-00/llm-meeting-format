"""Prospective G2 new-topic R0 and fixed-hash G3 original-topic R1 collection."""
import argparse
import hashlib
from pathlib import Path
import sys


def build(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path[:0]=[str(ext),str(base/'setup')]
    from common import load,sha,verify_bindings,write_new
    from dialogue_provenance import closure
    from run_e1_e2_queue import score_job,runtime
    target=root/'G2_G3_plan.json'
    if target.exists():
        plan=load(target);verify_bindings(plan['source_sha256']);return plan
    old_path=base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json'
    old=load(old_path);new_path=root/'data/M1_manifest.json';new=load(new_path)
    original_path=base/'data/e4_cache_manifest.json';original=load(original_path)
    models=[]
    for source in old['selected_models']:
        model=dict(source);model['execution_model_dir']=str(ext/'model_cache'/model['slug'])
        if not model['R0_operational'] or not model['R1_candidate_scoring_valid']:raise ValueError('Matched qualified readouts required')
        models.append(model)
    seed='voices-v31-G3-topics-20261101-v1'
    rank=lambda t:hashlib.sha256((seed+'|'+t['topic_id']).encode()).hexdigest()
    chosen=sorted(original['topics'],key=rank)[:2]
    decision=runtime(base);jobs=[];sources={str(p):sha(p) for p in [old_path,new_path,original_path,root/'G1_plan.json',
        root/'research_plan_v31_user.md',root/'g2_g3_plan.py',root/'successor_collection.py',root/'sealed_queue.py',
        ext/'common.py',ext/'checkpoint.py',ext/'panel_checkpoint.py',ext/'evidence_delta.py',ext/'dialogue_provenance.py',
        base/'code/inference_v3.py',base/'code/loader_v3.py',base/'setup/run_e1_e2_queue.py',base/'setup/runtime_decision.json']}
    for stage,topics,readout in [('G2',new['topics'],'R0'),('G3',chosen,'R1')]:
        for model in models:
            for topic in topics:
                spec=dict(topic['output']);path=Path(spec['path']);path=path if path.is_absolute() else base/path
                if sha(path)!=spec['sha256'] or spec['n_rows']!=49020:raise ValueError('Topic input changed')
                sources[str(path)]=spec['sha256']
                job=score_job(base,model,decision,{**spec,'path':path},'E1_R1' if readout=='R1' else topic['topic_id'],stage)
                output=root/'sealed'/stage/model['slug']/(topic['topic_id']+'_'+readout+'.jsonl')
                cmd=job['command'];cmd[cmd.index('--output')+1]=str(output);cmd[cmd.index('--model-dir')+1]=model['execution_model_dir']
                job.update(name=topic['topic_id'],topic_id=topic['topic_id'],output=str(output),log=str(output)+'.log',
                           effect_analysis_allowed=False,batch_size=16)
                if output.exists():raise ValueError('New stages must be prospectively registered')
                jobs.append(job)
    plan={'schema':'voices_v31_G2_G3_sealed_registration_v1','base':str(base),'extension':str(ext),'models':models,'jobs':jobs,
          'G2_topics':new['n_topics'],'G2_jobs':6*new['n_topics'],'G2_rows':6*new['n_topics']*49020,
          'G3_topics':[t['topic_id'] for t in chosen],'G3_hash_seed':seed,'G3_jobs':12,'G3_rows':588240,
          'source_sha256':closure(sources,base,ext),'predecessor':str(root/'G1_complete.json'),
          'stage_order':['G2','G3'],'runtime':'native BF16/eager/b16, unchanged calibration',
          'sealing':'Only quality/count/hash checks; neither runner computes effects',
          'rotation':'Pinned reproducible weights only after each stage-model raw-data backup is independently verified; later stages restore identical hashes'}
    if new['n_topics']!=14 or len(jobs)!=96:raise ValueError('Expected fourteen selected topics and 96 combined tasks')
    write_new(target,plan);return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();r=build(a.base,a.extension);print({k:r[k] for k in ['G2_jobs','G2_rows','G3_jobs','G3_rows','G3_topics']})
