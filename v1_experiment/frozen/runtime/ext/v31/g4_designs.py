"""M3: old-cache empirical extreme policies, fixed Pasch pair and hash controls."""
import argparse
import hashlib
from pathlib import Path
import sys

MODELS=['Qwen/Qwen3.5-4B','LiquidAI/LFM2.5-1.2B-Instruct','microsoft/Phi-4-mini-instruct']


def rank(*parts):return hashlib.sha256(('voices-v31-M3-v1|'+'|'.join(map(str,parts))).encode()).hexdigest()


def select(values,ids,context):
    if len(values)!=80 or len(set(ids))!=80:raise ValueError('All original eighty systems required')
    maximum=min(range(80),key=lambda i:(-values[i],rank(*context,'tie',ids[i])))
    minimum=min((i for i in range(80) if i!=maximum),key=lambda i:(values[i],rank(*context,'tie',ids[i])))
    controls=sorted((i for i in range(80) if i not in (minimum,maximum)),key=lambda i:rank(*context,'control',ids[i]))[:8]
    choices=[('empirical_min',minimum),('empirical_max',maximum)]+[(f'random_{j:02d}',i) for j,i in enumerate(controls)]
    return [{'design_id':name,'source_system_id':ids[i],'original_development_outcome':values[i]} for name,i in choices]


def build(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path[:0]=[str(ext),str(base/'code')]
    from common import load,sha,verify_bindings,write_new
    from dialogue_core import seed
    from e4_graphs import validate_sts,pair_projection
    old_path=base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json';old=load(old_path)
    agent_path=base/'data/e4_agents.json';meta_path=base/'data/e4_cache_manifest.json';sts_path=base/'data/e4_graphs/sts15_all80.json'
    topics=sorted(load(agent_path),key=lambda t:t['topic_id']);meta={t['topic_id']:t for t in load(meta_path)['topics']}
    systems=load(sts_path)['systems'];ids=[s['id'] for s in systems];by_id={s['id']:s['blocks'] for s in systems}
    sources={str(p):sha(p) for p in [old_path,agent_path,meta_path,sts_path,Path(__file__),root/'research_plan_v31_user.md',ext/'dialogue_core.py',base/'code/e4_graphs.py']}
    result={'schema':'voices_v31_M3_empirical_design_policies_v1','models':{},'source_sha256':sources,
            'user_confirmed_total_runs':1800,'runs_per_model':600,'rounds_per_run':2,'runs_per_policy':50,
            'selection':'Each fixed old topic/mapping/initial-condition context: measured minimum, measured maximum, eight hash-ranked distinct controls; plus the two original G6 Pasch designs',
            'sealed_outputs_read':False,'all_selection_data_are_original_six_development_topics':True,
            'systematic_model_change_from_v3':'Phi replaces Granite as requested by v3.1; original E6 registration preserved'}
    for repo in MODELS:
        points={}
        for job in old['jobs']:
            if job['model']!=repo:continue
            p=Path(job['deferred_analysis']['output_dir'])/(job['topic_id']+'_E4_analysis.json');points[job['topic_id']]=load(p)
            verify_bindings(points[job['topic_id']]['input_hashes']);sources[str(p)]=sha(p)
        contexts=[]
        for run in range(50):
            ti=run%6;cycle=run//6;topic=topics[ti];tid=topic['topic_id'];info=meta[tid]
            minority=(3,5,7)[(cycle+ti)%3];tau=1 if (cycle+ti)%2==0 else -1
            mapping_index=seed('mapping',tid,run)%20;point=points[tid]['E4b'];mapping=point['point_to_agent_mappings'][mapping_index]
            condition=next(c for c in point['configurations'] if c['n_target']==minority and c['tau']==tau)
            values=condition['outcomes_mapping_by_system']['measured'][mapping_index]
            selected=select(values,ids,(repo,tid,run,minority,tau,mapping_index))
            for choice in selected:
                choice['blocks']=sorted([sorted(mapping[v] for v in b) for b in by_id[choice['source_system_id']]])
                validate_sts(choice['blocks'])
            if any(pair_projection(c['blocks'])!=pair_projection(selected[0]['blocks']) for c in selected):raise ValueError('Selected STS pair projections differ')
            if pair_projection(info['g6_pasch_left'])!=pair_projection(info['g6_pasch_right']):raise ValueError('Original Pasch projections differ')
            contexts.append({'run':run,'topic_id':tid,'minority_count':minority,'target_stance':tau,
                             'mapping_index':mapping_index,'designs':selected,
                             'G6_minority_count':(2,3)[(cycle+ti)%2],
                             'G6_target_stance':1 if (cycle//2+ti)%2==0 else -1})
        result['models'][repo]=contexts
    result['native_workload']={'runs':1800,'meetings':3*(500*70+100*8),'utterances':3*3*(500*70+100*8),
                               'R0_readouts':3*42*(500*70+100*8),
                               'generation_batch_size':1,'readout_batch_size':16,
                               'ETA_policy':'Measure the first complete native run; budget estimates are not measured runtime'}
    write_new(root/'gate1/M3_design_selection.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();r=build(a.base,a.extension);print(r['native_workload'])
