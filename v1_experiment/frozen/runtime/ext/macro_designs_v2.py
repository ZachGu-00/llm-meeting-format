"""PK-only design selection preserves within-mapping contrasts before averaging."""
import argparse
import hashlib
import itertools
from pathlib import Path
import sys
import numpy as np
from common import load,sha,verify_bindings,write_new
from low_order import cache as low_order_cache

def farthest_designs(features,ids,count=10):
    features=np.asarray(features,dtype=float)
    if (features.ndim!=2 or features.shape[0]!=len(ids) or len(set(ids))!=len(ids)
        or not 2<=count<=len(ids) or features.shape[1]<1 or not np.isfinite(features).all()):
        raise ValueError('Complete finite candidate-by-context features required')
    distance=np.sqrt(np.mean((features[:,None,:]-features[None,:,:])**2,axis=-1))
    pairs=list(itertools.combinations(range(len(ids)),2))
    a,b=min(pairs,key=lambda p:(-distance[p[0],p[1]],ids[p[0]],ids[p[1]]));selected=[a,b]
    while len(selected)<count:
        remaining=[i for i in range(len(ids)) if i not in selected]
        best=min(remaining,key=lambda i:(-min(distance[i,j] for j in selected),ids[i]));selected.append(best)
    return [ids[i] for i in selected],distance

def prepare(base,extension,model_id,output):
    base,root=Path(base).resolve(),Path(extension).resolve();output=Path(output)
    if output.exists():
        result=load(output);verify_bindings(result['source_sha256']);return result
    sys.path.insert(0,str(base/'code'))
    from analyze_v3 import read_closed,execution_identity
    from e4_simulation import predicted_caches,single_round_batch
    from run_e4_analysis import initial_state
    from e4_graphs import validate_sts
    e9_path=root/'plans/e9_full/plan.json';e9=load(e9_path)
    score_path=Path(e9['e4_predecessor']['score_plan']);score=load(score_path)
    model=next(m for m in score['selected_models'] if m['repo_id']==model_id)
    kernel_path=Path(model['kernel_analysis']);kernel=load(kernel_path);verify_bindings(kernel['file_sha256'])
    graphs_path=base/'data/e4_graphs/sts15_all80.json';graphs=load(graphs_path)
    systems=sorted(graphs['systems'],key=lambda s:s['id']);ids=[s['id'] for s in systems]
    if len(ids)!=80:raise ValueError('Require original all80 systems')
    for system in systems:validate_sts(system['blocks'])
    sources={str(p):sha(p) for p in (e9_path,score_path,kernel_path,graphs_path,Path(__file__),Path(__file__).with_name('low_order.py'),root/'ADVANCEMENT_PROTOCOL.md')}
    sources.update(kernel['file_sha256']);features=[];contexts=[];topic_records=[]
    jobs=sorted([j for j in score['jobs'] if j['model']==model_id],key=lambda j:j['topic_id'])
    if len(jobs)!=6:raise ValueError('All six frozen topic inputs are required')
    for job in jobs:
        identity=execution_identity([job['output']])
        if identity['critical_identity']!=kernel['execution_identity']['critical_identity']:
            raise ValueError('Low-order cache/E1 native model identity differs')
        selected=[]
        for row in read_closed(job['input'],job['output']):
            if row['cache_kind'] in ('baseline','pair'):selected.append(row)
        if len(selected)!=3420:raise ValueError('Original low-order cache scope differs')
        cache=low_order_cache(selected,15);logits=predicted_caches(cache,kernel['point']['relation_kernels'])['PK']
        seed=int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+job['topic_id']).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed);mappings=[rng.permutation(15).tolist() for _ in range(20)]
        designs=[]
        for mapping in mappings:
            for system in systems:
                blocks=sorted(tuple(sorted(mapping[v] for v in b)) for b in system['blocks']);validate_sts(blocks);designs.append(blocks)
        for n_target,tau in itertools.product((3,5,7),(-1,1)):
            state=initial_state(15,n_target,tau)
            values=single_round_batch(logits,designs,state,tau).reshape(20,80)
            # Each column remains a specific shared topic/mapping/state context.
            features.append((values-values.mean(axis=1,keepdims=True)).T)
            contexts.extend({'topic_id':job['topic_id'],'mapping_index':i,'n_target':n_target,'tau':tau} for i in range(20))
        topic_records.append({'topic_id':job['topic_id'],'mapping_seed':seed,'point_to_agent_mappings':mappings})
        for p in (job['input'],job['output'],job['output']+'.run.json',job['output']+'.summary.json'):sources[str(p)]=sha(p)
    feature=np.concatenate(features,axis=1);chosen,distances=farthest_designs(feature,ids)
    artifact=output.with_suffix('.npz');output.parent.mkdir(parents=True,exist_ok=True)
    if artifact.exists():
        with np.load(artifact,allow_pickle=False) as old:
            if not np.array_equal(old['context_centered_PK'],feature) or not np.array_equal(old['pairwise_RMS_distance'],distances):
                raise ValueError('Existing PK-only design vectors differ')
    else:
        with artifact.open('xb') as f:np.savez_compressed(f,context_centered_PK=feature,pairwise_RMS_distance=distances,system_ids=np.array(ids))
    sources[str(artifact)]=sha(artifact)
    result={'schema':'voices_v3_PK_macro_selection_v1','status':'complete_verified','repo_id':model_id,
        'selected_system_ids':chosen,'all_system_ids':ids,'contexts':contexts,'topics':topic_records,
        'maximum_pairwise_RMS_distance':float(distances.max()),'artifact':str(artifact),
        'method':'farthest-first on context-centered PK vectors; retain all shared mapping/state contexts before RMS aggregation',
        'measured_joint_group_outcomes_used':False,'low_order_readouts':'original E4 empty/single-peer R0 only',
        'kernel':'independent frozen E1 relation kernel; no fitting on joint-group outcomes',
        'timing':'prospective selection for E6; does not claim to precede the historical E4 launch',
        'source_sha256':sources}
    write_new(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=prepare(a.base,a.extension,a.model,a.output);print({'model':r['repo_id'],'selected_system_ids':r['selected_system_ids']})
