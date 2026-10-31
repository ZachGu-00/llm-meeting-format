"""Executable E4 per-topic analysis; full observed matrices retained before summaries."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_v3 import read_closed,read_json,sha,execution_identity
from e4_graphs import pair_projection,validate_sts
from e4_simulation import cache_from_rows,predicted_caches,single_round,single_round_batch,five_group_outcome,design_metrics,simulate,save_cache

ROOT=Path(__file__).resolve().parents[1]

def initial_state(n,m,tau,agents=None):
    agents=list(range(n)) if agents is None else agents
    state=np.full(n,int(tau==-1),dtype=np.int8);state[agents[:m]]=int(tau==1)
    return state

def contrast_summary(observed,p0,pk):
    y,a,b=map(lambda x:np.asarray(x,dtype=float),(observed,p0,pk));den=float((y*y).sum())
    e0=float(((y-a)**2).sum());ek=float(((y-b)**2).sum())
    return {'n_contrasts':len(y),'mean_absolute_observed_difference':float(np.abs(y).mean()),
            'observed_delta_square_sum':den,'P0_error_square_sum':e0,'PK_error_square_sum':ek,
            'P0_R2_delta':1-e0/den if den>0 else None,'PK_R2_delta':1-ek/den if den>0 else None,
            'P0_unexplained_fraction_U':e0/den if den>0 else None,
            'P0_residual_U_rms':float(np.sqrt(e0/len(y))),
            'fraction_P0_residual_squared_explained_by_PK':1-ek/e0 if e0>0 else None,
            'zero_observed_contrasts':int((y==0).sum())}

def e4a(cache,predictions,meta):
    left,right=meta['g6_pasch_left'],meta['g6_pasch_right']
    if pair_projection(left)!=pair_projection(right):raise ValueError('G6 pair projection differs')
    records=[]
    for m,tau in itertools.product((2,3),(-1,1)):
        for selected in itertools.combinations(meta['g6_agent_ids'],m):
            state=initial_state(cache['n'],m,tau,list(selected))
            outcomes={method:[single_round(logits,design,state,tau) for design in (left,right)] for method,logits in predictions.items()}
            records.append({'n_target':m,'tau':tau,'target_agent_ids':list(selected),
                            'outcomes_left_right':outcomes,'delta_left_minus_right':{method:v[0]-v[1] for method,v in outcomes.items()}})
    d={name:[r['delta_left_minus_right'][name] for r in records] for name in predictions}
    return {'n_conditions':len(records),'records':records,'summary':contrast_summary(d['measured'],d['P0'],d['PK'])}

def e4b(cache,predictions,systems,topic_id,n_mappings=20):
    seed=int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+topic_id).encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(seed);mappings=[rng.permutation(cache['n']).tolist() for _ in range(n_mappings)]
    designs=[]
    for mapping in mappings:
        for system in systems:
            blocks=sorted(tuple(sorted(mapping[v] for v in block)) for block in system['blocks'])
            validate_sts(blocks);designs.append(blocks)
    configurations=[];all_dy=[];all_dp0=[];all_dpk=[]
    upper=np.triu_indices(len(systems),1)
    for m,tau in itertools.product((3,5,7),(-1,1)):
        state=initial_state(cache['n'],m,tau)
        arrays={name:single_round_batch(logits,designs,state,tau).reshape(n_mappings,len(systems)) for name,logits in predictions.items()}
        per_mapping=[]
        for mapping_id in range(n_mappings):
            outcomes={name:array[mapping_id] for name,array in arrays.items()}
            per_mapping.append(design_metrics(outcomes['measured'],outcomes['P0'],outcomes['PK']))
            for name,target in [('measured',all_dy),('P0',all_dp0),('PK',all_dpk)]:
                vector=outcomes[name];target.extend((vector[:,None]-vector[None,:])[upper].tolist())
        configurations.append({'n_target':m,'tau':tau,'outcomes_mapping_by_system':{name:array.tolist() for name,array in arrays.items()},
            'within_mapping_metrics':per_mapping,
            'secondary_mean_over_mappings_by_system':{name:array.mean(axis=0).tolist() for name,array in arrays.items()}})
    return {'n_systems':len(systems),'n_mappings':n_mappings,'mapping_seed':seed,'point_to_agent_mappings':mappings,
       'configurations':configurations,'summary':contrast_summary(all_dy,all_dp0,all_dpk),
       'primary_unit':'compare80 systems within each shared mapping and initial condition, then aggregate across mappings/topics',
       'uniform_mapping_expectation_note':'For single round, uniform expectation over all point-to-agent bijections is identical across STS15. Finite mapping-average signed differences are not intrinsic unlabeled-topology biases.'},designs

def e4c(cache,predictions,kernels,meta):
    records=[]
    for gi,group in enumerate(meta['five_person_sets']):
        triangles=list(itertools.combinations(group,3))
        if pair_projection([group]*3)!=pair_projection(triangles):raise ValueError('E4c projection differs')
        for bits,tau in itertools.product(itertools.product((0,1),repeat=5),(-1,1)):
            state=np.zeros(cache['n'],dtype=np.int8);state[group]=bits
            five=five_group_outcome(cache,group,state,kernels,tau)
            three={name:single_round(logits,triangles,state,tau) for name,logits in predictions.items()}
            records.append({'set_index':gi,'global_support_bits':bits,'tau':tau,
                'five_person_outcomes':five,'all_three_person_outcomes':three,
                'differences_five_minus_three':{'measured':five['measured']-three['measured'],'P0':five['P0']-three['P0'],
                    'PK_pair_truncated':five['PK_pair_truncated']-three['PK']}})
    values={k:[r['differences_five_minus_three'][k] for r in records] for k in ('measured','P0','PK_pair_truncated')}
    return {'n_conditions':len(records),'records':records,
        'summary':contrast_summary(values['measured'],values['P0'],values['PK_pair_truncated']),
        'PK_scope':'pair-truncated only: sum identifiable E1 pair kernels; no invented four-opponent high-order term',
        'outcome_normalization':'mean over agent-meeting incidences; 15 incidences in repeated-five vs30 in ten-triples',
        'tau_reversal_records_not_independent':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',action='append',required=True);p.add_argument('--results',action='append',required=True)
    p.add_argument('--kernel-analysis',required=True);p.add_argument('--cache-manifest',default=str(ROOT/'data/e4_cache_manifest.json'))
    p.add_argument('--sts',default=str(ROOT/'data/e4_graphs/sts15_all80.json'));p.add_argument('--topic-id');p.add_argument('--output-dir',required=True)
    p.add_argument('--mc-draws',type=int,default=0,help='0 means single-round only; full planned MC target100000 must be requested explicitly')
    p.add_argument('--mc-chunk',type=int,default=128);p.add_argument('--mc-rounds',type=int,default=6)
    a=p.parse_args()
    if len(a.inputs)!=len(a.results):raise ValueError('Each raw file requires its exact input')
    identity=execution_identity(a.results);kernel_source=read_json(a.kernel_analysis);kernels=kernel_source['point']['relation_kernels']
    if kernel_source.get('readout')!='R0':raise ValueError('This R0 E4 implementation requires matched R0 E1 kernels')
    if kernel_source['execution_identity']['critical_identity']!=identity['critical_identity']:
        raise ValueError('E1 kernel model/revision/remedy/options differ from the E4 model')
    meta=read_json(a.cache_manifest);sts=read_json(a.sts);topic_meta={t['topic_id']:t for t in meta['topics']}
    grouped={}
    fields=('status','cache_kind','focal_agent_id','peer_agent_ids','focal_stance','peer_stances','permutation','label_map','positive_letter','R0_y_AB','q')
    for source,raw in zip(a.inputs,a.results):
        for row in read_closed(source,raw):
            tid=row['topic_id']
            if a.topic_id and tid!=a.topic_id:continue
            grouped.setdefault(tid,[]).append({k:row[k] for k in fields})
    if not grouped:raise ValueError('No complete selected topic data')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for tid,rows in grouped.items():
        info=topic_meta[tid]
        if info['topic'] in kernel_source['topics']:raise ValueError('E1/E4 topic leakage')
        cache=cache_from_rows(rows);predictions=predicted_caches(cache,kernels)
        cache_path=out/(tid+'_caches.npz');save_cache(cache_path,cache,predictions)
        b,designs=e4b(cache,predictions,sts['systems'],tid)
        result={'topic_id':tid,'topic':info['topic'],'E1_topic_disjoint':True,'execution_identity':identity,
                'E1_execution_identity':kernel_source['execution_identity'],
                'cache_artifact':{'filename':cache_path.name,'sha256':sha(cache_path)},
                'option_mass':{'median':float(np.median([r['q'] for r in rows])),
                    'rows_below_0_5':sum(r['q']<.5 for r in rows),'n_rows':len(rows)},
                'E4a':e4a(cache,predictions,info),'E4b':b,'E4c':e4c(cache,predictions,kernels,info),
                'MC_completed':False,'topic_cluster_uncertainty_pending_aggregation':True,
                'E1_kernel_parameter_uncertainty_propagated':False,
                'input_hashes':{str(Path(f).resolve()):sha(f) for f in [*a.inputs,*a.results,a.kernel_analysis,a.cache_manifest,a.sts,__file__]}}
        if a.mc_draws:
            result['E4b_MC']={}
            for m,tau in itertools.product((3,5,7),(-1,1)):
                result['E4b_MC'][f'm{m}_tau{tau}']=simulate(predictions,designs,initial_state(15,m,tau),draws=a.mc_draws,
                    rounds=a.mc_rounds,chunk_size=a.mc_chunk,tau=tau,seed=20261030+m)
            result['MC_completed']=True;result['MC_full_planned_draws']=a.mc_draws>=100000
        with (out/(tid+'_E4_analysis.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps({'topic_id':tid,'E4a':result['E4a']['summary'],'E4b':b['summary'],'E4c':result['E4c']['summary'],'MC_completed':result['MC_completed']}),flush=True)

if __name__=='__main__':main()
