"""Pre-E4 amendment: independent mapping strata, common noise within each map."""
import hashlib
import itertools
import numpy as np
from e4_simulation import simulate as numpy_simulate
from run_e4_analysis import initial_state


def mapping_seed(topic_id,n_target,tau,mapping_id):
    text=f'voices_v3_dynamic_independent_maps_v1|{topic_id}|{n_target}|{tau}|{mapping_id}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)%(2**63-1)


def combine_independent_estimates(estimates):
    """Conditional SE for an equally weighted mean of independent MC strata."""
    means=np.array([r['means'] for r in estimates]);ses=np.array([r['Monte_Carlo_standard_errors'] for r in estimates])
    if means.shape!=ses.shape or not np.isfinite(means).all() or not np.isfinite(ses).all():raise ValueError('Invalid stratified MC estimates')
    n=len(means)
    return {'means':means.mean(axis=0).tolist(),
            'Monte_Carlo_standard_errors':(np.sqrt(np.sum(ses**2,axis=0))/n).tolist(),
            'between_mapping_SD_of_estimated_means':means.std(axis=0,ddof=1 if n>1 else 0).tolist(),
            'between_mapping_SD_scope':'descriptive heterogeneity, includes finite per-map MC noise; not the Monte Carlo SE of the stratified mean'}


def combine_mapping_runs(runs,seeds,total_draws):
    if not runs:raise ValueError('No mapping results')
    first=runs[0];nmap=len(runs)
    for r in runs:
        if any(r[key]!=first[key] for key in ('methods','n_designs','n_agents','rounds','draws','tau')):
            raise ValueError('Mapping strata differ in dimensions or allocation')
    if sum(r['draws'] for r in runs)!=total_draws or len(set(seeds))!=nmap:raise ValueError('Invalid independent mapping allocation')
    keys=('outcomes','paired_difference_to_design0','paired_residual_U_measured_minus_other_methods')
    output={key:first[key] for key in ('methods','n_agents','rounds','tau')}
    output.update(n_designs=nmap*first['n_designs'],n_systems=first['n_designs'],n_mappings=nmap,
                  draws=first['draws'],draws_scope='trajectories per mapping, not total per design',
                  total_draws_per_design=total_draws,mapping_seeds=list(seeds),
                  mapping_runs=runs,MC_full_planned_total_draws=total_draws>=100000)
    for key in keys:
        # Existing analysis expects design index = mapping * n_systems + system.
        output[key]={field:np.concatenate([np.asarray(r[key][field]) for r in runs],axis=1).tolist()
                     for field in ('means','Monte_Carlo_standard_errors')}
    output['stratified_means_by_system']={key:combine_independent_estimates([r[key] for r in runs]) for key in keys}
    output['paired_reference_scope']='design0 separately within each mapping; no cross-mapping reference contrast'
    output['interval_scope']='Monte Carlo error conditional on fixed caches/kernels and these independent mapping strata'
    output['common_random_numbers']='shared across 80 systems and measured/P0/PK within each map; independent seeds between maps'
    return output


def simulate_sts_stratified(caches,systems,mappings,topic_id,n_target,tau,total_draws=100000,
                            rounds=6,chunk_size=128,simulator=numpy_simulate,on_mapping=None):
    nmap=len(mappings)
    if nmap<1 or total_draws<2*nmap or total_draws%nmap:raise ValueError('Budget must divide evenly among mappings, with at least two trajectories per map')
    draws=total_draws//nmap;seeds=[];runs=[]
    for mi,mapping in enumerate(mappings):
        seed=mapping_seed(topic_id,n_target,tau,mi);seeds.append(seed)
        designs=[[tuple(sorted(mapping[v] for v in block)) for block in system['blocks']] for system in systems]
        kwargs=dict(rounds=rounds,draws=draws,chunk_size=chunk_size,seed=seed,tau=tau)
        call=lambda:simulator(caches,designs,initial_state(len(mapping),n_target,tau),**kwargs)
        run=on_mapping(mi,kwargs,call) if on_mapping else call()
        runs.append(run)
    return combine_mapping_runs(runs,seeds,total_draws)


def reindex_triple_caches(caches,agent_ids):
    agents=sorted(agent_ids)
    if len(set(agents))!=len(agents):raise ValueError('Duplicate selected agent')
    local={name:np.full((len(agents),len(agents),len(agents),8),np.nan) for name in caches}
    for j in range(len(agents)):
        for k,l in itertools.combinations([i for i in range(len(agents)) if i!=j],2):
            for name,cache in caches.items():local[name][j,k,l]=cache[agents[j],agents[k],agents[l]]
    return local,{agent:i for i,agent in enumerate(agents)}


def simulate_pasch(caches,meta,draws=100000,rounds=6,chunk_size=128,simulator=numpy_simulate,on_condition=None):
    local,index=reindex_triple_caches(caches,meta['g6_agent_ids']);agents=sorted(index)
    if len(agents)!=6:raise ValueError('Pasch dynamics must use exactly six agents')
    designs=[[[index[v] for v in block] for block in meta[key]] for key in ('g6_pasch_left','g6_pasch_right')]
    records=[]
    for n_target,tau in itertools.product((2,3),(-1,1)):
        for selected in itertools.combinations(agents,n_target):
            key=f'm{n_target}_tau{tau}_agents'+('_'.join(map(str,selected)))
            seed=mapping_seed(meta['topic_id'],n_target,tau,key)
            kwargs=dict(draws=draws,rounds=rounds,chunk_size=chunk_size,seed=seed,tau=tau)
            state=initial_state(6,n_target,tau,[index[v] for v in selected])
            call=lambda:simulator(local,designs,state,**kwargs)
            result=on_condition(key,kwargs,call) if on_condition else call()
            records.append({'condition_id':key,'n_target':n_target,'tau':tau,'target_agent_ids':list(selected),'simulation':result})
    return {'records':records,'n_conditions':len(records),'global_to_local_agent_ids':index,
            'outcome_denominator':6,'irrelevant_frozen_agents_included':False,
            'draws_per_design_per_condition':draws,'MC_full_planned_draws':draws>=100000}
