"""E4 topic-cluster intervals, with joint E1-kernel propagation for one round.

E1 topics and held-out E4 topics are resampled independently. A draw's three
relationship kernels are jointly estimated and shared across all E4 topics.
STS contrasts always stay within a common mapping and initial configuration.
Dynamic intervals are conditional on fitted kernels and saved Monte Carlo means.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from analyze_v3 import PAIRS, read_json, sha
from e4_simulation import load_cache, sigmoid
from run_e4_analysis import initial_state

ROOT=Path(__file__).resolve().parents[1]
RELATIONS=tuple(PAIRS)
CORE_STATS=('observed_mean_square_contrast','P0_mean_square_error','PK_mean_square_error')
SELECTION_STATS=('P0_selection_regret_maximize','PK_selection_regret_maximize',
                 'P0_selected_gain_over_uniform','PK_selected_gain_over_uniform')


def kernel_bootstrap(analysis,draws,seed):
    """Reconstruct item-level ratio components; keep the three kernels coupled."""
    ids=analysis['item_ids'];labels=np.array(analysis['topics']);v=np.array(analysis['item_singletons'])
    if len(ids)!=len(labels) or v.shape!=(len(ids),5):raise ValueError('Invalid E1 item-level kernel data')
    num=[];den=[]
    for relation,pairs in PAIRS.items():
        num.append([np.mean([analysis['mobius_by_item'][item][str((1<<a)|(1<<b))]
                            for a,b in pairs]) for item in ids])
        den.append(np.mean([.5*(np.abs(v[:,a])+np.abs(v[:,b])) for a,b in pairs],axis=0))
    num=np.array(num).T;den=np.array(den).T
    point=np.divide(num.sum(axis=0),den.sum(axis=0),out=np.full(3,np.nan),where=den.sum(axis=0)>0)
    recorded=np.array([analysis['point']['relation_kernels'][r] for r in RELATIONS],dtype=float)
    if not np.isfinite(point).all() or not np.allclose(point,recorded,rtol=1e-11,atol=1e-12):
        raise ValueError('E1 item-level kernels do not reproduce the recorded estimate')
    topics=sorted(set(labels));rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(topics),np.full(len(topics),1/len(topics)),size=draws)
    tn=np.array([num[labels==t].sum(axis=0) for t in topics])
    td=np.array([den[labels==t].sum(axis=0) for t in topics])
    numerator=weights@tn;denominator=weights@td
    samples=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    return point,samples,weights,topics


class OutcomeCompiler:
    """Exact sparse incidence averages of sigmoid(alpha+v+kernel correction)."""
    def __init__(self,cache):
        self.cache=cache;self.cells={};self.meetings={};self.bases=[];self.coefficients=[]
        self.meeting_cells=[];self.design_meetings=[];self.taus=[]

    def cell(self,j,peers,state):
        peers=tuple(sorted(peers));sj=int(state[j]);bits=(sj,*(int(state[k]) for k in peers))
        key=(j,peers,bits)
        if key not in self.cells:
            alpha=self.cache['alpha'][j,sj]
            v={k:self.cache['pair'][j,k,2*sj+int(state[k])]-alpha for k in peers}
            base=alpha+sum(v.values());coef=np.zeros(3)
            for k,l in itertools.combinations(peers,2):
                relation=0 if state[k]!=sj and state[l]!=sj else 2 if state[k]==sj and state[l]==sj else 1
                coef[relation]+=(1-2*sj)*.5*(abs(v[k])+abs(v[l]))
            if not np.isfinite(base) or not np.isfinite(coef).all():raise ValueError('Missing baseline/pair cache')
            self.cells[key]=len(self.bases);self.bases.append(base);self.coefficients.append(coef)
        return self.cells[key]

    def meeting(self,block,state):
        block=tuple(sorted(block));key=(block,tuple(int(state[j]) for j in block))
        if key not in self.meetings:
            self.meetings[key]=len(self.meeting_cells)
            self.meeting_cells.append([self.cell(j,[k for k in block if k!=j],state) for j in block])
        return self.meetings[key]

    def design(self,blocks,state,tau):
        self.design_meetings.append([self.meeting(b,state) for b in blocks]);self.taus.append(tau)

    def finish(self):
        def matrix(groups,ncols):
            r=[];c=[];w=[]
            for i,indices in enumerate(groups):
                r.extend([i]*len(indices));c.extend(indices);w.extend([1/len(indices)]*len(indices))
            return csr_matrix((w,(r,c)),shape=(len(groups),ncols))
        self.meeting_matrix=matrix(self.meeting_cells,len(self.bases))
        self.design_matrix=matrix(self.design_meetings,len(self.meeting_cells))
        self.bases=np.array(self.bases);self.coefficients=np.array(self.coefficients)
        self.taus=np.array(self.taus)[:,None]

    def evaluate(self,kernels):
        kernels=np.atleast_2d(kernels)
        p=sigmoid(self.bases[:,None]+self.coefficients@kernels.T)
        outcomes=self.design_matrix@(self.meeting_matrix@p)
        return (1-self.taus)/2+self.taus*outcomes


def compile_topic(cache,result,meta,systems):
    compiler=OutcomeCompiler(cache);sections={};expected={};start=0
    for r in result['E4a']['records']:
        state=initial_state(cache['n'],r['n_target'],r['tau'],r['target_agent_ids'])
        for blocks in (meta['g6_pasch_left'],meta['g6_pasch_right']):compiler.design(blocks,state,r['tau'])
    sections['E4a']=(slice(start,len(compiler.design_meetings)),(len(result['E4a']['records']),2));start=len(compiler.design_meetings)
    expected['E4a']={name:np.array([r['outcomes_left_right'][name] for r in result['E4a']['records']])
                       for name in ('measured','P0','PK')}
    designs=[]
    for mapping in result['E4b']['point_to_agent_mappings']:
        for system in systems:designs.append([tuple(sorted(mapping[v] for v in b)) for b in system['blocks']])
    for r in result['E4b']['configurations']:
        state=initial_state(cache['n'],r['n_target'],r['tau'])
        for blocks in designs:compiler.design(blocks,state,r['tau'])
    shape=(len(result['E4b']['configurations']),result['E4b']['n_mappings'],result['E4b']['n_systems'])
    sections['E4b']=(slice(start,len(compiler.design_meetings)),shape);start=len(compiler.design_meetings)
    expected['E4b']={name:np.array([r['outcomes_mapping_by_system'][name] for r in result['E4b']['configurations']])
                       for name in ('measured','P0','PK')}
    for r in result['E4c']['records']:
        group=meta['five_person_sets'][r['set_index']];state=np.zeros(cache['n'],dtype=np.int8)
        state[group]=r['global_support_bits']
        compiler.design([group],state,r['tau'])
        compiler.design(list(itertools.combinations(group,3)),state,r['tau'])
    sections['E4c']=(slice(start,len(compiler.design_meetings)),(len(result['E4c']['records']),2))
    expected['E4c']={name:np.array([[r['five_person_outcomes']['PK_pair_truncated' if name=='PK' else name],
                                  r['all_three_person_outcomes'][name]] for r in result['E4c']['records']])
                       for name in ('measured','P0','PK')}
    compiler.finish()
    return compiler,sections,expected


def statistics(part,y,p0,pk):
    """Per-topic sufficient means. PK has a trailing kernel-draw dimension."""
    y=np.asarray(y);p0=np.asarray(p0);pk=np.asarray(pk)
    if pk.shape[:-1]!=y.shape or p0.shape!=y.shape:raise ValueError('Outcome shape mismatch')
    if part=='E4b':
        n=y.shape[-1]
        if n<2:raise ValueError('At least two systems required')
        yc=y-y.mean(axis=-1,keepdims=True);a=p0-p0.mean(axis=-1,keepdims=True)
        b=pk-pk.mean(axis=-2,keepdims=True)
        # Sum of all pair differences squared = n * centered sum of squares.
        factor=2*n/(n-1)
        observed=factor*np.mean(yc**2);error0=factor*np.mean((yc-a)**2)
        errork=factor*np.mean((yc[...,None]-b)**2,axis=tuple(range(y.ndim)))
    else:
        dy=y[:,0]-y[:,1];a=p0[:,0]-p0[:,1];b=pk[:,0,:]-pk[:,1,:]
        observed=np.mean(dy**2);error0=np.mean((dy-a)**2);errork=np.mean((dy[:,None]-b)**2,axis=0)
    out={CORE_STATS[0]:np.full(pk.shape[-1],observed),CORE_STATS[1]:np.full(pk.shape[-1],error0),CORE_STATS[2]:errork}
    if part=='E4b':
        best=y.max(axis=-1);mean=y.mean(axis=-1)
        selected0=np.take_along_axis(y,np.argmax(p0,axis=-1)[...,None],axis=-1)[...,0]
        selectedk=np.take_along_axis(np.broadcast_to(y[...,None],pk.shape),np.argmax(pk,axis=-2)[...,None,:],axis=-2)[...,0,:]
        out.update({SELECTION_STATS[0]:np.full(pk.shape[-1],np.mean(best-selected0)),
                    SELECTION_STATS[1]:np.mean(best[...,None]-selectedk,axis=(0,1)),
                    SELECTION_STATS[2]:np.full(pk.shape[-1],np.mean(selected0-mean)),
                    SELECTION_STATS[3]:np.mean(selectedk-mean[...,None],axis=(0,1))})
    return out


def metric_values(stats):
    den,a,b=[np.asarray(stats[k]) for k in CORE_STATS]
    def divide(x,y):return np.divide(x,y,out=np.full(np.broadcast_shapes(x.shape,y.shape),np.nan),where=y>0)
    return {'observed_contrast_RMS':np.sqrt(den),'P0_residual_RMS':np.sqrt(a),'PK_residual_RMS':np.sqrt(b),
            'P0_unexplained_fraction_U':divide(a,den),'PK_unexplained_fraction_U':divide(b,den),
            'P0_R2_delta':1-divide(a,den),'PK_R2_delta':1-divide(b,den),
            'fraction_P0_residual_squared_explained_by_PK':1-divide(b,a),
            'PK_minus_P0_R2':divide(a-b,den),'MSE_improvement_P0_minus_PK':a-b,
            'RMSE_improvement_P0_minus_PK':np.sqrt(a)-np.sqrt(b),
            **{k:v for k,v in stats.items() if k in SELECTION_STATS}}


def interval_description(samples):
    samples=np.asarray(samples);bad=int((~np.isfinite(samples)).sum())
    return {'interval95':None if bad else np.quantile(samples,[.025,.975]).tolist(),
            'undefined_draws':bad,'undefined_draws_not_dropped':True}


def summarize(point_stats,conditional,propagated=None):
    point=metric_values(point_stats);conditional=metric_values(conditional)
    propagated=metric_values(propagated) if propagated is not None else None
    return {name:{'estimate':float(value) if np.isfinite(value) else None,
                  'conditional_on_E1_point_kernel':interval_description(conditional[name]),
                  **({'E1_kernel_and_E4_topic_propagated':interval_description(propagated[name])} if propagated is not None else {})}
            for name,value in point.items()}


def topic_aggregate(per_topic,weights):
    return {key:weights@np.array([topic[key] for topic in per_topic])/weights.shape[1] for key in per_topic[0]}


def dynamic_conditional(results,weights):
    if not any(r.get('MC_completed') for r in results):return {'status':'not_run'}
    if not all(r.get('MC_completed') for r in results):return {'status':'incomplete_topic_panel','no_completed_topic_subset_claimed_as_full':True}
    has_b=all('E4b_MC' in r for r in results);has_a=all('E4a_MC' in r for r in results)
    keys=list(results[0]['E4b_MC']) if has_b else [];summaries={};mcse={}
    for result in results:
        if has_b and list(result['E4b_MC'])!=keys:raise ValueError('Different dynamic configurations across topics')
    for key in keys:
        first=results[0]['E4b_MC'][key];methods=first['methods'];nrounds=first['rounds']
        if methods!=['measured','P0','PK']:raise ValueError('Unexpected dynamic method order')
        for r in results:
            mc=r['E4b_MC'][key]
            if mc['methods']!=methods or mc['rounds']!=nrounds:raise ValueError('Incompatible dynamic results')
        summaries[key]={}
        for round_id in range(nrounds):
            topic_stats=[]
            for r in results:
                mc=r['E4b_MC'][key];means=np.array(mc['outcomes']['means'])[:,:,round_id]
                y,a,b=means.reshape(3,r['E4b']['n_mappings'],r['E4b']['n_systems'])
                stat=statistics('E4b',y[None,:,:],a[None,:,:],b[None,:,:,None])
                topic_stats.append({k:float(v[0]) for k,v in stat.items()})
            point={k:np.mean([s[k] for s in topic_stats]) for k in topic_stats[0]}
            summaries[key][str(round_id+1)]=summarize(point,topic_aggregate(topic_stats,weights))
        mcse[key]=[]
        for r in results:
            mc=r['E4b_MC'][key];se=np.array(mc['outcomes']['Monte_Carlo_standard_errors'])
            mcse[key].append({'topic_id':r['topic_id'],'draws':mc['draws'],
                'draws_scope':mc.get('draws_scope','trajectories per mapping'),
                'total_draws_per_design':mc.get('total_draws_per_design'),
                'outcome_MCSE_median':float(np.median(se)),'outcome_MCSE_maximum':float(se.max()),
                'full_MCSE_in_source_topic_analysis':True})
    pasch={};pasch_mcse=[]
    if has_a:
        rounds=results[0]['E4a_MC']['records'][0]['simulation']['rounds']
        for r in results:
            records=r['E4a_MC']['records']
            if r['E4a_MC']['outcome_denominator']!=6 or len(records)!=70:raise ValueError('Incomplete or invalid G6 dynamics')
            for row in records:
                sim=row['simulation']
                if sim['n_agents']!=6 or sim['rounds']!=rounds or sim['methods']!=['measured','P0','PK']:
                    raise ValueError('Incompatible G6 simulation')
            ses=np.array([row['simulation']['outcomes']['Monte_Carlo_standard_errors'] for row in records])
            pasch_mcse.append({'topic_id':r['topic_id'],'draws_per_design_per_condition':r['E4a_MC']['draws_per_design_per_condition'],
                              'outcome_MCSE_median':float(np.median(ses)),'outcome_MCSE_maximum':float(ses.max())})
        for ri in range(rounds):
            topics=[]
            for r in results:
                values=np.array([row['simulation']['outcomes']['means'] for row in r['E4a_MC']['records']])[:,:,:,ri]
                stats=statistics('E4a',values[:,0,:],values[:,1,:],values[:,2,:,None])
                topics.append({k:float(v[0]) for k,v in stats.items()})
            point={k:np.mean([s[k] for s in topics]) for k in topics[0]}
            pasch[str(ri+1)]=summarize(point,topic_aggregate(topics,weights))
    return {'status':'complete_conditional_topic_aggregation' if has_a or has_b else 'incomplete_stage_panel','by_configuration_and_round':summaries,'MCSE':mcse,
        'E4b_complete_topic_panel':has_b,'E4a_complete_topic_panel':has_a,
        'E4a_by_round':pasch,'E4a_MCSE':pasch_mcse,
        'E1_kernel_parameter_uncertainty_propagated':False,
        'interval_scope':'E4 topic bootstrap of saved MC mean outcomes; fitted E1 kernels fixed',
        'MC_error_scope':'Outcome MCSE is retained separately. Nonlinear metric intervals do not integrate simulation uncertainty or unrecorded within-mapping contrast covariance.',
        'within_mapping_contrasts':True,'nested_kernel_resimulation_performed':False}


def aggregate(topic_paths,kernel_path,manifest_path,sts_path,draws=10000,seed=20261031,batch_size=64,dynamics_paths=None):
    if draws<2 or batch_size<1:raise ValueError('Require at least two draws and a positive batch size')
    results=[read_json(p) for p in topic_paths];order=np.argsort([r['topic_id'] for r in results])
    results=[results[i] for i in order];topic_paths=[Path(topic_paths[i]) for i in order]
    ids=[r['topic_id'] for r in results];names=[r['topic'] for r in results]
    if len(ids)<2 or len(set(ids))!=len(ids) or len(set(names))!=len(names):raise ValueError('Distinct E4 topics required')
    kernel=read_json(kernel_path);manifest=read_json(manifest_path);systems=read_json(sts_path)['systems']
    if kernel.get('readout')!='R0':raise ValueError('E4 requires R0 kernel analysis')
    if set(names)&set(kernel['topics']):raise ValueError('E1/E4 topics are not disjoint')
    meta={t['topic_id']:t for t in manifest['topics']}
    source_hashes={str(Path(p).resolve()):sha(p) for p in [*topic_paths,kernel_path,manifest_path,sts_path,__file__,
                   Path(__file__).with_name('e4_simulation.py'),Path(__file__).with_name('analyze_v3.py')]}
    if dynamics_paths:
        seen=set()
        for path in dynamics_paths:
            dynamic=read_json(path);tid=dynamic['topic_id']
            if tid in seen or tid not in ids:raise ValueError('Duplicate or unmatched dynamics topic')
            seen.add(tid);index=ids.index(tid)
            if dynamic['source_E4_analysis_sha256']!=sha(topic_paths[index]):raise ValueError('Dynamics use a different point analysis')
            results[index].update({k:dynamic[k] for k in ('MC_completed','E4a_MC','E4b_MC') if k in dynamic})
            source_hashes[str(Path(path).resolve())]=sha(path)
    point_k,kdraws,kw,e1topics=kernel_bootstrap(kernel,draws,seed)
    tw=np.random.default_rng(seed+1).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=draws)
    point_topics={part:[] for part in ('E4a','E4b','E4c')};propagated={};all_identity=[]
    for ti,(path,r) in enumerate(zip(topic_paths,results)):
        if r['E1_execution_identity']!=kernel['execution_identity']:raise ValueError('E4 was evaluated with a different E1 identity')
        if r['execution_identity']['critical_identity']!=kernel['execution_identity']['critical_identity']:
            raise ValueError('Model identity differs between E1 and E4')
        for identity in r['execution_identity']['runs']:
            all_identity.append((identity['engine'],identity['batch_size'],identity['calibration_sha256']))
        for needed in (kernel_path,manifest_path,sts_path):
            if sha(needed) not in r['input_hashes'].values():raise ValueError('Per-topic source provenance differs: '+str(needed))
        cache_path=path.parent/r['cache_artifact']['filename']
        if sha(cache_path)!=r['cache_artifact']['sha256']:raise ValueError('Cache artifact changed')
        source_hashes[str(cache_path.resolve())]=sha(cache_path)
        compiler,sections,expected=compile_topic(load_cache(cache_path),r,meta[r['topic_id']],systems)
        check=compiler.evaluate(np.stack([np.zeros(3),point_k]))
        for part,(sl,shape) in sections.items():
            computed=check[sl].reshape(*shape,2)
            if not np.allclose(computed[...,0],expected[part]['P0'],rtol=1e-10,atol=1e-11):raise ValueError('Compiled P0 differs: '+part)
            if not np.allclose(computed[...,1],expected[part]['PK'],rtol=1e-10,atol=1e-11):raise ValueError('Compiled PK differs: '+part)
            stats=statistics(part,expected[part]['measured'],expected[part]['P0'],computed[...,1,None])
            point_topics[part].append({key:float(value[0]) for key,value in stats.items()})
            if part not in propagated:propagated[part]={key:np.zeros(draws) for key in stats}
        for start in range(0,draws,batch_size):
            stop=min(draws,start+batch_size);valid=np.isfinite(kdraws[start:stop]).all(axis=1)
            values=np.full((len(compiler.design_meetings),stop-start),np.nan)
            if valid.any():values[:,valid]=compiler.evaluate(kdraws[start:stop][valid])
            for part,(sl,shape) in sections.items():
                stats=statistics(part,expected[part]['measured'],expected[part]['P0'],values[sl].reshape(*shape,stop-start))
                # Selection argmax cannot encode an undefined kernel draw as system 0.
                for key in SELECTION_STATS:
                    if key.startswith('PK_') and key in stats:stats[key][~valid]=np.nan
                for key,value in stats.items():propagated[part][key][start:stop]+=tw[start:stop,ti]*value/len(ids)
        print(json.dumps({'E4_bootstrap_topic_complete':r['topic_id'],'draws':draws,
                          'compiled_cells':len(compiler.bases),'compiled_meetings':len(compiler.meeting_cells)}),flush=True)
    if any(identity!=all_identity[0] for identity in all_identity):raise ValueError('E4 topics use different execution policies')
    sections_out={};draw_arrays={'E1_kernel_draws':kdraws,'E1_topic_weights':kw,'E4_topic_weights':tw}
    for part,topics in point_topics.items():
        point={key:np.mean([s[key] for s in topics]) for key in topics[0]};conditional=topic_aggregate(topics,tw)
        leave_out={}
        for i,tid in enumerate(ids):
            stats={key:np.mean([s[key] for j,s in enumerate(topics) if j!=i]) for key in topics[0]}
            leave_out[tid]={key:float(v) if np.isfinite(v) else None for key,v in metric_values(stats).items()}
        sections_out[part]={'metrics':summarize(point,conditional,propagated[part]),
            'topic_sufficient_statistics':dict(zip(ids,topics)),'leave_one_topic_out':leave_out,
            'PK_scope':'pair-truncated' if part=='E4c' else 'transferred pair relationship kernels'}
        for scope,stats in [('conditional',conditional),('propagated',propagated[part])]:
            for key,value in metric_values(stats).items():draw_arrays[f'{part}_{scope}_{key}']=value
    result={'topic_ids':ids,'topics':names,'n_topics':len(ids),'planned_six_topics_complete':len(ids)==6,
        'model_identity':kernel['execution_identity']['critical_identity'],'E1_topics':e1topics,
        'E1_kernel_point':dict(zip(RELATIONS,map(float,point_k))),
        'bootstrap':{'draws':draws,'seed_E1':seed,'seed_E4':seed+1,'unit':'topic',
            'E1_and_E4_topic_resampling_independent':True,'joint_three_kernel_draw_shared_across_all_E4_topics':True,
            'E1_invalid_kernel_draws':int((~np.isfinite(kdraws).all(axis=1)).sum()),
            'within_topic_all_items_mappings_conditions_and_label_pairs_kept_together':True,
            'topic_aggregation':'equal topic weights; ratios recomputed from mean squared contrast/error components',
            'intervals':'pointwise percentile95; not simultaneous intervals and not six-topic asymptotic guarantees'},
        'small_topic_panel_caveat':'Six E4 topics give limited independent information. Bootstrap intervals are approximate; report individual topics and leave-one-topic-out sensitivity. Mappings, states, design pairs and tau reversals do not increase the topic sample size.',
        'single_round_E1_kernel_uncertainty_propagated':True,'single_round':sections_out,
        'dynamic':dynamic_conditional(results,tw),
        'STS_comparison_scope':'80 systems contrasted within each shared mapping and initial state before topic aggregation; never contrast mapping-averaged system means',
        'cache_uncertainty_scope':'Finite prompts/agents and deterministic cache readouts are fixed within topics; bootstrap does not create model-sampling or token-level stochastic uncertainty.',
        'model_selection_scope':'Conditional on this selected model and E4 panel; not an unbiased model-population claim.',
        'file_sha256':source_hashes}
    return result,draw_arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--topics',action='append',required=True)
    p.add_argument('--dynamics',action='append',help='Optional dedicated dynamics JSON, one for each matching topic')
    p.add_argument('--kernel-analysis',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache-manifest',default=str(ROOT/'data/e4_cache_manifest.json'))
    p.add_argument('--sts',default=str(ROOT/'data/e4_graphs/sts15_all80.json'))
    p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
    output=Path(a.output);samples=output.with_name(output.stem+'_bootstrap.npz')
    if output.exists() or samples.exists():raise FileExistsError('Refusing to overwrite analysis or bootstrap artifacts')
    result,arrays=aggregate(a.topics,a.kernel_analysis,a.cache_manifest,a.sts,a.draws,a.seed,a.batch_size,a.dynamics)
    with samples.open('xb') as f:np.savez_compressed(f,**arrays)
    result['bootstrap_artifact']={'filename':samples.name,'sha256':sha(samples)}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'n_topics':result['n_topics'],'single_round_E1_kernel_uncertainty_propagated':True,
                      'dynamic_status':result['dynamic']['status'],'output':str(output)}))


if __name__=='__main__':main()
