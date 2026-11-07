"""Original E4b prediction ladder and outcome-blind triple-probe budgets."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from diagnostics_d1_d2 import split_caches,centered,moment

SEED=20261101


def features(cache,topic,pair_features):
    j,k,l,bits=np.nonzero(np.isfinite(cache['triple']));sj=bits//4;sk=(bits//2)%2;sl=bits%2
    alpha=cache['alpha'][j,sj];vk=cache['pair'][j,k,2*sj+sk]-alpha;vl=cache['pair'][j,l,2*sj+sl]-alpha
    p0=alpha+vk+vl;residual=cache['triple'][j,k,l,bits]-p0
    relation=np.where((sk!=sj)&(sl!=sj),0,np.where((sk==sj)&(sl==sj),1,2))
    scale=(1-2*sj)*(np.abs(vk)+np.abs(vl))/2
    kernel=np.eye(3)[relation]*scale[:,None]
    lookup={a:i for i,a in enumerate(pair_features['argument_ids'])}
    arg=np.array([[lookup[agent['arguments'][str(s)]['argument_id']] for s in (-1,1)] for agent in topic['agents']])
    a,b,c=arg[j,sj],arg[k,sk],arg[l,sl]
    cosine=np.asarray(pair_features['cosine_similarity']);same=np.asarray(pair_features['same_keypoint'],dtype=float)
    pp_cos=cosine[b,c];pp_same=same[b,c];f_cos=(cosine[a,b]+cosine[a,c])/2;f_same=(same[a,b]+same[a,c])/2
    content=np.concatenate([kernel,kernel*pp_cos[:,None],kernel*pp_same[:,None],kernel*f_cos[:,None],kernel*f_same[:,None]],axis=1)
    flexible=np.stack([alpha,vk+vl,np.minimum(vk,vl),np.maximum(vk,vl),np.minimum(np.abs(vk),np.abs(vl)),
                      np.maximum(np.abs(vk),np.abs(vl)),sj,(sk!=sj).astype(int)+(sl!=sj).astype(int),
                      pp_cos,pp_same,f_cos,np.abs(cosine[a,b]-cosine[a,c]),f_same],axis=1)
    return {'indices':(j,k,l,bits),'p0':p0,'residual':residual,'kernel':kernel,'content':content,'flexible':flexible,
            'block_ids':[tuple(sorted((int(a),int(b),int(c)))) for a,b,c in zip(j,k,l)]}


def fit_ridge(x,y,alpha=1.):
    scale=np.maximum(x.std(axis=0),1e-8);z=x/scale
    return np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@y)/scale


def probe_mask(blocks,topic,fraction):
    unique=sorted(set(blocks),key=lambda b:hashlib.sha256(('voices-v31-D4-probes-v1|'+topic+'|'+','.join(map(str,b))).encode()).hexdigest())
    count=int(np.ceil(len(unique)*fraction));chosen=set(unique[:count])
    return np.array([b in chosen for b in blocks]),count


def run(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';out=root/'diagnostics/D3_D4'
    sys.path[:0]=[str(root/'python_packages'),str(ext),str(base/'code')]
    from sklearn.ensemble import HistGradientBoostingRegressor
    import sklearn
    from common import load,sha,verify_bindings,write_new,atomic_status
    from analyze_v3 import read_closed
    from e4_simulation import load_cache,single_round_batch,predicted_caches
    from run_e4_analysis import initial_state
    old_path=base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json';old=load(old_path)
    topics_path=base/'data/e4_agents.json';topics={t['topic_id']:t for t in load(topics_path)}
    pair_path=root/'data/M2/pair_features.json';pair=load(pair_path);sts_path=base/'data/e4_graphs/sts15_all80.json';sts=load(sts_path)['systems']
    d1_protocol=root/'diagnostics/D1_D2/protocol.json';verify_bindings(load(d1_protocol)['source_sha256'])
    sources={str(p):sha(p) for p in [old_path,topics_path,pair_path,root/'data/M2/manifest.json',d1_protocol,
        Path(__file__),root/'diagnostics_d1_d2.py',root/'test_diagnostics_d3.py',root/'research_plan_v31_user.md']}
    protocol={'schema':'voices_v31_D3_D4_development_protocol_v1','scope':'Original36 E4b caches only; no G1/G2 data',
       'source_sha256':sources,'sklearn_version':sklearn.__version__,'seed':SEED,'bootstrap_draws':10000,
       'methods':{'P0':'frozen low-order cache construction','E1_PK':'original frozen E1 kernels',
         'n3_diagnostic':'fit U against centered n3 within k/tau and shared mappings on the target topic; upper bound only',
         'within_topic_PK':'three relation coefficients, ridge alpha1 on all target-topic individual triple residuals; upper bound only',
         'LOTO_PK':'same three coefficients fit to other five original topics',
         'LOTO_content_PK':'15 symmetric relation-by-content features; ridge alpha1, scaling fitted on other five topics',
         'LOTO_flexible':'13 symmetric individual features; HistGradientBoosting squared_error, 120 trees,15 leaves, lr.08,l2=1, no early stopping, other five topics',
         'probe10_PK':'fixed-hash 46 of455 unordered triples, all states/focals/labels retained; 4416 triple prompts plus3420 low-order prompts',
         'probe20_PK':'fixed-hash 91 of455 unordered triples, all states/focals/labels retained; 8736 triple prompts plus3420 low-order prompts'},
       'reliable_share':'1 - covariance(halfA observed - halfA prediction, halfB observed - halfB prediction) / original D1 U covariance; no clipping',
       'limitations':'Readout halves are deterministic perturbations. Within-topic fits and n3 are optimistic diagnostic bounds. Probe design predictions include calibrated triples and are not wholly unseen-design validation; held-out individual triple errors are separately reported.',
       'no_hyperparameter_tuning_on_target_topics':True}
    if sklearn.__version__!='1.7.2':raise ValueError('Pinned CPU model package differs')
    write_new(out/'protocol.json',protocol);all_records=[]
    for repo in old['selected_repos']:
        jobs=[j for j in old['jobs'] if j['model']==repo];cases=[]
        for job in jobs:
            tid=job['topic_id'];cache_path=Path(job['deferred_analysis']['output_dir'])/(tid+'_caches.npz')
            cache=load_cache(cache_path);f=features(cache,topics[tid],pair[tid])
            cases.append({'job':job,'cache':cache,'features':f})
        for ci,case in enumerate(cases):
            job=case['job'];tid=job['topic_id'];target=out/repo.replace('/','--')/(tid+'.json')
            if target.exists():
                record=load(target);verify_bindings(record['source_sha256']);all_records.append(record);continue
            atomic_status(out/'status.json',{'status':'running_D3_D4_original_E4b','model':repo,'topic':tid,'GPU_used':False})
            training=[c for i,c in enumerate(cases) if i!=ci]
            if len(training)!=5 or any(c['job']['topic_id']==tid for c in training):raise ValueError('Target topic leaked into LOTO training')
            f=case['features'];train_y=np.concatenate([c['features']['residual'] for c in training])
            coef={'within_topic_PK':fit_ridge(f['kernel'],f['residual'])}
            coef['LOTO_PK']=fit_ridge(np.concatenate([c['features']['kernel'] for c in training]),train_y)
            coef['LOTO_content_PK']=fit_ridge(np.concatenate([c['features']['content'] for c in training]),train_y)
            gbm=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,
                                              early_stopping=False,random_state=SEED)
            gbm.fit(np.concatenate([c['features']['flexible'] for c in training]),train_y)
            probes={}
            for frac,name in [(.1,'probe10_PK'),(.2,'probe20_PK')]:
                mask,count=probe_mask(f['block_ids'],tid,frac);coef[name]=fit_ridge(f['kernel'][mask],f['residual'][mask])
                prediction=f['kernel'][~mask]@coef[name]
                probes[name]={'selected_unordered_triples':count,'total_unordered_triples':455,'triple_prompt_cost':count*96,
                              'baseline_pair_prompt_cost':3420,'heldout_individual_residual_MSE':float(np.mean((f['residual'][~mask]-prediction)**2)),
                              'training_and_heldout_triple_sets_disjoint':True}
            halves,_=split_caches(read_closed(job['input'],job['output']))
            datasets=[(case['cache'],f),*((c,features(c,topics[tid],pair[tid])) for c in halves)]
            seed=int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+tid).encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);mappings=[rng.permutation(15) for _ in range(20)]
            designs=[sorted(tuple(sorted(mapping[v] for v in b)) for b in s['blocks']) for mapping in mappings for s in sts]
            blocks=np.asarray(designs);n3=[]
            for m,tau in itertools.product((3,5,7),(-1,1)):
                state=initial_state(15,m,tau);n3.extend(((state[blocks]==int(tau==1)).all(axis=2)).sum(axis=1).reshape(20,80))
            n3=np.asarray(n3)
            kernel_path=Path(job['deferred_analysis']['command'][job['deferred_analysis']['command'].index('--kernel-analysis')+1])
            kernels=load(kernel_path)['point']['relation_kernels'];outcomes=[]
            for cache,feat in datasets:
                predictions=predicted_caches(cache,kernels);predictions['E1_PK']=predictions.pop('PK')
                for name,c in coef.items():
                    features_key='content' if name=='LOTO_content_PK' else 'kernel'
                    a=np.full_like(cache['triple'],np.nan);a[feat['indices']]=feat['p0']+feat[features_key]@c;predictions[name]=a
                a=np.full_like(cache['triple'],np.nan);a[feat['indices']]=feat['p0']+gbm.predict(feat['flexible']);predictions['LOTO_flexible']=a
                result={name:[] for name in predictions}
                for m,tau in itertools.product((3,5,7),(-1,1)):
                    state=initial_state(15,m,tau)
                    for name,logits in predictions.items():result[name].extend(single_round_batch(logits,designs,state,tau).reshape(20,80))
                outcomes.append({k:np.asarray(v) for k,v in result.items()})
            for item in outcomes:item['n3_diagnostic']=item['P0'].copy()
            for ci2 in range(6):
                sl=slice(ci2*20,(ci2+1)*20);x=centered(n3[sl]);u=centered(outcomes[0]['measured'][sl]-outcomes[0]['P0'][sl])
                slope=float((x*u).sum()/(x*x).sum()) if (x*x).sum() else 0.
                for item in outcomes:item['n3_diagnostic'][sl]+=slope*x
            ua=outcomes[1]['measured']-outcomes[1]['P0'];ub=outcomes[2]['measured']-outcomes[2]['P0'];denom=moment(ua,ub)
            methods={}
            for name in outcomes[0]:
                if name=='measured':continue
                a=outcomes[1]['measured']-outcomes[1][name];b=outcomes[2]['measured']-outcomes[2][name]
                err=outcomes[0]['measured']-outcomes[0][name]
                methods[name]={'contrast_MSE':moment(err,err),'cross_half_residual_covariance':moment(a,b),
                               'reliable_U_share_explained':1-moment(a,b)/denom if denom else None}
            record={'model':repo,'topic_id':tid,'methods':methods,'D1_U_covariance':denom,'D4_probes':probes,
                    'LOTO_training_topics':[c['job']['topic_id'] for c in training],'coefficients':{k:v.tolist() for k,v in coef.items()},
                    'sealed_data_read':False,'source_sha256':{str(out/'protocol.json'):sha(out/'protocol.json'),str(kernel_path):sha(kernel_path)}}
            write_new(target,record);all_records.append(record)
    result={'status':'complete_D3_D4_original36_E4b','models':[],'sealed_data_read':False,
            'source_sha256':{str(out/'protocol.json'):sha(out/'protocol.json')},'confirmatory_claims':False}
    rng=np.random.default_rng(SEED);indices=rng.integers(0,6,(10000,6))
    for repo in old['selected_repos']:
        rows=[r for r in all_records if r['model']==repo];den=np.array([r['D1_U_covariance'] for r in rows]);methods={}
        for name in rows[0]['methods']:
            mse=np.array([r['methods'][name]['contrast_MSE'] for r in rows]);cov=np.array([r['methods'][name]['cross_half_residual_covariance'] for r in rows])
            share=1-cov.mean()/den.mean();bootstrap=1-cov[indices].mean(axis=1)/den[indices].mean(axis=1)
            methods[name]={'contrast_MSE':float(mse.mean()),'reliable_U_share_explained':float(share),
                           'topic95CI':np.quantile(bootstrap,[.025,.975]).tolist(),
                           'ratio_interval_unstable_if_D1_denominator_crosses_zero':True}
        result['models'].append({'model':repo,'methods':methods})
    for job in old['jobs']:
        p=out/job['model'].replace('/','--')/(job['topic_id']+'.json');result['source_sha256'][str(p)]=sha(p)
    write_new(out/'report.json',result);atomic_status(out/'status.json',{'status':result['status'],'GPU_used':False})
    print(json.dumps({'status':result['status'],'models':result['models']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();run(a.base,a.extension)
