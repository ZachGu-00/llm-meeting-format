"""Gate1-bound analysis of sealed G1/G2; optional matched G3 R1 sensitivity."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
from diagnostics_d1_d2 import split_caches,moment,centered,d1_statistics
from diagnostics_d3_d4 import features

SEED=20261101
EQUIVALENCE_MARGIN=.10


def holm(values):
    order=np.argsort(values);result=np.empty(len(values));last=0.
    for rank,index in enumerate(order):
        last=max(last,(len(values)-rank)*float(values[index]));result[index]=min(1.,last)
    return result.tolist()


def upper_p(point,bootstrap,null=0.):
    noise=np.asarray(bootstrap)-point
    return float((1+np.count_nonzero(noise>=point-null))/(len(noise)+1))


def lower_p(point,bootstrap,null):
    noise=np.asarray(bootstrap)-point
    return float((1+np.count_nonzero(noise<=point-null))/(len(noise)+1))


def slope(outcome,x):
    x=centered(x);y=centered(outcome);den=float((x*x).sum())
    return float((x*y).sum()/den) if den else 0.


def sign(value):return int(value>1e-12)-int(value< -1e-12)


def analyze_job(job,plan,root,predictors,pairs,topics,freeze,original=False):
    from common import load,sha,verify_bindings,write_new
    from analyze_v3 import read_closed
    from e4_simulation import predicted_caches,single_round_batch
    from run_e4_analysis import initial_state
    base=Path(plan['base']);readout=job['readout'];stage=job['stage'];repo=job['model'];tid=job['topic_id']
    target=root/'unsealed'/stage/repo.replace('/','--')/(tid+'_'+readout+'.json')
    if target.exists():
        result=load(target);verify_bindings(result['source_sha256']);return result
    seal=Path(job['output']+'.sealed.json')
    if not original:
        verified=load(seal);verify_bindings(verified['source_sha256'])
        if verified['status']!='sealed_quality_audited' or verified['rows']!=49020:raise ValueError('Sealed native evidence incomplete')
    rows=read_closed(job['input'],job['output'])
    if readout=='R1':rows=({**r,'R0_y_AB':r['R1_y_AB']} for r in rows)
    halves,count=split_caches(rows)
    if count!=49020:raise ValueError('Unexpected full topic count')
    full={k:(halves[0][k]+halves[1][k])/2 for k in ['alpha','pair','triple']};full.update(n=15,five={})
    model=predictors['models'][repo];kernel=model['readouts'][readout]['by_topic'][tid]
    if topics[tid]['topic'] in kernel['training_topics']:raise ValueError('E1 target-topic leakage')
    if stage=='G2' and tid in model.get('training_E4_topics',[]):raise ValueError('Content predictor target leakage')
    sts=load(base/'data/e4_graphs/sts15_all80.json')['systems']
    seed=int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+tid).encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(seed);mappings=[rng.permutation(15) for _ in range(20)]
    designs=[sorted(tuple(sorted(mapping[v] for v in b)) for b in system['blocks']) for mapping in mappings for system in sts]
    blocks=np.asarray(designs);n3=[]
    for m,tau in itertools.product((3,5,7),(-1,1)):
        state=initial_state(15,m,tau);n3.extend(((state[blocks]==int(tau==1)).all(axis=2)).sum(axis=1).reshape(20,80))
    n3=np.asarray(n3);outputs=[]
    for cache in [full,*halves]:
        pred=predicted_caches(cache,kernel['relation_kernels'])
        if 'content_coefficients' in model:
            f=features(cache,topics[tid],pairs[tid])
            for name,key,feature_key in [('content_PK','content_coefficients','content'),('matched_PK','matched_relation_only_coefficients','kernel')]:
                logits=np.full_like(cache['triple'],np.nan);logits[f['indices']]=f['p0']+f[feature_key]@np.asarray(model[key]);pred[name]=logits
        out={k:[] for k in pred}
        for m,tau in itertools.product((3,5,7),(-1,1)):
            state=initial_state(15,m,tau)
            for name,logits in pred.items():out[name].extend(single_round_batch(logits,designs,state,tau).reshape(20,80))
        outputs.append({k:np.asarray(v) for k,v in out.items()})
    y=outputs[0]['measured'];p0=outputs[0]['P0'];pk=outputs[0]['PK']
    d1,_=d1_statistics(y,p0,[x['measured'] for x in outputs[1:]],[x['P0'] for x in outputs[1:]],np.random.default_rng(SEED),permutations=0)
    by_condition=[]
    for index,(m,tau) in enumerate(itertools.product((3,5,7),(-1,1))):
        ix=slice(index*20,(index+1)*20);observed=slope((y-p0)[ix],n3[ix]);predicted=slope((pk-p0)[ix],n3[ix])
        by_condition.append({'k':m,'tau':tau,'observed_U_slope':observed,'predicted_correction_slope':predicted,
                             'E1_sign_aligned_U_slope':observed*sign(predicted)})
    metrics={}
    for name,values in outputs[0].items():
        if name=='measured':continue
        error=y-values;metrics[name]={'contrast_MSE':moment(error,error)}
    sources={str(root/'gate1/freeze.json'):sha(root/'gate1/freeze.json'),str(root/'gate1/frozen_predictors.json'):sha(root/'gate1/frozen_predictors.json')}
    sources[ str(seal) if not original else job['output'] ]=sha(seal if not original else job['output'])
    result={'schema':'voices_v31_postfreeze_topic_analysis_v1','model':repo,'stage':stage,'topic_id':tid,'readout':readout,
            'D1':d1,'H4_mean_aligned_slope':float(np.mean([r['E1_sign_aligned_U_slope'] for r in by_condition])),
            'n3_conditions':by_condition,'observed_U_n3_slope':slope(y-p0,n3),'E1_correction_n3_slope':slope(pk-p0,n3),
            'metrics':metrics,'source_sha256':sources,'freeze_analysis_code_sha256':freeze['analysis_code_sha256']}
    write_new(target,result);return result


def summarize(g1,g2,predictors,draws=10000):
    repos=list(dict.fromkeys(r['model'] for r in g2));tids=sorted({r['topic_id'] for r in g2})
    if len(repos)!=6 or len(tids)!=14 or len(g2)!=84:raise ValueError('Incomplete G2 confirmatory panel')
    lookup={(r['model'],r['topic_id']):r for r in g2}
    if len(lookup)!=84:raise ValueError('Duplicate confirmatory topic record')
    matrices={k:np.empty((6,14)) for k in ['reliable_U','Y_variance','H4','H5_E1','H5_matched']}
    for i,repo in enumerate(repos):
        norm=predictors['models'][repo]['fixed_loss_normalizer']
        if not norm>0:raise ValueError('Invalid pre-frozen loss normalization')
        for j,tid in enumerate(tids):
            r=lookup[repo,tid];matrices['reliable_U'][i,j]=r['D1']['cross_half_U_covariance']
            matrices['Y_variance'][i,j]=r['D1']['observed_contrast_variance'];matrices['H4'][i,j]=r['H4_mean_aligned_slope']
            matrices['H5_E1'][i,j]=(r['metrics']['PK']['contrast_MSE']-r['metrics']['content_PK']['contrast_MSE'])/norm
            matrices['H5_matched'][i,j]=(r['metrics']['matched_PK']['contrast_MSE']-r['metrics']['content_PK']['contrast_MSE'])/norm
    indices=np.random.default_rng(SEED).integers(0,14,size=(draws,14))
    point={k:float(v.mean()) for k,v in matrices.items()}
    boot={k:v[:,indices].mean(axis=(0,2)) for k,v in matrices.items()}
    fraction=point['reliable_U']/point['Y_variance'];fraction_boot=boot['reliable_U']/boot['Y_variance']
    tests=[{'name':'H2a_positive_U','estimate':point['reliable_U'],'CI95':np.quantile(boot['reliable_U'],[.025,.975]).tolist(),
            'p':upper_p(point['reliable_U'],boot['reliable_U'])},
           {'name':'H2a_below_10pct','estimate':fraction,'margin':EQUIVALENCE_MARGIN,'CI95':np.quantile(fraction_boot,[.025,.975]).tolist(),
            'p':lower_p(fraction,fraction_boot,EQUIVALENCE_MARGIN)},
           {'name':'H4_E1_aligned_n3','estimate':point['H4'],'CI95':np.quantile(boot['H4'],[.025,.975]).tolist(),
            'p':upper_p(point['H4'],boot['H4'])},
           {'name':'H5_content_gain','estimates':{k:point[k] for k in ['H5_E1','H5_matched']},
            'CI95':{k:np.quantile(boot[k],[.025,.975]).tolist() for k in ['H5_E1','H5_matched']},
            'p':max(upper_p(point[k],boot[k]) for k in ['H5_E1','H5_matched']),
            'intersection_test':'Content must beat both E1 PK and relation-only predictor matched on old-six-topic training'}]
    direction_rows=[]
    for collection in (g2,g1):
        for repo in dict.fromkeys(r['model'] for r in collection):
            rows=[r for r in collection if r['model']==repo]
            measured=float(np.mean([r['observed_U_n3_slope'] for r in rows]));predicted=float(np.mean([r['E1_correction_n3_slope'] for r in rows]))
            direction_rows.append({'model':repo,'stage':rows[0]['stage'],'topics':len(rows),'observed':measured,'predicted':predicted,
                                   'same_nonzero_sign':sign(measured)==sign(predicted) and sign(predicted)!=0})
    if len(direction_rows)!=11 or len({r['model'] for r in direction_rows})!=11:raise ValueError('H3 prime needs all eleven sealed model endpoints')
    successes=sum(r['same_nonzero_sign'] for r in direction_rows)
    p=sum(math.comb(11,k) for k in range(successes,12))/2**11
    tests.append({'name':'H3prime_direction','agreements':successes,'models':11,'p':p,'details':direction_rows,
                  'scope':'Original six on G2 new14 topics; five additions on G1 original6 topics; all measured endpoints newly sealed',
                  'limitation':'Different topic panels and related purposefully chosen models; binomial is a conditional reference, not random-model population inference'})
    for test,adjusted in zip(tests,holm([t['p'] for t in tests])):test['Holm_p']=adjusted;test['passes_R0_family_alpha05']=adjusted<.05
    return {'primary_family':tests,'n_topics':14,'models':6,'bootstrap_draws':draws,'bootstrap_unit':'shared topic clusters, same sampled indices across all six fixed models',
            'R1_requirement':'R0 findings remain provisional until G3 matched original-topic direction checks; G3 covers only original six, not all eleven H3-prime models',
            'topic_metrics':{k:v.tolist() for k,v in matrices.items()},'new_H2a_does_not_replace_original_H2':True}


def run(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path[:0]=[str(ext),str(base/'code')]
    from common import load,sha,verify_bindings,write_new
    freeze=load(root/'gate1/freeze.json');verify_bindings(freeze['source_sha256'])
    if freeze['analysis_code_sha256']!=sha(__file__) or freeze['status']!='frozen_before_any_sealed_effect_analysis':raise ValueError('Gate1 analysis freeze does not match')
    for stage in ['G1','G2']:
        done=load(root/(stage+'_complete.json'));verify_bindings(done['source_sha256'])
        if done['status']!=stage+'_complete_sealed':raise ValueError('Required collection stage not closed')
    predictors=load(root/'gate1/frozen_predictors.json');verify_bindings(predictors['source_sha256'])
    topics={t['topic_id']:t for t in load(base/'data/e4_agents.json')+load(root/'data/e4_agents_new.json')}
    pairs=load(root/'data/M2/pair_features.json');g1plan=load(root/'G1_plan.json');later=load(root/'G2_G3_plan.json')
    records={}
    for stage,plan in [('G1',g1plan),('G2',later)]:
        verify_bindings(plan['source_sha256']);records[stage]=[analyze_job(j,plan,root,predictors,pairs,topics,freeze) for j in plan['jobs'] if j['stage']==stage]
    result=summarize(records['G1'],records['G2'],predictors)
    result.update(schema='voices_v31_gate2_R0_results_v1',status='R0_complete_R1_qualification_pending',
                  source_sha256={str(root/'gate1/freeze.json'):sha(root/'gate1/freeze.json')})
    for stage,rows in records.items():
        for row in rows:
            p=root/'unsealed'/stage/row['model'].replace('/','--')/(row['topic_id']+'_R0.json');result['source_sha256'][str(p)]=sha(p)
    write_new(root/'gate2/R0_report.json',result)
    print(json.dumps({'status':result['status'],'primary_family':result['primary_family']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();run(a.base,a.extension)
