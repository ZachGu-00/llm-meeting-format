"""Matched original-topic R1 checks; explicitly limited to six original models."""
import argparse
from pathlib import Path
import sys
import numpy as np
from confirmatory import analyze_job,sign


def run(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path[:0]=[str(ext),str(base/'code')]
    from common import load,sha,verify_bindings,write_new
    freeze=load(root/'gate1/freeze.json');verify_bindings(freeze['source_sha256'])
    if freeze['source_sha256'][str(Path(__file__).resolve())]!=sha(__file__):raise ValueError('R1 sensitivity code is not frozen')
    prior=load(root/'gate2/R0_report.json');verify_bindings(prior['source_sha256'])
    completed=load(root/'G3_complete.json');verify_bindings(completed['source_sha256'])
    if completed['status']!='G3_complete_sealed' or completed['rows']!=588240:raise ValueError('G3 collection not closed')
    later=load(root/'G2_G3_plan.json');old=load(base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json')
    predictors=load(root/'gate1/frozen_predictors.json');pairs=load(root/'data/M2/pair_features.json')
    topics={t['topic_id']:t for t in load(base/'data/e4_agents.json')+load(root/'data/e4_agents_new.json')}
    records=[];sources={str(root/'gate2/R0_report.json'):sha(root/'gate2/R0_report.json'),str(root/'G3_complete.json'):sha(root/'G3_complete.json'),
                        str(root/'gate1/freeze.json'):sha(root/'gate1/freeze.json')}
    for job in later['jobs']:
        if job['stage']!='G3':continue
        r1=analyze_job(job,later,root,predictors,pairs,topics,freeze)
        original=next(j for j in old['jobs'] if j['model']==job['model'] and j['topic_id']==job['topic_id'])
        r0=analyze_job({**original,'stage':'G3_original_R0'},later,root,predictors,pairs,topics,freeze,original=True)
        records.append({'model':job['model'],'topic_id':job['topic_id'],'R0':r0,'R1':r1})
        for stage,readout in [('G3','R1'),('G3_original_R0','R0')]:
            p=root/'unsealed'/stage/job['model'].replace('/','--')/(job['topic_id']+'_'+readout+'.json');sources[str(p)]=sha(p)
    models=[]
    for repo in dict.fromkeys(r['model'] for r in records):
        rows=[r for r in records if r['model']==repo];entry={'model':repo,'topics':2,'readouts':{}}
        for readout in ['R0','R1']:
            rr=[r[readout] for r in rows];cov=np.mean([r['D1']['cross_half_U_covariance'] for r in rr]);var=np.mean([r['D1']['observed_contrast_variance'] for r in rr])
            norm=predictors['models'][repo]['fixed_loss_normalizer']
            gain=lambda comparator:float(np.mean([r['metrics'][comparator]['contrast_MSE']-r['metrics']['content_PK']['contrast_MSE'] for r in rr]))/norm
            entry['readouts'][readout]={'U_covariance':float(cov),'Y_variance':float(var),'U_fraction':float(cov/var),'H4_aligned_slope':float(np.mean([r['H4_mean_aligned_slope'] for r in rr])),
                'H5_gain_vs_E1':gain('PK'),'H5_gain_vs_matched':gain('matched_PK'),
                'observed_U_n3_slope':float(np.mean([r['observed_U_n3_slope'] for r in rr])),
                'predicted_U_n3_slope':float(np.mean([r['E1_correction_n3_slope'] for r in rr]))}
        models.append(entry)
    pooled={}
    for readout in ['R0','R1']:
        pooled[readout]={k:float(np.mean([m['readouts'][readout][k] for m in models])) for k in models[0]['readouts'][readout]}
        pooled[readout]['U_fraction']=pooled[readout]['U_covariance']/pooled[readout]['Y_variance']
    r1=pooled['R1'];r0=pooled['R0']
    checks={'H2a_positive_U':r1['U_covariance']>0 and r0['U_covariance']>0,
            'H2a_below_10pct':r1['U_fraction']<.1 and r0['U_fraction']<.1,
            'H4_E1_aligned_n3':r1['H4_aligned_slope']>0 and r0['H4_aligned_slope']>0,
            'H5_content_gain':all(x[k]>0 for x in [r0,r1] for k in ['H5_gain_vs_E1','H5_gain_vs_matched']),
            'H3prime_direction':None}
    tests=[]
    for t in prior['primary_family']:
        check=checks[t['name']]
        tests.append({**t,'matched_G3_R1_point_direction_agrees':check,
                      'passes_R0_and_available_R1_direction':bool(t['passes_R0_family_alpha05'] and check is True)})
    result={'schema':'voices_v31_gate2_R1_qualified_results_v1','status':'G1_G2_R0_and_G3_R1_analysis_complete',
            'primary_family':tests,'matched_models':models,'pooled_matched_two_topic_sensitivity':pooled,
            'R1_points_are_direction_checks_not_independent_significance_or_equivalence_tests':True,
            'H3prime_R1_scope':{'models_measured':6,'models_in_R0_sign_test':11,'full_eleven_model_R1_validation':False,
                'six_observed_n3_signs_match_across_readouts':sum(sign(m['readouts']['R0']['observed_U_n3_slope'])==sign(m['readouts']['R1']['observed_U_n3_slope']) and sign(m['readouts']['R1']['observed_U_n3_slope'])!=0 for m in models)},
            'limitations':['G3 contains only two original topics, not the fourteen new confirmatory topics.',
                'Content sensitivity transfers the frozen R0-trained predictor to R1 low-order features.',
                'Five new models lack G3 R1 measurements; H3-prime is not fully validated across readouts.',
                'A point estimate below10% is a sensitivity check, not a second equivalence test.'],
            'source_sha256':sources}
    write_new(root/'gate2/report.json',result);print({'status':result['status']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();run(a.base,a.extension)
