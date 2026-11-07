"""Check plot-data receipts, intervention controls and numerical reconstruction."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from itertools import combinations
import hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'mainline';DATA=OUT/'figure_data'

def read(path,**kwargs):return pd.read_csv(path,**kwargs)

def main():
    manifest=read(OUT/'figure_data_manifest.csv')
    assert len(manifest)==20 and not manifest.file.duplicated().any()
    count=0
    for r in manifest.itertuples():
        p=ROOT/r.file
        assert p.is_file() and p.stat().st_size==r.bytes
        with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r.sha256
        data=read(p);assert len(data)==r.rows and list(data)==json.loads(r.columns)
        count+=len(data)
    assert count==371057
    t=list(combinations(range(5),3));big=tuple(range(5))
    p1=[(0,2,4),(0,3,5),(1,2,5),(1,3,4)]
    p2=[(0,2,5),(0,3,4),(1,2,4),(1,3,5)]
    base=t*2+p1*5;seat=t*2+p2*5;resize=[big]*6+p1*5
    def projection(groups):return Counter(p for g in groups for p in combinations(sorted(g),2))
    assert projection(base)==projection(seat)==projection(resize)
    def moved(a,b):return sum(n*len(g)*(len(g)-1)//2 for g,n in (Counter(a)-Counter(b)).items())
    assert [moved(base,seat),moved(seat,base),moved(base,resize),moved(resize,base)]==[60]*4
    condition=read(DATA/'F2_E4c_jensen_conditions.csv',dtype={'support_bits':str})
    assert len(condition)==15552 and condition.support_bits.str.len().eq(5).all()
    assert condition.groupby(['stage','model','topic_id','readout']).size().eq(96).all()
    j_error=np.abs(condition.delta_local_minus_static+condition.jensen_gap_triangles).max()
    mean_error=np.abs(condition.simple_mean_static_delta).max()
    mean_j_error=np.abs(condition.simple_mean_local_delta+condition.simple_mean_jensen_gap).max()
    assert max(j_error,mean_error,mean_j_error)<1e-12
    raw=read(ROOT/'analysis/five_vs_three.csv.gz',dtype={'support_bits':str})
    key=['stage','model','topic_id','readout','set_id','support_bits']
    merged=condition.merge(raw,on=key,validate='one_to_one',suffixes=('','_original'))
    assert len(merged)==len(condition)
    for a,b in [('delta_observed','delta_observed_original'),('delta_local','delta_size_E2'),
                ('delta_static','delta_static_projection_E2'),('simple_mean_local_delta','delta_present_mean')]:
        np.testing.assert_allclose(merged[a],merged[b],rtol=0,atol=1e-12)
    data=read(DATA/'F2_prediction_topic.csv')
    scores=read(DATA/'F2_prediction_summary.csv')
    matched=read(DATA/'F1_matched_conditions.csv.gz',dtype={'support_bits':str})
    assert len(matched)==311040 and matched.support_bits.str.len().eq(6).all()
    grouping=['stage','model','topic_id','readout']
    assert not matched.duplicated(grouping+['set_id','ordered_agents','support_bits']).any()
    assert matched.groupby(grouping).size().eq(1920).all()
    reconciled=0
    for family,column in [('matched_seat','seat_observed'),('matched_size','size_observed'),
                          ('matched_seat_incidence','delta_regroup_incidence'),('matched_size_incidence','delta_resize_incidence')]:
        actual=matched.assign(squared=matched[column]**2).groupby(grouping,as_index=False).squared.mean()
        reference=data[(data.family==family)&(data.method=='zero')]
        comparison=actual.merge(reference,on=grouping,validate='one_to_one')
        assert len(comparison)==162
        np.testing.assert_allclose(comparison.squared,comparison.obs_ms,rtol=0,atol=1e-12)
        reconciled+=len(comparison)
    for r in scores[(scores.stage=='G2')&(scores.family=='matched_size')&scores.method.isin(
                    ['P0','present_mean','size_E2','state_E2','static_projection_E2'])].itertuples():
        g=data[(data.stage==r.stage)&(data.model==r.model)&(data.family==r.family)&(data.method==r.method)]
        assert len(g)==14
        np.testing.assert_allclose(1-g.error_ms.mean()/g.obs_ms.mean(),r.r2_delta,rtol=0,atol=1e-12)
        cond=matched[(matched.stage=='G2')&(matched.model==r.model)]
        by=cond.assign(error=(cond.size_observed-cond['size_'+r.method])**2).groupby('topic_id').error.mean()
        truth=cond.assign(squared=cond.size_observed**2).groupby('topic_id').squared.mean()
        np.testing.assert_allclose(1-by.mean()/truth.mean(),r.r2_delta,rtol=0,atol=1e-12)
    runs=read(DATA/'F4_dialogue_runs.csv')
    assert len(runs)==1152 and not runs.duplicated(['dataset','model','run_id']).any()
    assert runs.groupby('dataset').size().to_dict()=={'G4_original':864,'LFM_3090_20261106':288}
    kernels=read(DATA/'F4_G4_kernel_definitions.csv')
    assert len(kernels)==9 and kernels.n_peers.eq(2).all()
    assert (~kernels.is_current_E2_size_weight_predictor).all()
    for r in kernels.itertuples():
        plan=json.loads((ROOT/r.plan_source).read_text())
        assert abs(r.kernel-plan['kernels'][r.relation])<1e-12
    checks={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
            'figure_source_csvs':len(manifest),'exported_rows_including_overlapping_summaries':count,
            'matched_pair_projection':'exactly_equal','moved_dyadic_exposures_each_side':60,
            'reconstructed_conditions':len(condition),'max_jensen_identity_error':float(j_error),
            'matched_condition_rows_reconciled':len(matched),'matched_topic_second_moments_reconciled':reconciled,
            'max_simple_mean_static_contrast':float(mean_error),
            'max_simple_mean_jensen_identity_error':float(mean_j_error),
            'contrast_scores_recomputed':30,'original_runs':864,'replication_runs':288,
            'G4_prediction_identity':'E1_relation_kernel_not_E2_size_weights',
            'validation_scope':'numerical source data; no rendered figures required','new_model_inference':False,
            'inference_status':'exploratory; pointwise intervals; topic clusters; fixed models'}
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
