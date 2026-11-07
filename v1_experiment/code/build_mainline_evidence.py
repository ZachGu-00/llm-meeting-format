"""Reanalyse saved measurements and export every numerical source for the mainline figures.

No model inference or new fitted target labels. Confidence intervals resample topics,
sharing the resampled topic indices across the fixed models in a stage.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'mainline'
DATA = OUT / 'figure_data'
SEED, DRAWS = 20261105, 20000
LABELS = {
    'LiquidAI--LFM2.5-1.2B-Instruct':'LFM2.5-1.2B',
    'Qwen--Qwen3.5-4B':'Qwen3.5-4B', 'Qwen--Qwen3.5-2B':'Qwen3.5-2B',
    'google--gemma-4-E2B-it':'Gemma4-E2B', 'google--gemma-4-E4B-it':'Gemma4-E4B',
    'ibm-granite--granite-4.2-3b':'Granite4.2-3B',
    'microsoft--Phi-4-mini-instruct':'Phi4-mini', 'openbmb--MiniCPM5-2B':'MiniCPM5-2B',
    'HuggingFaceTB--SmolLM3-3B':'SmolLM3-3B',
    'mistralai--Ministral-3-3B-Instruct-2512-BF16':'Ministral3-3B',
    'nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16':'Nemotron3-Nano-4B',
}
RECORDS = []

def read(name, **kwargs):
    return pd.read_csv(ROOT / name, **kwargs)

def save(name, frame, figure, purpose, source, unit):
    frame = frame.copy()
    if 'model' in frame and 'model_label' not in frame:
        frame['model_label'] = frame['model'].map(LABELS)
    target = DATA / name
    frame.to_csv(target, index=False, encoding='utf-8-sig', float_format='%.17g')
    RECORDS.append({'file':'mainline/figure_data/'+name, 'figure':figure,
                    'purpose':purpose, 'source_files':source, 'analysis_unit':unit,
                    'rows':len(frame), 'columns':json.dumps(list(frame),ensure_ascii=False),
                    'bytes':target.stat().st_size, 'sha256':sha(target)})

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def weights(n):
    return np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=DRAWS)/n

def ci(x):
    return np.quantile(x,[.025,.975]).tolist()

def sig(x):
    return np.exp(-np.logaddexp(0.,-x))

def matched(metrics):
    keys=['stage','model','topic_id','readout','topic']
    records=[]
    for label, first, second in [('person','matched_size','matched_seat'),
                                  ('incidence','matched_size_incidence','matched_seat_incidence')]:
        a=metrics[(metrics.family==first)&(metrics.method=='zero')]
        b=metrics[(metrics.family==second)&(metrics.method=='zero')]
        z=a[keys+['obs_ms']].merge(b[keys+['obs_ms']],on=keys,validate='one_to_one',suffixes=('_size','_seat'))
        z['weighting']=label
        z['size_rms_pp']=100*np.sqrt(z.obs_ms_size)
        z['seat_rms_pp']=100*np.sqrt(z.obs_ms_seat)
        z['difference_pp']=z.size_rms_pp-z.seat_rms_pp
        z['size_larger']=z.difference_pp>0
        records.append(z)
    z=pd.concat(records,ignore_index=True)
    save('F1_matched_topic.csv',z,'F1','逐议题的同人对照；主分析G2/person，其余为敏感性',
         'analysis/topic_metrics.csv','topic; fixed models; paired intervention')
    keep=['stage','model','topic_id','readout','set_id','ordered_agents','support_bits',
          'base_obs','regroup_obs','resize_obs','delta_regroup_incidence','delta_resize_incidence']
    keep += [prefix+'_'+method for prefix in ['seat','size'] for method in
             ['observed','P0','present_mean','size_E2','state_E2','static_projection_E2']]
    conditions=read('analysis/matched_six.csv.gz',usecols=keep,dtype={'support_bits':str})
    save('F1_matched_conditions.csv.gz',conditions,'F1/F2',
         '全部单个初始状态和成员映射的同人干预结果及主预测器差值，直接复算主图',
         'analysis/matched_six.csv.gz','configuration nested in topic; support_bits is six-character text')
    summary=read('analysis/paired_designs.csv')
    summary=summary[summary['first'].isin(['matched_size','matched_size_incidence'])]
    save('F1_matched_model_summary.csv',summary,'F1','逐模型RMS、配对差异及议题聚类区间',
         'analysis/paired_designs.csv','topic clusters; 20000 draws; fixed models')
    pooled=[]
    for (stage,label),g in z.groupby(['stage','weighting'],sort=True):
        aa=g.pivot(index='model',columns='topic_id',values='obs_ms_size').sort_index(axis=1)
        bb=g.pivot(index='model',columns='topic_id',values='obs_ms_seat').reindex(index=aa.index,columns=aa.columns)
        assert not aa.isna().any().any() and not bb.isna().any().any()
        a,b=aa.to_numpy(),bb.to_numpy();w=weights(a.shape[1])
        boot=(100*(np.sqrt(w@a.T)-np.sqrt(w@b.T))).mean(axis=1)
        point=100*(np.sqrt(a.mean(1))-np.sqrt(b.mean(1))).mean()
        lo,hi=ci(boot)
        pooled.append({'stage':stage,'weighting':label,'n_models':a.shape[0],'n_topics':a.shape[1],
                       'rms_difference_pp':point,'ci_lo':lo,'ci_hi':hi,
                       'positive_model_topic_cells':int(g.size_larger.sum()),'model_topic_cells':len(g),
                       'positive_models':int((a.mean(1)>b.mean(1)).sum()),
                       'seed':SEED,'draws':DRAWS,'inference':'exploratory_pointwise_95pct'})
    pooled=pd.DataFrame(pooled)
    old=json.loads((ROOT/'analysis/pooled_primary.json').read_text())
    main=pooled[(pooled.stage=='G2')&(pooled.weighting=='person')].iloc[0]
    assert abs(main.rms_difference_pp-old['matched_rms_difference_pp'])<1e-12
    np.testing.assert_allclose([main.ci_lo,main.ci_hi],old['interval95'],atol=1e-12)
    save('F1_pooled_contrasts.csv',pooled,'F1','共享议题重采样的跨固定模型平均；各阶段分别统计',
         'F1_matched_topic.csv; analysis/pooled_primary.json','topic clusters shared across fixed models')
    return pooled

def predictors(metrics):
    save('F2_prediction_topic.csv',metrics,'F2','各设计族、预测器、阶段的逐议题观测与误差二阶矩',
         'analysis/topic_metrics.csv','topic; configurations within topic are averaged')
    scores=read('analysis/method_scores.csv')
    save('F2_prediction_summary.csv',scores,'F2','逐模型预测性能；负R2保留',
         'analysis/method_scores.csv','topic clusters; 20000 draws')
    paired=read('analysis/paired_predictors.csv')
    selected=paired[((paired.method=='size_E2')&(paired.reference=='static_projection_E2'))|
                    ((paired.method=='state_E2')&(paired.reference=='size_E2'))|
                    ((paired.method=='size_E2')&(paired.reference=='present_mean'))]
    save('F2_placement_and_complexity_pairs.csv',selected,'F2','同权重放置位置、增加立场参数、学习权重相对简单均值的配对对照',
         'analysis/paired_predictors.csv','topic clusters; exploratory pointwise intervals')
    return scores,paired

def jensen():
    """Independently reconstruct E4c predictions and verify the averaging identity."""
    scalar=read('analysis/weights.csv.gz');scalar=scalar[scalar.source=='size']
    lookup={(r.stage,r.model,r.topic_id,r.readout,int(r.n_peers_or_order)):r.weight_opp_or_scalar
            for r in scalar.itertuples()}
    reference=read('analysis/five_vs_three.csv.gz',dtype={'support_bits':str})
    ref={(r.stage,r.model,r.topic_id,r.readout,int(r.set_id),r.support_bits):r for r in reference.itertuples()}
    manifests=[]
    for name in ['e4_cache_manifest.json','M1_manifest.json']:
        manifests.extend(json.loads((ROOT/'inputs'/name).read_text())['topics'])
    meta={r['topic_id']:r for r in manifests}
    coverage=read('coverage.csv');records=[];maxerror=0.
    st=((np.arange(32)[:,None]>>np.arange(4,-1,-1))&1)
    for row in coverage.itertuples():
        tag='__'.join([row.stage,row.model,row.topic_id,row.readout])
        w2=lookup[row.stage,row.model,row.topic_id,row.readout,2]
        w4=lookup[row.stage,row.model,row.topic_id,row.readout,4]
        with np.load(ROOT/'cache'/(tag+'.npz')) as c:
            for si,ss in enumerate(meta[row.topic_id]['five_person_sets']):
                agents=sorted(ss);logits=[];big=[];mean_logits=[];mean_big=[]
                for jpos,j in enumerate(agents):
                    sj=st[:,jpos];alpha=c['alpha'][j,sj]
                    peers=[a for a in range(5) if a!=jpos]
                    v={k:c['pair'][j,agents[k],2*sj+st[:,k]]-alpha for k in peers}
                    x=np.stack([alpha+w2*(v[a]+v[b]) for a,b in combinations(peers,2)],axis=1)
                    logits.append(x);big.append(alpha+w4*sum(v.values()))
                    mean_logits.append(np.stack([alpha+(v[a]+v[b])/2 for a,b in combinations(peers,2)],axis=1))
                    mean_big.append(alpha+sum(v.values())/4)
                x=np.stack(logits,axis=1);large=sig(np.stack(big,axis=1)).mean(axis=1)
                small_local=sig(x).mean(axis=(1,2));small_static=sig(x.mean(axis=2)).mean(axis=1)
                dl=large-small_local;ds=large-small_static;jg=small_local-small_static
                mx=np.stack(mean_logits,axis=1);mb=sig(np.stack(mean_big,axis=1)).mean(axis=1)
                ml=sig(mx).mean(axis=(1,2));ms=sig(mx.mean(axis=2)).mean(axis=1)
                for k in range(32):
                    bits=format(k,'05b');r=ref[row.stage,row.model,row.topic_id,row.readout,si,bits]
                    errors=[abs(dl[k]-r.delta_size_E2),abs(ds[k]-r.delta_static_projection_E2),abs(dl[k]-ds[k]+jg[k]),
                            abs(mb[k]-ml[k]-r.delta_present_mean),abs(mb[k]-ms[k])]
                    maxerror=max(maxerror,*errors)
                    records.append({'stage':row.stage,'model':row.model,'topic_id':row.topic_id,'readout':row.readout,
                        'set_id':si,'agents':' '.join(map(str,agents)),'support_bits':bits,
                        'five_observed':r.five_observed,'triangles_observed':r.triangles_observed,
                        'delta_observed':r.delta_observed,'size_weight_two_peers':w2,'size_weight_four_peers':w4,
                        'effective_weight_large':w4,'effective_weight_triangles':w2/2,
                        'five_local_and_static':large[k],'triangles_local':small_local[k],'triangles_static':small_static[k],
                        'delta_local':dl[k],'delta_static':ds[k],'delta_local_minus_static':dl[k]-ds[k],
                        'jensen_gap_triangles':jg[k],'mean_within_person_event_logit_variance':x[k].var(axis=1).mean(),
                        'simple_mean_five':mb[k],'simple_mean_triangles_local':ml[k],
                        'simple_mean_triangles_static':ms[k],'simple_mean_static_delta':mb[k]-ms[k],
                        'simple_mean_local_delta':mb[k]-ml[k],'simple_mean_jensen_gap':ml[k]-ms[k]})
    df=pd.DataFrame(records);assert maxerror<1e-12 and len(df)==len(reference)
    save('F2_E4c_jensen_conditions.csv',df,'F2/S1','同一套人数权重下逐条件重建；局部/静态预测差等于负Jensen差',
         'cache/*.npz; analysis/weights.csv.gz; inputs/*manifest.json; analysis/five_vs_three.csv.gz',
         'model-topic-set-support configuration; support_bits must remain text')
    summaries=[]
    for keys,g in df.groupby(['stage','model','topic_id','readout'],sort=True):
        obs=g.delta_observed.to_numpy();local=g.delta_local.to_numpy();static=g.delta_static.to_numpy()
        q=float(np.mean(obs**2))
        summaries.append(dict(zip(['stage','model','topic_id','readout'],keys))|
            {'n_conditions':len(g),'observed_ms':q,'local_error_ms':float(np.mean((obs-local)**2)),
             'static_error_ms':float(np.mean((obs-static)**2)),
             'jensen_rms_pp':float(100*np.sqrt(np.mean(g.jensen_gap_triangles**2))),
             'local_minus_static_mse_gain':float(np.mean((obs-static)**2-(obs-local)**2)),
             'effective_weight_large':float(g.effective_weight_large.iloc[0]),
             'effective_weight_triangles':float(g.effective_weight_triangles.iloc[0])})
    summary=pd.DataFrame(summaries)
    save('F2_E4c_jensen_topic.csv',summary,'F2/S1','逐议题的Jensen差、有效边权和误差改善；与原逐议题指标独立对账',
         'F2_E4c_jensen_conditions.csv','topic; reconstruction of existing scalar-weight predictor')
    metric=read('analysis/topic_metrics.csv');metric=metric[(metric.family=='E4c')&(metric.method=='size_E2')]
    check=summary.merge(metric,on=['stage','model','topic_id','readout'],validate='one_to_one')
    np.testing.assert_allclose(check.local_error_ms,check.error_ms,atol=1e-12)
    return {'conditions_reconstructed':len(df),'max_absolute_prediction_or_identity_error':maxerror,
            'raw_caches_used':len(coverage),'topic_metrics_reconciled':len(check)}

def micro():
    item=read('micro/count_curves_by_item.csv')
    save('F3_count_items.csv',item,'F3','所有逐题人数×立场条件，包含e2d/e2e；主图只用e2e无同盟者0至6反对者',
         'micro/count_curves_by_item.csv','item nested in topic; e2d and e2e must not be pooled')
    summaries=[]
    selected=item[(item.condition=='e2e')&(item.n_allies==0)&(item.n_opponents.between(0,6))]
    for (model,n),g in selected.groupby(['model','n_opponents'],sort=True):
        by=g.groupby('topic').delta_opposition_logit.agg(['sum','count']).sort_index()
        w=weights(len(by));boot=(w@by['sum'].to_numpy())/(w@by['count'].to_numpy());lo,hi=ci(boot)
        summaries.append({'model':model,'n_opponents':int(n),'n_items':len(g),'n_topics':len(by),
                          'mean_opposition_logit_delta':g.delta_opposition_logit.mean(),'ci_lo':lo,'ci_hi':hi,
                          'seed':SEED,'draws':DRAWS,'inference':'exploratory_pointwise_topic_cluster'})
    save('F3_count_summary.csv',pd.DataFrame(summaries),'F3','人数曲线及按议题聚类的点态区间',
         'F3_count_items.csv','item-weighted mean; topic-cluster bootstrap')
    save('F3_neutral_items.csv',read('micro/neutral_by_item.csv'),'F3/S2','配对的中性发言操作逐题数据',
         'micro/neutral_by_item.csv','paired item nested in topic')
    save('F3_micro_summary.csv',read('tables/04_micro_count_and_neutral_effects.csv'),'F3/S2','人数指数及中性效应，未定义区间保留为空',
         'tables/04_micro_count_and_neutral_effects.csv','frozen item/topic analysis; exploratory classification')
    fixed=read('micro/fixed_share.csv')
    save('F3_fixed_share.csv',fixed,'S2','固定对立/同盟占比时放大人数，含full64和sensitivity44两套',
         'micro/fixed_share.csv','paired item nested in topic; unadjusted intervals')
    return fixed

def dialogue():
    old=read('dialogue/original/run_statistics.csv');old['dataset']='G4_original'
    new=read('dialogue/replication_3090/LFM_288_runs.csv').rename(columns={'panel':'dataset','agents':'n_agents','gain':'mse_gain','mean_words':'mean_word_count'})
    new['model']='LiquidAI--LFM2.5-1.2B-Instruct'
    targets=old[old.model==new.model.iloc[0]].set_index('run_id').target_stance
    new['target_stance']=new.run_id.map(targets)
    fields=['dataset','model','run_id','topic_id','family','design','run_number','n_agents','meetings','updates',
            'target_stance','round1_target_share','round2_target_share','P0_logit_mse','PK_logit_mse','mse_gain','utterances','mean_word_count']
    runs=pd.concat([old[fields],new[fields]],ignore_index=True)
    assert len(runs)==1152 and not runs.duplicated(['dataset','model','run_id']).any()
    save('F4_dialogue_runs.csv',runs,'F4','原864次与独立288次复现的逐运行底表；使用dataset+model+run_id连接',
         'dialogue/original/run_statistics.csv; dialogue/replication_3090/LFM_288_runs.csv',
         'run nested in 6 topics; datasets must remain distinct')
    original=read('tables/05_original_dialogue_prediction.csv').rename(columns={'Holm_p_six_tests':'Holm_p'})
    original['Holm_family']='original_six_tests'
    replica=read('tables/06_LFM_3090_replication.csv').rename(columns={'Holm_p_two_tests':'Holm_p'})
    replica['Holm_family']='replication_two_tests'
    for frame in [original,replica]:frame['model']=frame.model.str.replace('/','--',regex=False)
    save('F4_dialogue_summary.csv',pd.concat([original,replica],ignore_index=True),'F4',
         '已冻结的条件预测改善和各自Holm家族；不是多轮终点优化',
         'tables/05_original_dialogue_prediction.csv; tables/06_LFM_3090_replication.csv','six topic clusters')
    save('F4_LFM_replication_pairs.csv',read('tables/08_LFM_original_replication_pairs.csv'),'F4/S3',
         'LFM原/新硬件逐运行配对，用于稳健性而非样本量合并',
         'tables/08_LFM_original_replication_pairs.csv','paired run nested in topic')
    kernels=[]
    for path in sorted((ROOT/'protocols/original_G4_model_plans').glob('*.json')):
        plan=json.loads(path.read_text());model=path.stem
        for relation,value in plan['kernels'].items():
            kernels.append({'model':model,'relation':relation,'kernel':value,
                            'n_peers':2,'prediction':'PK = alpha + va + vb - focal_stance * kernel * (abs(va)+abs(vb))/2',
                            'training_source':'E1 relation kernel frozen in original G4 plan',
                            'is_current_E2_size_weight_predictor':False,
                            'plan_source':path.relative_to(ROOT).as_posix()})
    assert len(kernels)==9
    save('F4_G4_kernel_definitions.csv',pd.DataFrame(kernels),'F4/S3',
         'G4的PK定义及冻结E1核；明确区别于当前E2人数加权模型',
         'protocols/original_G4_model_plans/*.json; code/frozen_replication/dialogue_core.py',
         'model-relation coefficient; not a new fit')

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    metrics=read('analysis/topic_metrics.csv')
    pooled=matched(metrics);scores,paired=predictors(metrics);identity=jensen();fixed=micro();dialogue()
    save('T_frozen_confirmatory_tests.csv',read('tables/07_frozen_confirmatory_tests.csv'),'Table/S4',
         '保留原预设检验及未获支持结论','tables/07_frozen_confirmatory_tests.csv','original frozen hypothesis family')
    save('T_model_versions.csv',read('tables/09_model_versions.csv'),'Table',
         '分析模型的原下载版本和出处','tables/09_model_versions.csv','model metadata')
    pd.DataFrame(RECORDS).to_csv(OUT/'figure_data_manifest.csv',index=False,encoding='utf-8-sig')
    relevant=paired[(paired.stage=='G2')&(paired.family.isin(['E4c','matched_size']))&
                    (((paired.method=='size_E2')&(paired.reference=='static_projection_E2'))|
                     ((paired.method=='state_E2')&(paired.reference=='size_E2')))]
    fixed_main=fixed[(fixed.cohort=='full64')&fixed.ratio.isin(['2:1','1:1'])]
    result={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','new_model_inference':False,
            'bootstrap':{'seed':SEED,'draws':DRAWS,'unit':'topic','models':'fixed','status':'exploratory'},
            'matched_pooled':pooled.to_dict('records'),'jensen_reconstruction':identity,
            'selected_paired_results':relevant.to_dict('records'),
            'fixed_share_intervals_excluding_zero':fixed_main[(fixed_main.ci_lo>0)|(fixed_main.ci_hi<0)].to_dict('records'),
            'figure_source_files':len(RECORDS),'figure_source_rows':sum(r['rows'] for r in RECORDS)}
    (OUT/'evidence_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','figure_source_files':len(RECORDS),'figure_source_rows':result['figure_source_rows'],
                      'jensen_reconstruction':identity,'matched_G2':pooled[pooled.stage=='G2'].to_dict('records')}),flush=True)

if __name__=='__main__':
    main()
