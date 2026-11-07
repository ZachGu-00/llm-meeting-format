"""Topic-cluster paired inference and independent reconciliation of exports."""
import csv,gzip,hashlib,itertools,json,math,shutil
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from extract_render import OUT,BASE,digest
from fit_weights import read
D=OUT/'analysis';A=OUT/'inputs';B=20000
def write(name,fields,rows):
    with name.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)
def interval(a):return np.quantile(a,[.025,.975]).tolist() if np.isfinite(a).all() else [np.nan,np.nan]
def weights(n):
    return np.random.default_rng(20261105).multinomial(n,np.full(n,1/n),size=B)/n
def estimate(v,w,sqrt=False):
    fn=np.sqrt if sqrt else lambda v:v
    return [fn(v.mean()),*interval(fn(w@v))]
def stats():
    rows=list(csv.DictReader((D/'topic_metrics.csv').open()));sums=defaultdict(dict)
    for r in rows:sums[r['stage'],r['model'],r['family'],r['method']][r['topic_id']]=(float(r['obs_ms']),float(r['error_ms']))
    summ=[];pairs=[];pred=[];primary=[]
    for key in sorted(sums):
        stage,model,fam,m=key;ts=sorted(sums[key]);v=np.array([sums[key][t] for t in ts]);w=weights(len(ts));obs,err=v.T
        r2=1-err.mean()/obs.mean();br2=1-(w@err)/(w@obs)
        summ.append([*key,len(ts),*(100*np.array(estimate(obs,w,True))),*(100*np.array(estimate(err,w,True))),r2,*interval(br2)])
        if m not in ('size_E2','state_E2','static_projection_E2','E1_order234_identified'):continue
        for base in ('zero','P0','present_mean','side_mean','side_max','size_E2','static_projection_E2','E1_order2'):
            if m==base:continue
            ref=np.array([sums[stage,model,fam,base][t][1] for t in ts]);z=ref-err
            d=z.mean()/obs.mean();ci=interval((w@z)/(w@obs))
            pred.append([stage,model,fam,m,base,len(ts),z.mean(),*interval(w@z),d,*ci,int(np.sum(z>0))])
    for stage,model in sorted({k[:2] for k in sums}):
        for first,second in [('E4c','E4b'),('E4c','E4a'),('E4a','E4b'),('matched_size','matched_seat'),('matched_size_incidence','matched_seat_incidence')]:
            a=sums[stage,model,first,'zero'];b=sums[stage,model,second,'zero'];ts=sorted(a);w=weights(len(ts))
            aa=np.array([a[t][0] for t in ts]);bb=np.array([b[t][0] for t in ts])
            x=100*np.sqrt(aa.mean());y=100*np.sqrt(bb.mean());dist=100*(np.sqrt(w@aa)-np.sqrt(w@bb))
            pairs.append([stage,model,first,second,len(ts),x,y,x-y,*interval(dist),x/y,int((aa>bb).sum())])
            if stage=='G2' and first=='matched_size':primary.append((model,ts,aa,bb))
    assert len(primary)==6 and all(x[1]==primary[0][1] for x in primary)
    aa=np.stack([r[2] for r in primary]);bb=np.stack([r[3] for r in primary]);w=weights(aa.shape[1])
    bs=100*(np.sqrt(w@aa.T)-np.sqrt(w@bb.T)).mean(1)
    main={'models':[x[0] for x in primary],'topic_ids':primary[0][1],'draws':B,'estimand':'equal model average of model-specific RMS differences; shared resampling of 14 topics','matched_rms_difference_pp':float(100*(np.sqrt(aa.mean(1))-np.sqrt(bb.mean(1))).mean()),'interval95':interval(bs),'positive_models':int((aa.mean(1)>bb.mean(1)).sum()),'models_fixed_not_sampled':True,'exploratory':True}
    (D/'pooled_primary.json').write_text(json.dumps(main,indent=2))
    write(D/'method_scores.csv','stage model family method n_topics obs_rms_pp obs_lo obs_hi pred_rmse_pp err_lo err_hi r2_delta r2_lo r2_hi'.split(),summ)
    write(D/'paired_designs.csv','stage model first second n_topics first_rms_pp second_rms_pp difference_pp lo_pp hi_pp ratio topics_first_larger'.split(),pairs)
    write(D/'paired_predictors.csv','stage model family method reference n_topics mse_gain lo hi r2_gain lo_r2 hi_r2 topics_improved'.split(),pred)
    # Reconcile all 18 measured magnitudes AND 18 P0 R² values against the independent old report.
    hist=read(BASE/'analysis_results/E4_final/E4_brief.json');checks=[]
    for model in hist['models']:
        name=model['repo_id'].replace('/','--')
        for fam,record in model['single_round'].items():
            if fam not in ('E4a','E4b','E4c'):continue
            subset=[x for x in summ if x[:4]==['E4',name,fam,'P0']]
            r=subset[0];orig=record['observed_contrast_RMS']['estimate']*100;oldr=record['P0_R2_delta']['estimate']
            checks.append([name,fam,orig,r[5],r[5]-orig,oldr,r[11],r[11]-oldr])
    assert len(checks)==18 and max(max(abs(x[4]),abs(x[7])) for x in checks)<1e-8,checks
    write(D/'original_reconciliation.csv','model family frozen_rms_pp reproduced_rms_pp difference_pp frozen_P0_R2 reproduced_P0_R2 r2_difference'.split(),checks)
    print(main,flush=True)
def auxiliary():
    nd=OUT/'micro';neutral=[];neus=[]
    for p in A.glob('*__e2a_neutral0_R0_analysis.json'):
        e=read(p);q=p.with_name(p.name.replace('neutral0','neutral3'))
        if not q.exists():continue
        f=read(q);lookup={k:i for i,k in enumerate(f['item_ids'])};zs=[]
        for i,(item,t) in enumerate(zip(e['item_ids'],e['topics'])):
            a=np.mean(e['item_singletons'][i][:3]);b=np.mean(f['item_singletons'][lookup[item]][:3])
            neutral.append([p.name.split('__')[0],item,t,a,b,b-a]);zs.append(b-a)
        topics,idx=np.unique(e['topics'],return_inverse=True);w=weights(len(topics))[:,idx];v=np.array(zs)
        neus.append([p.name.split('__')[0],len(v),len(topics),v.mean(),*interval((w@v)/w.sum(1))])
    write(nd/'neutral_by_item.csv','model item_id topic single_opp_0neutral single_opp_3neutral paired_difference'.split(),neutral)
    write(nd/'neutral_summary.csv','model n_items n_topics paired_logit_difference ci_lo ci_hi'.split(),neus)
    mats=[];des=[]
    for fn in ('e4_agents.json','e4_agents_new.json'):
        for t in read(A/fn):
            for j in t['agents']:
                for stance,x in j['arguments'].items():
                    mats.append([t['topic_id'],t['topic'],j['agent_id'],stance,x.get('argument_id'),x.get('kp_id'),x.get('kp_text'),x.get('argument'),x.get('text_sha256')])
    write(nd/'discussion_material.csv','topic_id topic agent_id stance argument_id keypoint_id keypoint text text_sha256'.split(),mats)
    for p in sorted(A.glob('G4__*__analysis.json')):
        report=read(p);model=report['repo_id'].replace('/','--')
        for r in report['runs']:
            rounds=sorted(r['rounds'],key=lambda x:x['round']);g=r['generation_quality']
            des.append([model,r['run_id'],r['topic_id'],r['family'],r['design_id'],r['run_number'],r['n_agents'],r['n_meetings'],r['n_updates'],r['target_stance'],*[s['observed_target_adoption'] for s in rounds],r['metrics']['P0']['logit_squared_error'],r['metrics']['PK']['logit_squared_error'],r['primary_P0_minus_PK_logit_squared_error'],g['utterances'],g['word_count_mean']])
    assert len(des)==864 and len({(s[0],s[1]) for s in des})==864
    write(OUT/'dialogue'/'all_864_runs.csv','model run_id topic_id family design run_number n_agents meetings updates target_stance round1_target_share round2_target_share P0_logit_mse PK_logit_mse mse_gain utterances mean_word_count'.split(),des)
    native=list(csv.DictReader((OUT/'dialogue/run_endpoints.csv').open()));index={(r[0],r[1]):r for r in des}
    for r in native:
        full=index[r['model'],r['run_id']]
        assert abs(float(r['final_target_share'])-full[11])<1e-12
    print('G4_reconciled',len(native),'of',len(des),flush=True)
def audit_backup():
    m=read(BASE/'MANIFEST.json')['files'];att=[];seen={}
    for name,ref in sorted(m.items()):
        p=BASE/name;exists=p.is_file();h=''
        if exists and p.stat().st_size==ref['bytes']:
            obj=p.stat();k=(obj.st_dev,obj.st_ino,obj.st_size);h=seen.get(k) or digest(p);seen[k]=h
        status='verified' if h==ref['sha256'] else ('missing' if not exists else 'size_or_hash_mismatch')
        att.append([name,ref['bytes'],ref['sha256'],status])
    write(OUT/'backup_audit.csv','source expected_bytes expected_sha256 status'.split(),att)
    counts=Counter(x[-1] for x in att);print('source audit',counts,flush=True)
    if counts['size_or_hash_mismatch']:print('corrupt source entries:',[x[0] for x in att if x[-1]=='size_or_hash_mismatch'][:8],flush=True)
if __name__=='__main__':stats();auxiliary();audit_backup()

