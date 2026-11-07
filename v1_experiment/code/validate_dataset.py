"""Validate source hashes, dataset separation and reported numerical summaries."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,gzip,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
def rows(path):
    p=ROOT/path;opener=gzip.open if p.name.endswith('.gz') else open
    with opener(p,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b,tol=1e-8):
    if abs(float(a)-float(b))>tol:raise ValueError(f'Numeric mismatch {a} versus {b}')
def source_checks():
    imported=list(rows('provenance/source_map.csv'))
    for r in imported:
        p=ROOT/r['file']
        if not p.is_file() or p.stat().st_size!=int(r['bytes']) or sha(p)!=r['sha256']:raise ValueError('Changed source '+r['file'])
    return len(imported)
def group_checks():
    d=defaultdict(list)
    for r in rows('analysis/topic_metrics.csv'):
        d[r['stage'],r['model'],r['family'],r['method']].append((r['topic_id'],float(r['obs_ms']),float(r['error_ms'])))
    for r in rows('analysis/method_scores.csv'):
        z=d[r['stage'],r['model'],r['family'],r['method']];n=len(z)
        q=sum(x[1] for x in z)/n;e=sum(x[2] for x in z)/n
        close(r['obs_rms_pp'],100*math.sqrt(q));close(r['pred_rmse_pp'],100*math.sqrt(e));close(r['r2_delta'],1-e/q)
    for r in rows('analysis/paired_designs.csv'):
        a=d[r['stage'],r['model'],r['first'],'zero'];b=d[r['stage'],r['model'],r['second'],'zero']
        aa={t:q for t,q,e in a};bb={t:q for t,q,e in b};assert set(aa)==set(bb)
        x=100*math.sqrt(sum(aa.values())/len(aa));y=100*math.sqrt(sum(bb.values())/len(bb))
        close(r['difference_pp'],x-y);close(r['ratio'],x/y)
        assert int(r['topics_first_larger'])==sum(aa[t]>bb[t] for t in aa)
    check=list(rows('analysis/original_reconciliation.csv'))
    assert len(check)==18 and max(max(abs(float(r['difference_pp'])),abs(float(r['r2_difference']))) for r in check)<1e-10
    main=list(rows('tables/01_same_people_group_effect.csv'));assert len(main)==6 and all(float(r['lo_pp'])>0 for r in main)
    return {'topic_level_groups':len(d),'original_frozen_values_reconciled':36,
            'matched_main_models':len(main),'matched_main_direction_agreement':sum(int(r['topics_first_larger']) for r in main)}
def dialogue_checks(folder,is_replication):
    if is_replication:
        statistics=list(rows(folder+'/LFM_288_runs.csv'));modelkey='panel';msekey='gain';expected_runs=288;expected_units=51552;wordkey='words'
    else:
        statistics=list(rows(folder+'/run_statistics.csv'));modelkey='model';msekey='mse_gain';expected_runs=864;expected_units=154656;wordkey='word_count'
    idx={(r[modelkey],r['run_id']):r for r in statistics};assert len(idx)==expected_runs
    units=set();scores=defaultdict(lambda:[0,0.,0.]);updates_per_run=Counter()
    for r in rows(folder+'/member_updates.csv.gz'):
        key=(r[modelkey],r['run_id']);unit=(*key,r['meeting'],r['focal']);assert unit not in units;units.add(unit)
        y=float(r['measured_logit']);p0=float(r['P0_logit']);pk=float(r['PK_logit'])
        assert all(map(math.isfinite,(y,p0,pk)));assert int(r['before']) in (-1,1) and int(r['after']) in (-1,1)
        probability=math.exp(-max(0.,-y))/(math.exp(-max(0.,-y))+math.exp(-max(0.,y)))
        assert int(r['after'])==(1 if float(r['draw'])<probability else -1)
        z=scores[key];z[0]+=1;z[1]+=(p0-y)**2;z[2]+=(pk-y)**2;updates_per_run[key]+=1
    assert len(units)==expected_units and set(scores)==set(idx)
    for k,z in scores.items():
        r=idx[k];close(r['P0_logit_mse'],z[1]/z[0]);close(r['PK_logit_mse'],z[2]/z[0]);close(r[msekey],(z[1]-z[2])/z[0])
        assert z[0]==int(r['updates'])
    text_units=set()
    for r in rows(folder+'/utterances.csv.gz'):
        key=(r[modelkey],r['run_id'],r['meeting'],r['focal']);assert key not in text_units;text_units.add(key)
        assert int(r[wordkey])==len(r['text'].split())
    assert text_units==units
    if not is_replication:
        ends=list(rows(folder+'/run_endpoints.csv'));assert len(ends)==864
        for r in ends:close(r['final_target_share'],idx[r['model'],r['run_id']]['round2_target_share'])
        assert Counter(r['model'] for r in statistics)==Counter({'Qwen--Qwen3.5-4B':288,'LiquidAI--LFM2.5-1.2B-Instruct':288,'microsoft--Phi-4-mini-instruct':288})
    else:
        count=0
        for r in rows(folder+'/R0_readouts.csv.gz'):
            y=float(r['y_AB']);s=float(r['support_logit']);close(s,y if r['positive_letter']=='A' else -y,1e-12)
            assert 0<=float(r['q'])<=1.000001;count+=1
        assert count==721728
    return {'runs':expected_runs,'utterances':len(units),'state_rule_verified':True,'prediction_losses_recomputed':True}
def raw_checks():
    cov=list(rows('coverage.csv'));assert len(cov)==162
    counts=Counter();total=0
    for r in cov:
        name='__'.join(r[k] for k in ('stage','model','topic_id','readout'))+'.csv.gz'
        with gzip.open(ROOT/'raw/render'/name,'rb') as f:n=sum(chunk.count(b'\n') for chunk in iter(lambda:f.read(1048576),b''))-1
        assert n==49020==int(r['rows']);total+=n;counts[r['stage']]+=1
    assert total==7941240 and counts==Counter({'E4':36,'G1':30,'G2':84,'G3':12})
    return {'tasks':162,'physical_readouts':total,'stages':dict(counts)}

def material_checks():
    arguments={};keypoints={};positive=set()
    for split in ('train','dev','test'):
        folder='test_data' if split=='test' else 'kpm_data'
        base='inputs/source_corpus/kpa/'+folder+'/'
        for r in rows(base+'arguments_'+split+'.csv'):arguments[split+':'+r['arg_id']]=r
        for r in rows(base+'key_points_'+split+'.csv'):keypoints[split+':'+r['key_point_id']]=r
        for r in rows(base+'labels_'+split+'.csv'):
            if r['label']=='1':positive.add((split+':'+r['arg_id'],split+':'+r['key_point_id']))
    material=list(rows('micro/discussion_material.csv'))
    for r in material:
        a=arguments[r['argument_id']];k=keypoints[r['keypoint_id']]
        assert r['text'].strip()==a['argument'].strip() and r['keypoint'].strip()==k['key_point'].strip()
        assert r['topic']==a['topic']==k['topic'] and int(r['stance'])==int(a['stance'])==int(k['stance'])
        assert (r['argument_id'],r['keypoint_id']) in positive
        assert hashlib.sha256(r['text'].encode('utf-8')).hexdigest()==r['text_sha256']
    assert len(material)==600 and len({r['topic_id'] for r in material})==20
    versions=list(rows('tables/09_model_versions.csv'))
    assert len(versions)==11 and all(r['archived_download_revision'] for r in versions)
    return {'discussion_material_rows':600,'discussion_topics':20,
            'all_arguments_and_positive_labels_matched_to_source_corpus':True,'model_versions':11}
def main():
    result={'utc':datetime.now(timezone.utc).isoformat(),'imported_source_files_verified':source_checks(),
            'main_analysis':group_checks(),'original_dialogue':dialogue_checks('dialogue/original',False),
            'replication_dialogue':dialogue_checks('dialogue/replication_3090',True),'main_raw':raw_checks()}
    result['source_material_and_versions']=material_checks()
    original=list((ROOT/'raw/original_native_supplement').rglob('*.tar.gz'))
    replica=list((ROOT/'raw/replication_native').glob('*.tar.gz'))
    assert len(original)==864 and len(replica)==288
    original_keys={(p.parent.name,p.name.removesuffix('.tar.gz')) for p in original}
    expected_original={(r['model'],r['run_id']) for r in rows('dialogue/original/run_statistics.csv')}
    replica_keys={p.name.removesuffix('.tar.gz') for p in replica}
    expected_replica={r['run_id'] for r in rows('dialogue/replication_3090/LFM_288_runs.csv')}
    assert original_keys==expected_original and replica_keys==expected_replica
    result['native_bundle_counts']={'original':len(original),'replication_3090':len(replica)}
    result['native_run_id_coverage']='exact_match_to_original_and_replication_tables'
    result['status']='PASS'
    (ROOT/'provenance/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

