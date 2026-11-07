"""Summarize the complete new LFM panel using the original analysis definitions."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path('/workspace/voices_backup_recheck_20261106');RUN=ROOT/'LFM_rerun_3090'
sys.path[:0]=[str(ROOT/'runtime/ext'),str(ROOT/'runtime/base/code')]
from dialogue_analysis import family_analysis,bootstrap_indices
PANEL='LFM_3090_20261106'
def read(p):return json.loads(Path(p).read_text())
def savecsv(p,header,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def render(row):
    info=row['bundle'];p=Path(info['archive'])
    with p.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=info['archive_sha256']:raise ValueError(p)
    updates=[];utterances=[];readouts=[]
    with tarfile.open(p,'r|gz') as t:
        for m in t:
            if m.name=='run_config.json':
                config=json.load(t.extractfile(m))
            elif m.name.endswith('/completion.json'):
                e=json.load(t.extractfile(m))['event'];meeting=e['meeting']
                for u in e['member_updates']:
                    l=u['logits']
                    updates.append([meeting['round'],meeting['index'],u['focal_agent_id'],u['before'],u['after'],u['sampled_uniform'],l['measured'],l['P0'],l['PK'],e['q_mean']])
            elif '/generation_' in m.name and m.name.endswith('.json'):
                g=json.load(t.extractfile(m));utterances.append([g['meeting']['index'],g['focal_agent_id'],g['seed'],g['word_count'],g['generated_token_count'],g['reached_token_budget'],g['normalized_display_text']])
            elif m.name.endswith('/raw.jsonl'):
                for line in t.extractfile(m):
                    r=json.loads(line);y=r['R0_y_AB'];support=y if r['positive_letter']=='A' else -y
                    readouts.append([r['record_id'],r['round'],r['meeting_index'],r['focal_agent_id'],r['condition'],r['permutation'],r['label_map'],r['positive_letter'],y,support,r['q'],r['prompt_token_count']])
    prefix=[PANEL,row['run_id'],config['topic_id'],config['family'],config['design_id'],config['run_number']]
    if len(updates)!=row['summary']['n_updates'] or len(updates)!=len(utterances) or len(readouts)!=len(updates)*14:raise ValueError('incomplete export '+row['run_id'])
    return [prefix+r for r in updates],[prefix+r for r in utterances],[prefix+r for r in readouts]
def main():
    out=RUN/'export';out.mkdir(exist_ok=True)
    plan=read(RUN/'plan.json');reports=[read(p) for p in sorted((RUN/'run_status').glob('*.json'))]
    wanted={r['run_id'] for r in plan['runs']}
    if len(reports)!=288 or {r['run_id'] for r in reports}!=wanted:raise ValueError('full panel incomplete')
    rows=[r['summary'] for r in reports];family=[family_analysis(rows,f,10000,20261101)[0] for f in ('G6','STS15')]
    # Preserve the original six-comparison report. This repetition has its own two-family correction.
    rawp=[f['primary_conditional_loss_improvement']['two_sided_recentered_bootstrap_p'] for f in family]
    order=np.argsort(rawp);running=0.;adjusted=[0.,0.]
    for i,j in enumerate(order):
        running=max(running,min(1.,(len(order)-i)*rawp[j]));adjusted[j]=running
    for f,p in zip(family,adjusted):f['replication_two_family_Holm_p']=p
    old=read('/workspace/voices_export_20261105/inputs/G4__LiquidAI--LFM2.5-1.2B-Instruct__analysis.json')['runs'];oldmap={r['run_id']:r for r in old}
    fields='panel run_id topic_id family design run_number agents meetings updates round1_target_share round2_target_share P0_logit_mse PK_logit_mse gain q_mean utterances mean_words'.split()
    records=[];paired=[];comparisons=[]
    for row in rows:
        rounds=sorted(row['rounds'],key=lambda x:x['round']);previous=oldmap[row['run_id']];pr=sorted(previous['rounds'],key=lambda x:x['round'])
        records.append([PANEL,row['run_id'],row['topic_id'],row['family'],row['design_id'],row['run_number'],row['n_agents'],row['n_meetings'],row['n_updates'],rounds[0]['observed_target_adoption'],rounds[1]['observed_target_adoption'],row['metrics']['P0']['logit_squared_error'],row['metrics']['PK']['logit_squared_error'],row['primary_P0_minus_PK_logit_squared_error'],row['q_mean'],row['generation_quality']['utterances'],row['generation_quality']['word_count_mean']])
        paired.append([row['run_id'],row['topic_id'],row['family'],row['design_id'],row['run_number'],previous['primary_P0_minus_PK_logit_squared_error'],row['primary_P0_minus_PK_logit_squared_error'],pr[1]['observed_target_adoption'],rounds[1]['observed_target_adoption']])
    for f in ('G6','STS15'):
        subset=[r for r in rows if r['family']==f]
        runnumbers=sorted({r['run_number'] for r in subset});designs=sorted({r['design_id'] for r in subset});look={(r['run_number'],r['design_id']):r for r in subset}
        delta=np.array([[look[n,d]['primary_P0_minus_PK_logit_squared_error']-oldmap[look[n,d]['run_id']]['primary_P0_minus_PK_logit_squared_error'] for d in designs] for n in runnumbers])
        topics=[look[n,designs[0]]['topic_id'] for n in runnumbers]
        samples=np.array([delta[ix].mean() for ix in bootstrap_indices(topics,10000,20261106)])
        comparisons.append({'family':f,'new_minus_old_conditional_gain':float(delta.mean()),'interval95':np.quantile(samples,[.025,.975]).tolist(),'scope':'paired run contexts within six old topics; new trajectories may differ'})
    savecsv(out/'LFM_288_runs.csv',fields,records)
    savecsv(out/'old_vs_3090.csv','run_id topic_id family design run_number old_gain new_gain old_round2_target_share new_round2_target_share'.split(),paired)
    savecsv(out/'native_bundle_index.csv','panel run_id archive_bytes sha256'.split(),[[PANEL,r['run_id'],r['bundle']['archive_bytes'],r['bundle']['archive_sha256']] for r in reports])
    totals={'updates':0,'utterances':0,'readouts':0};prefix='panel run_id topic_id family design run_number'.split()
    with gzip.open(out/'member_updates.csv.gz','wt',newline='',encoding='utf-8',compresslevel=3) as a,gzip.open(out/'utterances.csv.gz','wt',newline='',encoding='utf-8',compresslevel=3) as b,gzip.open(out/'R0_readouts.csv.gz','wt',newline='',encoding='utf-8',compresslevel=3) as c,ProcessPoolExecutor(max_workers=4) as pool:
        wa,wb,wc=map(csv.writer,(a,b,c))
        wa.writerow(prefix+'round meeting focal before after draw measured_logit P0_logit PK_logit q_mean'.split())
        wb.writerow(prefix+'meeting focal seed words tokens hit_token_limit text'.split())
        wc.writerow(prefix+'record_id round meeting focal condition permutation label_map positive_letter y_AB support_logit q prompt_tokens'.split())
        for updates,utterances,readouts in pool.map(render,reports,chunksize=2):
            wa.writerows(updates);wb.writerows(utterances);wc.writerows(readouts)
            totals['updates']+=len(updates);totals['utterances']+=len(utterances);totals['readouts']+=len(readouts)
    backup=Path('/workspace');manifest=read(backup/'MANIFEST.json')['files'];available=0;diagnostics_missing=0
    for name,spec in manifest.items():
        is_native=name.startswith('G4_native/LiquidAI--LFM2.5-1.2B-Instruct/')
        is_diagnostic=name in ('analysis_results/D1_D6/D3_D4/status.json','analysis_results/D1_D6/D5/status.json','analysis_results/D1_D6/D6/status.json')
        if not (is_native or is_diagnostic):continue
        path=backup/name;valid=False
        if path.is_file():
            with path.open('rb') as f:valid=hashlib.file_digest(f,'sha256').hexdigest()==spec['sha256']
        if is_native:available+=int(valid)
        if is_diagnostic:diagnostics_missing+=int(not valid)
    result={'utc':datetime.now(timezone.utc).isoformat(),'panel':PANEL,'verified_native_runs':288,'original_native_runs_available':available,'original_native_bundles_unreadable':288-available,'original_diagnostic_status_files_unavailable':diagnostics_missing,'counts':totals,'families':family,'old_new_comparison':comparisons,'new_platform_validation':read(RUN/'platform_validation.json'),'analysis_scope':'full preselected 288-run new-platform replication, exact original checkpoint and prompts; all old artifacts preserved'}
    (out/'replication_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINALIZED',totals,flush=True)
if __name__=='__main__':main()

