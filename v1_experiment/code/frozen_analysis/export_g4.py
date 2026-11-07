"""Export all available native G4 events, model predictions, and generated remarks."""
from extract_render import BASE,OUT,digest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,json,tarfile,hashlib
def run(job):
    rel,sha=job;p=BASE/rel
    if not p.is_file():return rel,'missing',[],[],None
    if digest(p)!=sha:return rel,'hash_mismatch',[],[],None
    rows=[];words=[];events=[];gens=[]
    with tarfile.open(p,'r|gz') as tar:
        for m in tar:
            if m.name=='run_config.json':cfg=json.load(tar.extractfile(m))
            elif m.name=='run_completion.json':end=json.load(tar.extractfile(m))
            elif m.name.endswith('/completion.json'):events.append(json.load(tar.extractfile(m))['event'])
            elif '/generation_' in m.name and m.name.endswith('.json'):gens.append(json.load(tar.extractfile(m)))
    if len(events)!=len(end['meetings']):raise ValueError(rel)
    common=[p.parent.name,cfg['run_id'],cfg['topic_id'],cfg['topic'],cfg['family'],cfg['design_id'],cfg['run_number']]
    for e in sorted(events,key=lambda x:x['meeting']['index']):
        met=e['meeting']
        for u in e['member_updates']:
            l=u['logits'];rows.append([*common,met['round'],met['index'],' '.join(map(str,met['members'])),u['focal_agent_id'],u['before'],u['after'],u['sampled_uniform'],l['measured'],l['P0'],l['PK'],e['q_mean']])
    for x in gens:
        met=x['meeting'];mi=met['index'] if isinstance(met,dict) else int(met)
        words.append([*common,mi,x['focal_agent_id'],x['word_count'],x['reached_token_budget'],x['normalized_display_text']])
    if len(rows)!=len(words):raise ValueError(rel+' generation count')
    tau=cfg['target_stance'];final=end['final_state']
    return rel,'verified',rows,words,[*common,tau,len(cfg['agent_ids']),len(events),sum(x==tau for x in cfg['initial_state'])/len(final),sum(x==tau for x in final)/len(final),sha]
def main():
    manifest=json.loads((BASE/'MANIFEST.json').read_text())['files'];dest=OUT/'dialogue';dest.mkdir(exist_ok=True)
    jobs=sorted((k,v['sha256']) for k,v in manifest.items() if k.startswith('G4_native/') and k.endswith('.tar.gz')); assert len(jobs)==864
    common='model run_id topic_id topic family design run_number'.split()
    with gzip.open(dest/'member_updates.csv.gz','wt',encoding='utf-8',newline='',compresslevel=3) as f,gzip.open(dest/'utterances.csv.gz','wt',encoding='utf-8',newline='',compresslevel=3) as g,(dest/'run_endpoints.csv').open('w',newline='') as h,(dest/'native_coverage.csv').open('w',newline='') as ix, ProcessPoolExecutor(max_workers=4) as pool:
        w,z,e,i=map(csv.writer,(f,g,h,ix));w.writerow(common+'round meeting members focal before after draw measured_logit P0_logit PK_logit meeting_q_mean'.split());z.writerow(common+'meeting focal word_count hit_token_limit text'.split());e.writerow(common+'target_stance n_members n_meetings initial_target_share final_target_share bundle_sha256'.split());i.writerow(['bundle','status','n_updates'])
        for path,status,rows,words,end in pool.map(run,jobs,chunksize=3):
            i.writerow([path,status,len(rows)])
            if end:w.writerows(rows);z.writerows(words);e.writerow(end)
if __name__=='__main__':main()
