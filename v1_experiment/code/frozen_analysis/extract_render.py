"""Extract all E4-family render-level measurements and validate archived hashes.
Run on the restored Linux volume. Existing inputs are opened read-only.
"""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import re
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

BASE = Path(os.environ.get('VOICES_BACKUP', '/workspace'))
OUT = Path(os.environ.get('VOICES_OUT', '/workspace/voices_export_20261105'))
RAW = OUT/'raw'/'render'
CACHE = OUT/'cache'
WORK = OUT/'work'
EXPECTED = {}
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def archive_digest(path):
    key = path.relative_to(BASE).as_posix()
    got = digest(path)
    if got != EXPECTED[key]:
        raise ValueError(f'Archive hash mismatch: {key}')
    return got
def raw_file(stage, model, topic, readout):
    return f'{stage}__{model}__{topic}__{readout}'
def parse_member(archive, member, stage, model, topic, readout, archive_sha):
    name=raw_file(stage,model,topic,readout)
    receipt=WORK/(name+'.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['archive_sha256']==archive_sha and old['member']==member.name:
            return old
        # Some cumulative archives repeat a readout verbatim; retain its first copy.
        data=archive.extractfile(member)
        if hashlib.file_digest(data,'sha256').hexdigest()!=old['source_sha256']:
            raise ValueError('Conflicting raw copies: '+name)
        return old
    alpha=np.full((15,2),np.nan); pair=np.full((15,15,4),np.nan)
    triple=np.full((15,15,15,8),np.nan); five={}
    buckets={}; counts=Counter(); qsum=0.; qn=0; sourcehash=hashlib.sha256()
    path=RAW/(name+'.csv.gz'); temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    labels={}; totals=0; ids=set()
    with gzip.open(temp,'wt',encoding='utf-8',newline='',compresslevel=2) as f:
        writer=csv.writer(f); writer.writerow(('record_id','kind','focal','peers','state_bits','permutation','label_map','positive_letter','y_AB','support_logit','q','prompt_tokens'))
        for line in archive.extractfile(member):
            sourcehash.update(line); row=json.loads(line); kind=row['cache_kind']
            if row['status']!='ok' or row['topic_id']!=topic or row['readout']!=readout: raise ValueError(name)
            rid=row['record_id']; j=int(row['focal_agent_id']); peers=tuple(map(int,row['peer_agent_ids']))
            bits=tuple(int(x==1) for x in (row['focal_stance'],*row['peer_stances']))
            idx=0
            for v in bits: idx=2*idx+v
            pos=int(row['permutation']); lm=int(row['label_map']); posletter=row['positive_letter']
            if rid in ids or posletter not in ('A','B') or j in peers or tuple(sorted(peers))!=peers: raise ValueError(name+' duplicate or shape')
            ids.add(rid); original=float(row[readout+'_y_AB']); val=original if posletter=='A' else -original
            if not np.isfinite(val): raise ValueError(name+' nonfinite')
            k=(kind,j,peers,idx); entries=buckets.setdefault(k,{})
            if (pos,lm) in entries: raise ValueError('duplicate render '+name)
            entries[pos,lm]=(val,posletter)
            q=row.get('q'); writer.writerow((rid,kind,j,' '.join(map(str,peers)),''.join(map(str,bits)),pos,lm,posletter,original,val,'' if q is None else q,row.get('prompt_token_count','')))
            if q is not None: qsum+=q;qn+=1
            labels[row['topic']]=1; counts[kind]+=1;totals+=1
    if counts!=dict(baseline=60,pair=3360,triple=43680,five=1920) or len(labels)!=1: raise ValueError(f'{name}: {counts}')
    for (kind,j,peers,idx),values in buckets.items():
        positions={p for p,l in values}
        if len(positions)!=(1 if kind=='baseline' else 2):raise ValueError('missing positions')
        for p in positions:
            if values[p,0][1]==values[p,1][1]:raise ValueError('label reversal failed')
        v=sum(x[0] for x in values.values())/len(values)
        if kind=='baseline':alpha[j,idx]=v
        elif kind=='pair':pair[j,peers[0],idx]=v
        elif kind=='triple':triple[j,peers[0],peers[1],idx]=v
        else:five[(j,*peers,idx)]=v
    if not np.isfinite(alpha).all() or np.isfinite(pair).sum()!=840 or np.isfinite(triple).sum()!=10920 or len(five)!=480:raise ValueError('incomplete group states')
    keys=sorted(five)
    cached=CACHE/(name+'.npz'); ctemp=cached.with_name(cached.name+f'.{os.getpid()}.tmp')
    with ctemp.open('wb') as f:
        np.savez_compressed(f,alpha=alpha,pair=pair,triple=triple,five_keys=np.array(keys,dtype=np.int16),five_values=np.array([five[x] for x in keys]))
    os.replace(temp,path);os.replace(ctemp,cached)
    out=dict(stage=stage,model=model,topic_id=topic,topic=next(iter(labels)),readout=readout,rows=totals,**counts,unique_states=len(buckets),source_sha256=sourcehash.hexdigest(),archive_sha256=archive_sha,member=member.name,archive=os.path.relpath(archive.name,BASE),q_mean=qsum/qn if qn else None)
    rp=receipt.with_name(receipt.name+f'.{os.getpid()}.tmp');rp.write_text(json.dumps(out));os.replace(rp,receipt)
    print(name,flush=True)
    return out
def process(pathname):
    path=Path(pathname);sig=archive_digest(path); reports=[]
    with tarfile.open(path,'r|gz') as a:
        for m in a:
            if not m.isfile(): continue
            mm=re.fullmatch(r'(?:extension/v31/sealed/(G[123])/([^/]+)|results/([^/]+)/e4_cache)/((?:V3|V31)_E4_TOPIC_\d+)_(R[01])\.jsonl',m.name)
            if mm:
                stage,model,original,topic,readout=mm.groups()
                reports.append(parse_member(a,m,stage or 'E4',model or original,topic,readout,sig))
            elif (m.name.endswith(('/e4_cache_manifest.json','/e4_agents_new.json','/M1_manifest.json','/sts15_all80.json','/e4_agents.json','_E4_analysis.json')) and m.size<4000000):
                # identical source names in incremental archives are de-duplicated by content.
                data=a.extractfile(m).read(); h=hashlib.sha256(data).hexdigest()
                dest=WORK/'reference'/(h[:12]+'__'+Path(m.name).name)
                if not dest.exists():dest.write_bytes(data)
    return len(reports)
def main():
    global EXPECTED
    for d in (RAW,CACHE,WORK/'reference'):d.mkdir(parents=True,exist_ok=True)
    EXPECTED={s.split(maxsplit=1)[1].strip().lstrip('*'):s.split(maxsplit=1)[0] for s in (BASE/'SHA256SUMS').read_text().splitlines()}
    paths=sorted((BASE/'original_v3').glob('*_E4_topic_*.tar.gz'))
    paths+=sorted(p for p in (BASE/'extension_archives').glob('*.tar.gz') if re.search(r'G[123]_sealed_(?:model|task)_',p.name))
    # fork workers inherit the manifest on Linux.
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(process,paths,chunksize=1))
    allrows=[json.loads(p.read_text()) for p in WORK.glob('*__*__*__R?.json')]
    found=Counter(r['stage'] for r in allrows)
    if found!=dict(E4=36,G1=30,G2=84,G3=12):raise ValueError(f'incomplete inventory {found}')
    with (OUT/'coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(sorted(allrows,key=lambda r:(r['stage'],r['model'],r['topic_id'])))
    print('COMPLETE',found,flush=True)
if __name__=='__main__':main()
