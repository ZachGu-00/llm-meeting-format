"""E2d/e count grids; no invented unobserved named-peer singleton/Mobius cells."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_v3 import read_closed,execution_identity,sha,interval

def aggregate_counts(rows,readout='R0'):
    groups=defaultdict(dict);topics={};q=[]
    for r in rows:
        item=r['item_id'];key=(int(r['n_opponents']),int(r['n_allies']),r['permutation'],r['label_map'])
        if key in groups[item]:raise ValueError('Duplicate count-grid render')
        groups[item][key]=(r[readout+'_y_AB']*(1 if r['positive_letter']=='A' else -1),r['positive_letter'])
        if item in topics and topics[item]!=r['topic']:raise ValueError('Changing item topic')
        topics[item]=r['topic']
        if readout=='R0':q.append(r['q'])
    ids=sorted(groups);conditions=None;values=[]
    for item in ids:
        records=groups[item];observed=sorted({k[:2] for k in records})
        if conditions is None:conditions=observed
        if conditions!=observed:raise ValueError('Unequal count grids')
        out=[]
        for no,na in conditions:
            slots=sorted({k[2] for k in records if k[:2]==(no,na)});cell=[]
            for slot in slots:
                a,b=records[no,na,slot,0],records[no,na,slot,1]
                if a[1]==b[1]:raise ValueError('No positive label reversal')
                cell.append((a[0]+b[0])/2)
            out.append(np.mean(cell))
        values.append(out)
    if not values or (0,0) not in conditions:raise ValueError('Empty data or missing baseline')
    values=np.asarray(values);delta=values-values[:,[conditions.index((0,0))]]
    return {'item_ids':ids,'topics':[topics[i] for i in ids],'conditions':conditions,'y':values,'delta':delta,'q_median':float(np.median(q)) if q else None}

def stats(panel,ix):
    curve=panel['delta'][ix].mean(axis=0);conditions=panel['conditions']
    result={f'delta_{o}_{a}':float(v) for (o,a),v in zip(conditions,curve)}
    ns=np.array([o for o,a in conditions if a==0 and o>0],dtype=float)
    ys=np.array([curve[j] for j,(o,a) in enumerate(conditions) if a==0 and o>0])
    result['power_t_no_allies']=float(np.polyfit(np.log(ns),np.log(ys),1)[0]) if len(ns)>=2 and np.all(ys>0) else None
    result['power_fit_R2_log']=None
    if result['power_t_no_allies'] is not None:
        prediction=np.polyval(np.polyfit(np.log(ns),np.log(ys),1),np.log(ns));den=((np.log(ys)-np.log(ys).mean())**2).sum()
        if den>0:result['power_fit_R2_log']=float(1-((np.log(ys)-prediction)**2).sum()/den)
    # A count-grid contrast, not the E1 named-argument C (heterogeneous peers lack all singletons).
    if {(2,0),(0,2),(1,1),(1,0)}<=set(conditions):
        den=abs(curve[conditions.index((1,0))]);num=curve[conditions.index((2,0))]+curve[conditions.index((0,2))]-2*curve[conditions.index((1,1))]
        result['count_grouping_contrast']=float(num/den) if den>0 else None
    return result

def analyze(panel,draws=10000,seed=20261030):
    point=stats(panel,np.arange(len(panel['item_ids'])));samples={k:[] for k in point}
    topics=sorted(set(panel['topics']));clusters=[np.where(np.array(panel['topics'])==t)[0] for t in topics];rng=np.random.default_rng(seed)
    for _ in range(draws):
        ix=np.concatenate([clusters[j] for j in rng.integers(len(clusters),size=len(clusters))])
        for k,v in stats(panel,ix).items():samples[k].append(v)
    return {'n_items':len(panel['item_ids']),'n_topics':len(topics),'conditions':panel['conditions'],'point':point,
            'bootstrap':{'draws':draws,'seed':seed,'unit':'topic','intervals':{k:{'interval95':interval(v),'undefined_draws':sum(x is None for x in v)} for k,v in samples.items()}},
            'q_median':panel['q_median'],'item_ids':panel['item_ids'],'item_count_deltas':panel['delta'].tolist(),
            'not_the_full_named_argument_Mobius_kernel':True,'nonpositive_power_fit_values_not_dropped':True,
            'power_t_scope':'pooled no-ally count curve; log(delta)=intercept+t*log(n); undefined if any mean delta is nonpositive',
            'count_contrast_scope':'delta(2,0)+delta(0,2)-2delta(1,1), normalized by abs(delta(1,0)); not E1 named-peer C without exchangeability'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--results',required=True)
    p.add_argument('--readout',default='R0',choices=['R0','R1']);p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=10000)
    a=p.parse_args();identity=execution_identity([a.results]);panel=aggregate_counts(read_closed(a.inputs,a.results),a.readout)
    result=analyze(panel,a.draws);result['execution_identity']=identity
    result['file_sha256']={str(Path(f).resolve()):sha(f) for f in (a.inputs,a.results,__file__)}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'n_items':result['n_items'],'point':result['point']}))
