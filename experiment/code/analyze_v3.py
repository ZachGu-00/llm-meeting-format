"""V3 E1/E2 estimands: measured singleton rules, C, and transferable pair kernels.

No regression parameters are fitted on the multi-peer response cells.
The CLI accepts multiple closed raw/input pairs, allowing old20 + new180 reuse.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

SEED=20261030
OPP=(0,1,2);ALLY=(3,4)
PAIRS={'opponent_opponent':[(0,1),(0,2),(1,2)],
       'opponent_ally':list(itertools.product(OPP,ALLY)), 'ally_ally':[(3,4)]}
C_MASKS={0,1,2,4,8,16,3,5,6,24,9,17,10,18,12,20}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_closed(source,raw):
    """Stream source/output alignment; verify whole-file hashes and closure first."""
    source,raw=Path(source),Path(raw)
    run=read_json(str(raw)+'.run.json');summary=read_json(str(raw)+'.summary.json')
    if summary['status']!='complete' or summary['output_sha256']!=sha(raw):raise ValueError('Raw run not closed or changed')
    if summary['n_input']!=summary['n_output'] or summary['n_input']!=run['n_input']:raise ValueError('Incomplete count')
    if run['input_file_sha256']!=sha(source):raise ValueError('Source changed')
    run_sha=sha(str(raw)+'.run.json')
    if summary.get('run_file_sha256')!=run_sha:raise ValueError('Run identity changed')
    count=0
    with source.open(encoding='utf-8-sig') as fs,raw.open(encoding='utf-8-sig') as fr:
        for sl,rl in itertools.zip_longest(fs,fr):
            if sl is None or rl is None:raise ValueError('Source/result count mismatch')
            s,r=json.loads(sl),json.loads(rl);sid=s.get('record_id',s.get('id'))
            if r.get('record_id',r.get('id'))!=sid:raise ValueError('Row identity mismatch')
            if any(r.get(k)!=v for k,v in s.items()):raise ValueError('Source row modified in result')
            if r.get('status')!='ok' or r.get('run_file_sha256')!=run_sha:raise ValueError('Bad row/run binding')
            count+=1;yield r
    if count!=run['n_input']:raise ValueError('Closure count mismatch')

def aggregate(rows,readout='R0',filters=None,keep_ids=None,exclude_ids=None):
    """Average labels inside slots, then slots, preserving physical design balance."""
    cells=defaultdict(dict);metadata={};seen=set();q_values=[]
    field=readout+'_y_AB'
    for row in rows:
        if filters and any(row.get(k)!=v for k,v in filters.items()):continue
        item=row['item_id']
        if keep_ids is not None and item not in keep_ids:continue
        if exclude_ids and item in exclude_ids:continue
        rid=row.get('record_id',row.get('id'))
        if rid in seen:raise ValueError('Duplicate source/result ID across runs')
        seen.add(rid);mask=row['subset_mask']
        if not isinstance(mask,int) or not 0<=mask<32:raise ValueError('E1 named-role estimator requires five-role subset masks')
        value=float(row[field])*(1 if row['positive_letter']=='A' else -1)
        if not math.isfinite(value):raise ValueError('Nonfinite output-layer value')
        key=(mask,row['permutation'],row['label_map'])
        if key in cells[item]:raise ValueError('Duplicate factorial cell; filter experiment/neutral_count first')
        cells[item][key]=(value,row['positive_letter'])
        if item in metadata and metadata[item]!=row['topic']:raise ValueError('Changing topic within item')
        metadata[item]=row['topic']
        if readout=='R0':q_values.append(float(row['q']))
    if not cells:raise ValueError('No selected rows')
    item_ids=sorted(cells);all_masks=None;y=[]
    for item in item_ids:
        records=cells[item];masks=sorted({k[0] for k in records})
        if all_masks is None:all_masks=masks
        if masks!=all_masks:raise ValueError('Items have different observed subset panels')
        mean=np.full(32,np.nan)
        for mask in masks:
            slots=sorted({k[1] for k in records if k[0]==mask});values=[]
            for slot in slots:
                if (mask,slot,0) not in records or (mask,slot,1) not in records:raise ValueError('Missing label mate')
                a,b=records[mask,slot,0],records[mask,slot,1]
                if a[1]==b[1]:raise ValueError('Label reversal did not reverse positive option')
                values.append((a[0]+b[0])/2)
            mean[mask]=np.mean(values)
        y.append(mean)
    return {'item_ids':item_ids,'topics':[metadata[i] for i in item_ids],'masks':all_masks,
            'y':np.array(y),'n_physical_rows':len(seen),'q_median':float(np.median(q_values)) if q_values else None}

def mobius_partial(y,masks):
    result=np.full_like(y,np.nan)
    for mask in masks:
        subsets=[s for s in masks if s&mask==s]
        if len(subsets)!=2**mask.bit_count():continue
        result[:,mask]=sum(((-1)**(mask.bit_count()-s.bit_count()))*y[:,s] for s in subsets)
    return result

def rule_predictions(singletons,masks):
    predictions={name:[] for name in ('additive','present_mean','side_mean','side_max')}
    for mask in masks:
        peers=[j for j in range(5) if mask>>j&1]
        sides=[[j for j in side if j in peers] for side in (OPP,ALLY)]
        predictions['additive'].append(singletons[:,peers].sum(axis=1))
        predictions['present_mean'].append(singletons[:,peers].mean(axis=1))
        predictions['side_mean'].append(sum(singletons[:,s].mean(axis=1) if s else np.zeros(len(singletons)) for s in sides))
        strongest=[]
        for side in sides:
            if not side:strongest.append(np.zeros(len(singletons)));continue
            local=singletons[:,side];strongest.append(local[np.arange(len(local)),np.abs(local).argmax(axis=1)])
        predictions['side_max'].append(sum(strongest))
    return {k:np.stack(v,axis=1) for k,v in predictions.items()}

def ratio(numerator,denominator):
    den=float(np.sum(denominator));return float(np.sum(numerator)/den) if den>0 else None

def sufficient(panel):
    y=panel['y'];masks=panel['masks'];alpha=y[:,0]
    v=np.stack([y[:,1<<j]-alpha for j in range(5)],axis=1)
    if not np.all(np.isfinite(v)):raise ValueError('All five measured singletons and empty baseline are required')
    m=mobius_partial(y,masks);cn=np.full(len(y),np.nan)
    cd=np.abs(v[:,:3]).mean(axis=1)
    if C_MASKS<=set(masks):
        contrasts=[]
        for a,b in PAIRS['opponent_opponent']:
            contrasts.append(m[:,(1<<a)|(1<<b)]+m[:,24]-.5*sum(m[:,(1<<x)|(1<<z)] for x,z in ((a,3),(b,4),(a,4),(b,3))))
        cn=np.mean(contrasts,axis=0)
    kernels={}
    for relation,pairs in PAIRS.items():
        if all(((1<<a)|(1<<b)) in masks for a,b in pairs):
            kernels[relation]=(np.mean([m[:,(1<<a)|(1<<b)] for a,b in pairs],axis=0),
                              np.mean([.5*(np.abs(v[:,a])+np.abs(v[:,b])) for a,b in pairs],axis=0))
    joint=[s for s in masks if s.bit_count()>=2];delta=y[:,joint]-alpha[:,None]
    rules=rule_predictions(v,joint) if joint else {}
    rule_sums={name:((pred-delta)**2).sum(axis=1) for name,pred in rules.items()}
    centered=((delta-delta.mean(axis=1,keepdims=True))**2).sum(axis=1) if joint else np.zeros(len(y))
    interaction_masks=[s for s in masks if s.bit_count()>=2 and np.isfinite(m[:,s]).all()]
    mden=np.stack([np.abs(v[:,[j for j in range(5) if mask>>j&1]]).mean(axis=1) for mask in interaction_masks],axis=1)
    curves={}
    for no,na in itertools.product(range(4),range(3)):
        selected=[mask for mask in masks if (mask&7).bit_count()==no and (mask&24).bit_count()==na]
        if selected:curves[f'{no}_{na}']=y[:,selected].mean(axis=1)-alpha
    return {'y':y,'m':m,'v':v,'alpha':alpha,'C_num':cn,'C_den':cd,'kernels':kernels,
            'interaction_masks':interaction_masks,'interaction_denominators':mden,'curve_deltas':curves,
            'joint_masks':joint,'delta':delta,'rule_errors':rule_sums,'within_sst':centered}

def point(s,indices=None):
    ix=np.arange(len(s['y'])) if indices is None else np.asarray(indices)
    c=ratio(s['C_num'][ix],s['C_den'][ix]) if np.all(np.isfinite(s['C_num'][ix])) else None
    result={'C':c,'relation_kernels':{r:ratio(num[ix],den[ix]) for r,(num,den) in s['kernels'].items()},'rules':{}}
    delta=s['delta'][ix];within=float(s['within_sst'][ix].sum());pooled=float(((delta-delta.mean())**2).sum())
    for name,errors in s['rule_errors'].items():
        error=float(errors[ix].sum())
        result['rules'][name]={'R2_within_item':1-error/within if within>0 else None,
                               'R2_pooled_delta':1-error/pooled if pooled>0 else None,'SSE':error}
    means=s['m'][ix][:,s['interaction_masks']].mean(axis=0)
    dens=s['interaction_denominators'][ix].mean(axis=0)
    result['interactions']={str(mask):{'mean_raw':float(num),'normalized':float(num/den) if den>0 else None}
                            for mask,num,den in zip(s['interaction_masks'],means,dens)}
    result['count_delta_means']={name:float(values[ix].mean()) for name,values in s['curve_deltas'].items()}
    return result

def flat_metrics(p):
    return {'C':p['C'],**{'kernel_'+k:v for k,v in p['relation_kernels'].items()},
            **{name+'_'+kind:value for name,values in p['rules'].items() for kind,value in values.items() if kind.startswith('R2')},
            **{'m_'+mask+'_'+kind:value for mask,values in p['interactions'].items() for kind,value in values.items()},
            **{'count_delta_'+name:value for name,value in p['count_delta_means'].items()}}

def interval(values,level=.95):
    if any(v is None or not math.isfinite(v) for v in values):return None
    return np.quantile(values,[(1-level)/2,(1+level)/2]).tolist()

def holm(pvalues):
    """None entries remain unresolved and still count in the declared family size."""
    keys=sorted((k for k,v in pvalues.items() if v is not None),key=lambda k:pvalues[k])
    result={k:None for k in pvalues};previous=0.;m=len(pvalues)
    for rank,key in enumerate(keys):
        previous=max(previous,min(1.,(m-rank)*pvalues[key]));result[key]=previous
    return result

def analyze(panel,draws=10000,seed=SEED):
    s=sufficient(panel);estimate=point(s);flat=flat_metrics(estimate)
    topics=sorted(set(panel['topics']));groups=[np.where(np.array(panel['topics'])==t)[0] for t in topics]
    rng=np.random.default_rng(seed);samples={k:[] for k in flat}
    for _ in range(draws):
        ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        for k,v in flat_metrics(point(s,ix)).items():samples[k].append(v)
    intervals={k:{'interval95':interval(v),'undefined_draws':sum(x is None or not math.isfinite(x) for x in v)} for k,v in samples.items()}
    h1={'C_left_tail_bootstrap_p':None,'C_equivalence_margin':.1,'C_equivalence_bootstrap_p':None,
        'C_interval90':interval(samples['C'],.9),'method':'centered_topic_cluster_bootstrap; approximate one-sided/TOST tests',
        'model_types_must_be_defined_independently_of_the_tested_C':True}
    if intervals['C']['undefined_draws']==0:
        centered=np.asarray(samples['C'])-flat['C'];n=draws+1
        h1['C_left_tail_bootstrap_p']=float((1+(centered<=flat['C']).sum())/n)
        h1['C_equivalence_bootstrap_p']=float(max((1+(centered>=flat['C']+.1).sum())/n,(1+(centered<=flat['C']-.1).sum())/n))
    curves={}
    for no,na in itertools.product(range(4),range(3)):
        masks=[mask for mask in panel['masks'] if (mask&7).bit_count()==no and (mask&24).bit_count()==na]
        if masks:
            values=s['y'][:,masks].mean(axis=1)
            curves[f'{no}_{na}']={'n_opponents':no,'n_allies':na,'mean_y':float(values.mean()),
                                  'mean_delta':float((values-s['alpha']).mean()),'masks':masks}
    return {'n_items':len(panel['item_ids']),'n_topics':len(topics),'masks':panel['masks'],
      'full32':set(panel['masks'])==set(range(32)),'n_physical_rows':panel['n_physical_rows'],'q_median':panel['q_median'],
      'point':estimate,'bootstrap':{'draws':draws,'seed':seed,'unit':'topic','intervals':intervals,'undefined_draws_not_dropped':True},
      'H1':h1,'count_curves':curves,'item_ids':panel['item_ids'],'topics':panel['topics'],
      'item_singletons':s['v'].tolist(),'item_C_numerators':s['C_num'].tolist() if np.isfinite(s['C_num']).all() else None,
      'item_C_denominators':s['C_den'].tolist(),
      'mobius_by_item':{item:{str(mask):float(s['m'][i,mask]) for mask in panel['masks'] if np.isfinite(s['m'][i,mask])} for i,item in enumerate(panel['item_ids'])},
      'rule_prediction_policy':'No multi-peer fitted parameters. Alpha and v are measured from empty/singletons of each evaluated item.',
      'rule_R2_primary':'within_item; pooled_delta secondary; negative R2 retained',
      'side_max_policy':'largest absolute singleton within each prespecified stance side, retain observed sign; ties role order',
      'C_formula':'mean_items mean_opponent_pairs [mOO+mAA-0.5*(four_cross_m)] / mean_items mean_abs_opponent_singletons',
      'relation_kernel_policy':'ratio of class-mean interaction to class-mean average absolute singleton effects; no itemwise ratio averaging',
      'invariance_scope':'fixed output readout; no arbitrary monotone transformation or latent belief claim'}

def ids_from(path):
    value=read_json(path)
    if isinstance(value,dict):value=value['item_ids']
    return {v if isinstance(v,str) else v['item_id'] for v in value}

def execution_identity(paths,allow_mixed=False):
    runs=[read_json(str(path)+'.run.json') for path in paths]
    first=runs[0]
    def critical(run):
        engine=run['engine'];parent=engine['parent_loader']
        return {'repo':parent['repo_id'],'revision':parent['revision'],'precision':engine['precision'],
                'remedy':engine.get('remedy','original'),'model_kind':engine.get('model_kind','instruct'),
                'option_ids':run['option_ids']}
    if any(critical(run)!=critical(first) for run in runs):raise ValueError('Mixed models/revisions/remedies/options cannot be pooled')
    same=all(all(run[k]==first[k] for k in ('engine','batch_size','calibration_file_sha256')) for run in runs)
    if not same and not allow_mixed:raise ValueError('Mixed execution policies require explicit --allow-mixed-execution and scientific justification')
    return {'same_execution_policy':same,'mixed_execution_explicitly_allowed':bool(allow_mixed),
            'critical_identity':critical(first),'runs':[{'path':str(p),'run_sha256':sha(str(p)+'.run.json'),
                'batch_size':r['batch_size'],'calibration_sha256':r['calibration_file_sha256'],
                'engine':r['engine']} for p,r in zip(paths,runs)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',action='append',required=True);p.add_argument('--results',action='append',required=True)
    p.add_argument('--readout',choices=['R0','R1'],default='R0');p.add_argument('--keep-items');p.add_argument('--exclude-items')
    p.add_argument('--filters',default='{}');p.add_argument('--draws',type=int,default=10000);p.add_argument('--output',required=True)
    p.add_argument('--allow-mixed-execution',action='store_true')
    a=p.parse_args()
    if len(a.inputs)!=len(a.results):raise ValueError('Each input requires its matching closed result')
    identity=execution_identity(a.results,a.allow_mixed_execution)
    rows=itertools.chain.from_iterable(read_closed(source,raw) for source,raw in zip(a.inputs,a.results))
    panel=aggregate(rows,a.readout,json.loads(a.filters),ids_from(a.keep_items) if a.keep_items else None,ids_from(a.exclude_items) if a.exclude_items else None)
    result=analyze(panel,a.draws);result['readout']=a.readout
    result['execution_identity']=identity
    result['selection']={'keep_items':a.keep_items,'exclude_items':a.exclude_items,'filters':json.loads(a.filters)}
    paths=[Path(__file__),*map(Path,a.inputs+a.results)]
    if a.keep_items:paths.append(Path(a.keep_items))
    if a.exclude_items:paths.append(Path(a.exclude_items))
    result['file_sha256']={str(path.resolve()):sha(path) for path in paths}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'n_items':result['n_items'],'n_topics':result['n_topics'],'point':result['point'],'H1':result['H1']},ensure_ascii=False))

if __name__=='__main__':main()
