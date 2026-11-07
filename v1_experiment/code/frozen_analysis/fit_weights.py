"""Transfer coefficients use only E1/E2e rows outside the E4 target topic."""
from pathlib import Path
import csv,itertools,json
import numpy as np
from extract_render import OUT
AUX=OUT/'inputs'
def read(p):return json.loads(p.read_text())
def items():
    return {r['item_id']:r['topic'] for r in read(AUX/'e1_semantic64_items.json')}
def fit(model,topic):
    micro=AUX/(model+'__e2e_R0_analysis.json'); weights={}; sides={}; coeff={}; records=[]
    # Return absent predictions as NaN, never as a zero-effect estimate.
    if micro.exists():
        e=read(micro);topics=items();use=[i for i,k in enumerate(e['item_ids']) if topics[k]!=topic]
        delta=np.array(e['item_count_deltas'])[use];cond=list(map(tuple,e['conditions']))
        so=delta[:,cond.index((1,0))];sa=delta[:,cond.index((0,1))]
        for n in (2,4):
            cells=[(o,a) for o,a in cond if o+a==n]
            xx=np.concatenate([o*so+a*sa for o,a in cells]);yy=np.concatenate([delta[:,cond.index((o,a))] for o,a in cells])
            weights[n]=float(xx@yy/(xx@xx)) if xx@xx>0 else np.nan
            records.append(('size',n,'','',weights[n],'',len(use),'E2e'))
            for o in range(n+1):
                a=n-o
                if (o,a) not in cells:
                    sides[n,o]=(weights[n],weights[n]);records.append(('state_size_fallback',n,o,a,weights[n],weights[n],len(use),'cell_absent'));continue
                active=[i for i,k in enumerate((o,a)) if k]
                x=np.stack([o*so,a*sa],axis=1)[:,active];y=delta[:,cond.index((o,a))]
                gram=x.T@x;ridge=0.1*np.trace(gram)/len(active)
                vals=np.full(2,weights[n])
                if ridge>0:vals[active]=np.linalg.solve(gram+ridge*np.eye(len(active)),x.T@y+ridge*weights[n])
                sides[n,o]=tuple(vals);records.append(('state',n,o,a,*vals,len(use),'E2e_ridge_0.1'))
    else:
        for n in (2,4):
            weights[n]=np.nan
            for o in range(n+1):sides[n,o]=(np.nan,np.nan)
    candidates=[AUX/(model+'__'+s+'_R0_analysis.json') for s in ('confirmatory180','holdout200')]
    src=next((p for p in candidates if p.exists()),None)
    if src:
        e=read(src);ii=[i for i,k in enumerate(e['topics']) if k!=topic]
        single=np.array(e['item_singletons'])[ii]
        agg={}
        for mask in range(1,32):
            subset=[k for k in range(5) if mask&(1<<k)]
            if len(subset)<2:continue
            k=(len(subset),sum(j<3 for j in subset))
            values=np.array([e['mobius_by_item'][e['item_ids'][i]][str(mask)] for i in ii])
            a,b=agg.get(k,(0.,0.));agg[k]=(a+values.sum(),b+np.abs(single[:,subset]).mean(axis=1).sum())
        for (n,o),(a,b) in agg.items():
            coeff[n,o]=a/b if b>0 else np.nan
            records.append(('E1_order',n,o,n-o,coeff[n,o],'',len(ii),src.name))
    return weights,sides,coeff,records
def micro_export():
    byid=items(); dest=OUT/'micro';dest.mkdir(exist_ok=True)
    with (dest/'count_curves_by_item.csv').open('w',newline='') as f,(dest/'E1_mobius_by_item.csv').open('w',newline='') as g:
        w=csv.writer(f);w.writerow('model condition item_id topic n_opponents n_allies delta_opposition_logit'.split());v=csv.writer(g);v.writerow('model source item_id topic subset_mask order n_opponents coefficient_opposition_logit'.split())
        for p in sorted(AUX.glob('*__e2?_R0_analysis.json')):
            model,cond=p.name.split('__',1);d=read(p)
            for item,row in zip(d['item_ids'],d['item_count_deltas']):
                for (o,a),value in zip(d['conditions'],row):w.writerow((model,cond.split('_')[0],item,byid[item],o,a,value))
        for p in sorted(AUX.glob('*__*0_R0_analysis.json')):
            if not ('holdout200' in p.name or 'confirmatory180' in p.name):continue
            model=p.name.split('__',1)[0];d=read(p)
            for item,topic in zip(d['item_ids'],d['topics']):
                for m,c in d['mobius_by_item'][item].items():
                    mask=int(m);v.writerow((model,p.name.split('__')[1].split('_')[0],item,topic,mask,mask.bit_count(),(mask&7).bit_count(),c))
    with (dest/'fixed_share.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow('model cohort n_items n_topics ratio start_o start_a end_o end_a difference_logit ci_lo ci_hi'.split())
        keep={x['item_id'] for x in read(AUX/'e1_semantic_remaining44_items.json')}
        for p in sorted(AUX.glob('*__e2e_R0_analysis.json')):
            d=read(p);conditions=list(map(tuple,d['conditions']))
            for label,ids in [('full64',set(byid)),('sensitivity44',keep)]:
                ii=[i for i,k in enumerate(d['item_ids']) if k in ids];t,ix=np.unique([byid[d['item_ids'][i]] for i in ii],return_inverse=True)
                boot=np.random.default_rng(20261105).multinomial(len(t),np.ones(len(t))/len(t),size=20000)[:,ix];delta=np.array(d['item_count_deltas'])[ii]
                for a,b in [((2,1),(6,3)),((1,1),(3,3)),((1,0),(6,0))]:
                    shift=delta[:,conditions.index(b)]-delta[:,conditions.index(a)]
                    ci=np.quantile((boot@shift)/boot.sum(axis=1),[.025,.975])
                    w.writerow([p.name.split('__')[0],label,len(ii),len(t),f'{a[0]}:{a[1]}',*a,*b,shift.mean(),*ci])

