"""Measured one-step outcomes and transferred pairwise predictions. No LLM inference."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,hashlib,itertools,json,os,shutil
from pathlib import Path
import numpy as np
from extract_render import OUT,WORK,CACHE
from fit_weights import fit,read,micro_export
AUX=OUT/'inputs';DEST=OUT/'analysis';SHARDS=WORK/'analysis_shards'
METHODS=['observed','P0','present_mean','size_E2','state_E2','E1_order2','E1_order234_identified','side_mean','side_max','static_projection_E2']
N=len(METHODS)
def sig(x):return np.exp(-np.logaddexp(0.,-x))
def states(n):
    return ((np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1).astype(int)
def encode(bits):return bits @ (1<<np.arange(bits.shape[1]-1,-1,-1))
def blocksums(table,groups,members,st):
    """Returns sum over appearances per agent and method. Last method stays on logit scale."""
    out=np.zeros((len(st),len(members),N));counts=np.zeros(len(members),int)
    offsets={j:i for i,j in enumerate(members)}
    for group in groups:
        g=tuple(sorted(group));loc=[offsets[j] for j in g];code=encode(st[:,loc])
        out[:,loc,:]+=table[g][code];counts[loc]+=1
    return out,counts
def finish(sums,counts,person=True):
    node=np.divide(sums,counts[None,:,None])
    node[:,:,-1]=sig(node[:,:,-1])
    return node.mean(axis=1) if person else np.average(node,axis=1,weights=counts)
def make_table(c,meta,model,topic):
    weights,sides,coeff,records=fit(model,topic);tab={};four={tuple(k):v for k,v in zip(c['five_keys'],c['five_values'])}
    for g in itertools.chain(itertools.combinations(range(15),3),[tuple(sorted(s)) for s in meta['five_person_sets']]):
        st=states(len(g));n=len(g)-1;matrix=np.empty((len(st),len(g),N))
        for jpos,j in enumerate(g):
            pp=[i for i in range(len(g)) if i!=jpos];peers=[g[i] for i in pp];sj=st[:,jpos];sp=st[:,pp];opp=sp!=sj[:,None];num=opp.sum(1);alpha=c['alpha'][j,sj]
            v=np.stack([c['pair'][j,k,2*sj+sp[:,a]]-alpha for a,k in enumerate(peers)],axis=1)
            idx=encode(np.column_stack((sj,sp)))
            obs=c['triple'][j,*peers,idx] if n==2 else np.array([four[(j,*peers,int(x))] for x in idx])
            so=(v*opp).sum(1);sa=(v*~opp).sum(1);base=alpha+v.sum(1)
            w=np.array([sides[n,int(o)] for o in num]);scaled=alpha+weights[n]*v.sum(1)
            second=np.zeros(len(st));higher=np.zeros(len(st))
            for t in range(2,n+1):
                for subset in itertools.combinations(range(n),t):
                    no=opp[:,subset].sum(1)
                    rel=np.array([coeff.get((t,int(o)),0.) for o in no])
                    correction=(1-2*sj)*rel*np.abs(v[:,subset]).mean(1)
                    if t==2:second+=correction
                    else:higher+=correction
            side_mean=alpha+so/np.maximum(num,1)+sa/np.maximum(n-num,1)
            side_max=alpha.copy()
            for mask in (opp,~opp):
                index=np.argmax(np.where(mask,np.abs(v),-1.),axis=1)
                side_max+=np.where(mask.any(axis=1),v[np.arange(len(st)),index],0.)
            yy=np.stack((obs,base,alpha+v.mean(1),scaled,alpha+w[:,0]*so+w[:,1]*sa,base+second,base+second+higher,side_mean,side_max,scaled),axis=1)
            if not coeff:yy[:,5:7]=np.nan
            matrix[:,jpos,:]=yy
        matrix[:,:,:-1]=sig(matrix[:,:,:-1]);tab[g]=matrix
    return tab,records
def projection(blocks):
    return Counter(p for g in blocks for p in itertools.combinations(sorted(g),2))
def stash_metrics(result,family,dy):
    y=dy[:,0];q=float(np.mean(y*y))
    for i,key in enumerate(METHODS[1:],1):
        p=dy[:,i];err=float(np.mean((y-p)**2))
        result.append([family,key,len(y),q,err])
    result.append([family,'zero',len(y),q,q])
def add_table(f,name,fields):
    fp=gzip.open(f/(name+'.csv.gz'),'wt',newline='',compresslevel=2);w=csv.writer(fp);w.writerow(fields);return fp,w
def task(info):
    stage,model,tid,readout=info;tag='__'.join(info);here=SHARDS/tag;here.mkdir(exist_ok=True,parents=True)
    old=tid.startswith('V3_');md=read(AUX/('e4_cache_manifest.json' if old else 'M1_manifest.json'));meta=next(x for x in md['topics'] if x['topic_id']==tid)
    common=list(info);prefix=['stage','model','topic_id','readout']
    with np.load(CACHE/(tag+'.npz')) as c:table,coef=make_table(c,meta,model,meta['topic'])
    results=[];wfile,w=add_table(here,'weights',prefix+'source n_peers_or_order n_opp n_ally weight_opp_or_scalar weight_ally n_training_items training_rule'.split());w.writerows([common+list(v) for v in coef]);wfile.close()
    # Focal target sign + and - are complementary in E4c. Keep the 32 unique support states.
    fc,w=add_table(here,'five_vs_three',prefix+['set_id','agents','support_bits','five_observed','triangles_observed']+['delta_'+m for m in METHODS])
    dy=[]
    for gi,ss in enumerate(meta['five_person_sets']):
        S=tuple(sorted(ss));st=states(5);tri=list(itertools.combinations(S,3));assert projection([S]*3)==projection(tri)
        a,na=blocksums(table,[S],S,st);b,nb=blocksums(table,tri,S,st);x=finish(a,na);y=finish(b,nb);dd=x-y;dy.extend(dd)
        for row,v in enumerate(dd):w.writerow(common+[gi,' '.join(map(str,S)),format(row,'05b'),x[row,0],y[row,0],*v])
    fc.close();stash_metrics(results,'E4c',np.array(dy))
    # G6 original registered initial configurations, including target sign.
    fa,w=add_table(here,'six_regroup',prefix+['target_stance','initial_support_bits','agents','left_obs','right_obs']+['delta_'+m for m in METHODS])
    agents=tuple(meta['g6_agent_ids']);left=meta['g6_pasch_left'];right=meta['g6_pasch_right'];assert projection(left)==projection(right)
    st=states(6);sa,na=blocksums(table,left,agents,st);sb,nb=blocksums(table,right,agents,st);a=finish(sa,na);b=finish(sb,nb);dy=[]
    for tau in (-1,1):
        for i,s in enumerate(st):
            if sum(s==int(tau==1)) not in (2,3):continue
            d=tau*(a[i]-b[i]);dy.append(d);w.writerow(common+[tau,format(i,'06b'),' '.join(map(str,agents)),a[i,0] if tau==1 else 1-a[i,0],b[i,0] if tau==1 else 1-b[i,0],*d])
    fa.close();stash_metrics(results,'E4a',np.array(dy))
    # STS: reproduce the archived mapping and use the exact variance identity for all pairs.
    ss=read(AUX/'sts15_all80.json')['systems'];templates=np.array([s['blocks'] for s in ss])
    rng=np.random.default_rng(int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+tid).encode()).hexdigest()[:16],16))
    perms=np.stack([rng.permutation(15) for i in range(20)]);blocks=np.sort(perms[:,templates],axis=-1)
    tg=list(itertools.combinations(range(15),3));lookup=np.zeros((15,15,15),int)
    for k,g in enumerate(tg):lookup[g]=k
    where=lookup[blocks[...,0],blocks[...,1],blocks[...,2]]
    tabs=np.stack([table[g][:,:,:-1].mean(1) for g in tg]);gi=np.array(tg)
    fb,w=add_table(here,'sts_outcomes',prefix+['target_stance','target_count','mapping','system']+METHODS[:-1])
    observed_ms=[];errors={m:[] for m in METHODS[1:]}
    for m,tau in itertools.product((3,5,7),(-1,1)):
        state=np.full(15,int(tau==-1));state[:m]=int(tau==1)
        codes=encode(state[gi]);values=tabs[np.arange(len(gi)),codes][where].mean(axis=2)
        if tau==-1:values=1-values
        truth=values[:,:,0];factor=2*80/79
        observed_ms.append(factor*truth.var(axis=1).mean())
        for k,method in enumerate(METHODS[1:-1],1):errors[method].append(factor*(truth-values[:,:,k]).var(axis=1).mean())
        # Equal-size blocks and fixed pair multiplicities imply zero contrast for the static projection.
        errors[METHODS[-1]].append(observed_ms[-1] if np.isfinite(coef_sample(table)) else np.nan)
        for j in range(20):
            for k in range(80):w.writerow(common+[tau,m,j,k,*values[j,k]])
    fb.close();den=float(np.mean(observed_ms));cuts=6*20*3160
    results.extend([['E4b',method,cuts,den,np.mean(err)] for method,err in errors.items()]);results.append(['E4b','zero',cuts,den,den])
    # Matched six: 2 copies of all ten S-triples plus 5 copies of one Pasch system.
    # Both changes redistribute 60 dyadic exposure events. Person weights are held equal.
    p1=[(0,2,4),(0,3,5),(1,2,5),(1,3,4)];p2=[(0,2,5),(0,3,4),(1,2,4),(1,3,5)]
    t=list(itertools.combinations(range(5),3));big=tuple(range(5))
    assert projection(t*2+p1*5)==projection(t*2+p2*5)==projection([big]*6+p1*5)
    fm,w=add_table(here,'matched_six',prefix+['set_id','ordered_agents','support_bits','base_obs','regroup_obs','resize_obs','delta_regroup_incidence','delta_resize_incidence']+['seat_'+k for k in METHODS]+['size_'+k for k in METHODS])
    seat=[];size=[];incseat=[];incsize=[];st=states(6)
    for gi,S in enumerate(meta['five_person_sets']):
        for outsider in sorted(set(range(15))-set(S)):
            seed=int(hashlib.sha256(f'voices_matched6_v1|{tid}|{gi}|{outsider}'.encode()).hexdigest()[:16],16)
            ag=tuple(np.random.default_rng(seed).permutation(sorted(S)))+(outsider,)
            def sums(gs):
                return blocksums(table,[tuple(ag[j] for j in g) for g in gs],ag,st)
            tr,ct=sums(t);bigsum,cb=sums([big]);ps,cp=sums(p1);pt,cq=sums(p2)
            def three(person):
                return [finish(2*tr+5*ps,2*ct+5*cp,person),finish(2*tr+5*pt,2*ct+5*cq,person),finish(6*bigsum+5*ps,6*cb+5*cp,person)]
            base,regroup,resize=three(True);v,vr,vs=three(False);ds=regroup-base;dz=resize-base;dsi=vr-v;dzi=vs-v
            seat.extend(ds);size.extend(dz);incseat.extend(dsi);incsize.extend(dzi)
            for i in range(64):w.writerow(common+[gi,' '.join(map(str,ag)),format(i,'06b'),base[i,0],regroup[i,0],resize[i,0],dsi[i,0],dzi[i,0],*ds[i],*dz[i]])
    fm.close()
    for fam,a in [('matched_seat',seat),('matched_size',size),('matched_seat_incidence',incseat),('matched_size_incidence',incsize)]:stash_metrics(results,fam,np.asarray(a))
    return [common+[meta['topic']]+r for r in results]
def coef_sample(table):
    return next(iter(table.values()))[0,0,-1]
def main():
    DEST.mkdir(exist_ok=True);SHARDS.mkdir(parents=True,exist_ok=True)
    with (OUT/'coverage.csv').open() as f:keys=[tuple(r[k] for k in ('stage','model','topic_id','readout')) for r in csv.DictReader(f)]
    micro_export()
    with (DEST/'topic_metrics.csv').open('w',newline='') as f,ProcessPoolExecutor(max_workers=6) as pool:
        w=csv.writer(f);w.writerow('stage model topic_id readout topic family method n_contrasts obs_ms error_ms'.split())
        for rows in pool.map(task,keys,chunksize=1):w.writerows(rows)
    for name in ('weights','five_vs_three','six_regroup','sts_outcomes','matched_six'):
        with gzip.open(DEST/(name+'.csv.gz'),'wb',compresslevel=3) as out:
            for i,key in enumerate(keys):
                with gzip.open(SHARDS/('__'.join(key))/(name+'.csv.gz'),'rb') as inp:
                    if i:inp.readline()
                    shutil.copyfileobj(inp,out,1048576)
    print('GROUP_DONE',len(keys),flush=True)
if __name__=='__main__':main()

