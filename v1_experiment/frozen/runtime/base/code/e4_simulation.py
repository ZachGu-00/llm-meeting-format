"""E4 measured readout caches, P0/PK and coupled sequential-meeting Monte Carlo.

Global cache orientation is support (+1), independent of the target tau.
E1 relationship kernels are oriented toward opposition to the focal agent;
therefore the PK correction changes sign with the focal agent's current stance.
"""
from __future__ import annotations
from collections import defaultdict
import itertools
import math
import numpy as np


def save_cache(path,cache,predictions):
    """Portable non-pickled cache, including observed five-person meetings."""
    keys=sorted(cache['five'])
    with open(path,'xb') as f:
        np.savez_compressed(f,alpha=cache['alpha'],pair=cache['pair'],**predictions,
            five_focal=np.array([k[0] for k in keys],dtype=np.int16),
            five_peers=np.array([k[1] for k in keys],dtype=np.int16).reshape(-1,4),
            five_bits=np.array([k[2] for k in keys],dtype=np.int8).reshape(-1,5),
            five_logits=np.array([cache['five'][k] for k in keys]))


def load_cache(path):
    with np.load(path,allow_pickle=False) as f:
        required={'alpha','pair','measured','five_focal','five_peers','five_bits','five_logits'}
        if not required<=set(f.files):raise ValueError('Cache lacks five-person states; rerun per-topic analysis from closed scores')
        five={(int(j),tuple(map(int,p)),tuple(map(int,b))):float(y)
              for j,p,b,y in zip(f['five_focal'],f['five_peers'],f['five_bits'],f['five_logits'])}
        return {'n':len(f['alpha']),'alpha':f['alpha'],'pair':f['pair'],
                'triple':f['measured'],'five':five,'orientation':'global_support'}

def sigmoid(x):
    x=np.asarray(x,dtype=float)
    return np.exp(-np.logaddexp(0.,-x))

def cache_from_rows(rows,n=15):
    buckets=defaultdict(dict)
    for row in rows:
        if row.get('status')!='ok':raise ValueError('Non-ok measured cache row')
        kind=row['cache_kind'];j=int(row['focal_agent_id']);peers=tuple(row['peer_agent_ids'])
        if tuple(sorted(peers))!=peers or j in peers:raise ValueError('Cache peers must be sorted and exclude focal')
        bits=tuple(int(s==1) for s in (row['focal_stance'],*row['peer_stances']))
        key=(kind,j,peers,bits);render=(row['permutation'],row['label_map'])
        if render in buckets[key]:raise ValueError('Duplicate cache render')
        value=float(row['R0_y_AB'])*(1 if row['positive_letter']=='A' else -1)
        if not math.isfinite(value):raise ValueError('Nonfinite cache logit')
        buckets[key][render]=(value,row['positive_letter'])
    alpha=np.full((n,2),np.nan);pair=np.full((n,n,4),np.nan);triple=np.full((n,n,n,8),np.nan);five={}
    for (kind,j,peers,bits),renders in buckets.items():
        positions=sorted({p for p,l in renders});values=[]
        if len(positions)!=(1 if kind=='baseline' else 2):raise ValueError('Missing cache position')
        for pos in positions:
            if (pos,0) not in renders or (pos,1) not in renders:raise ValueError('Missing cache label')
            a,b=renders[pos,0],renders[pos,1]
            if a[1]==b[1]:raise ValueError('Positive label does not reverse')
            values.append((a[0]+b[0])/2)
        value=float(np.mean(values));index=sum(bit<<(len(bits)-1-k) for k,bit in enumerate(bits))
        if kind=='baseline':alpha[j,bits[0]]=value
        elif kind=='pair':pair[j,peers[0],index]=value
        elif kind=='triple':triple[j,peers[0],peers[1],index]=value
        elif kind=='five':five[(j,peers,bits)]=value
        else:raise ValueError('Unknown cache kind')
    for j in range(n):
        if not np.isfinite(alpha[j]).all():raise ValueError('Missing focal baseline states')
        for k in range(n):
            if k!=j and not np.isfinite(pair[j,k]).all():raise ValueError('Missing directed pair states')
        for k,l in itertools.combinations([v for v in range(n) if v!=j],2):
            if not np.isfinite(triple[j,k,l]).all():raise ValueError('Missing triple cache states')
    return {'alpha':alpha,'pair':pair,'triple':triple,'five':five,'n':n,'orientation':'global_support'}

def predicted_caches(cache,kernels):
    n=cache['n'];p0=np.full_like(cache['triple'],np.nan);pk=np.full_like(p0,np.nan)
    for j in range(n):
        for k,l in itertools.combinations([v for v in range(n) if v!=j],2):
            for sj,sk,sl in itertools.product((0,1),repeat=3):
                index=4*sj+2*sk+sl;alpha=cache['alpha'][j,sj]
                vk=cache['pair'][j,k,2*sj+sk]-alpha;vl=cache['pair'][j,l,2*sj+sl]-alpha
                relation='opponent_opponent' if sk!=sj and sl!=sj else 'ally_ally' if sk==sj and sl==sj else 'opponent_ally'
                if kernels[relation] is None:raise ValueError('Undefined E1 relationship kernel cannot predict E4')
                p0[j,k,l,index]=alpha+vk+vl
                pk[j,k,l,index]=alpha+vk+vl+(1-2*sj)*kernels[relation]*(abs(vk)+abs(vl))/2
    return {'measured':cache['triple'],'P0':p0,'PK':pk}

def single_round(logits,blocks,state,tau=1):
    state=np.asarray(state,dtype=np.int8);values=[]
    for block in blocks:
        for j in block:
            k,l=sorted(v for v in block if v!=j)
            y=logits[j,k,l,4*int(state[j])+2*int(state[k])+int(state[l])]
            if not np.isfinite(y):raise ValueError('Missing observed group/state cache')
            values.append(float(sigmoid(tau*y)))
    if not values:raise ValueError('Empty design')
    return float(np.mean(values))

def single_round_batch(logits,designs,states,tau=1):
    blocks=np.asarray(designs,dtype=int);states=np.asarray(states,dtype=np.int8)
    if states.ndim==1:states=np.broadcast_to(states,(len(blocks),len(states)))
    outcome=np.zeros(len(blocks));di=np.arange(len(blocks))[:,None]
    for pos in range(3):
        other=[p for p in range(3) if p!=pos];j=blocks[:,:,pos];k=blocks[:,:,other[0]];l=blocks[:,:,other[1]]
        index=4*states[di,j]+2*states[di,k]+states[di,l]
        y=logits[j,k,l,index]
        if not np.isfinite(y).all():raise ValueError('Missing batch group/state cache')
        outcome+=sigmoid(tau*y).sum(axis=1)
    return outcome/(blocks.shape[1]*3)

def five_group_outcome(cache,group,state,kernels,tau=1):
    observed=[];p0=[];pk=[]
    for j in group:
        peers=tuple(sorted(v for v in group if v!=j));sj=int(state[j]);bits=(sj,*(int(state[k]) for k in peers))
        observed.append(float(sigmoid(tau*cache['five'][j,peers,bits])))
        alpha=cache['alpha'][j,sj];v={k:cache['pair'][j,k,2*sj+int(state[k])]-alpha for k in peers}
        base=alpha+sum(v.values());correction=0
        for k,l in itertools.combinations(peers,2):
            relation='opponent_opponent' if state[k]!=sj and state[l]!=sj else 'ally_ally' if state[k]==sj and state[l]==sj else 'opponent_ally'
            correction+=(1-2*sj)*kernels[relation]*(abs(v[k])+abs(v[l]))/2
        p0.append(float(sigmoid(tau*base)));pk.append(float(sigmoid(tau*(base+correction))))
    return {'measured':float(np.mean(observed)),'P0':float(np.mean(p0)),'PK_pair_truncated':float(np.mean(pk))}

def design_metrics(observed,p0,pk):
    """Aligned design predictions; pairwise contrast scoring, not raw outcome R2."""
    from scipy.stats import spearmanr
    y,a,b=map(lambda v:np.asarray(v,dtype=float),(observed,p0,pk))
    if y.shape!=a.shape or y.shape!=b.shape or y.ndim!=1 or len(y)<2:raise ValueError('Aligned multiple designs required')
    upper=np.triu_indices(len(y),1);dy=(y[:,None]-y[None,:])[upper]
    result={'n_designs':len(y),'observed_std':float(y.std()),'observed_range':float(np.ptp(y))}
    for name,pred in [('P0',a),('PK',b)]:
        dp=(pred[:,None]-pred[None,:])[upper];den=float((dy*dy).sum());u=dy-dp
        rho=float(spearmanr(y,pred).statistic) if y.std()>0 and pred.std()>0 else None
        nonzero=dy!=0
        result[name]={'R2_delta':1-float((u*u).sum())/den if den>0 else None,'Spearman':rho,
                      'unexplained_fraction_U':float((u*u).sum())/den if den>0 else None,
                      'sign_accuracy_all_contrasts':float((np.sign(dp)==np.sign(dy)).mean()),
                      'sign_accuracy_nonzero_observed':float((np.sign(dp[nonzero])==np.sign(dy[nonzero])).mean()) if nonzero.any() else None,
                      'observed_zero_contrasts':int((~nonzero).sum()),'mean_absolute_residual_U':float(np.abs(u).mean()),
                      'U_rms':float(np.sqrt(np.mean(u*u))),
                      'selection_regret_maximize':float(y.max()-y[np.argmax(pred)]),
                      'selected_gain_over_uniform_design':float(y[np.argmax(pred)]-y.mean()),
                      'selection_regret_minimize':float(y[np.argmin(pred)]-y.min())}
    residual0=dy-(a[:,None]-a[None,:])[upper];residualk=dy-(b[:,None]-b[None,:])[upper]
    den=float((residual0*residual0).sum())
    result['fraction_P0_residual_squared_explained_by_PK']=1-float((residualk*residualk).sum())/den if den>0 else None
    return result

def simulate(logit_caches,designs,initial_states,rounds=6,draws=100000,chunk_size=256,seed=20261030,tau=1):
    """Vectorized over methods, designs and trajectories; synchronous within meeting.

    Common random priorities and per-member draws are indexed by the canonical
    triple, trajectory and round, and shared across all designs and methods.
    Thus each design gets a uniform random meeting order and valid coupling.
    """
    names=list(logit_caches);caches=np.stack([logit_caches[n] for n in names]);n=caches.shape[1]
    if names[0]!='measured':raise ValueError('First cache must be measured for paired prediction residuals')
    designs=[sorted(tuple(sorted(b)) for b in blocks) for blocks in designs]
    if len({len(d) for d in designs})!=1:raise ValueError('Batch requires equal group counts; use separate E4a/E4b batches')
    if any(any(len(set(b))!=3 for b in d) or len(set(d))!=len(d) for d in designs):raise ValueError('Distinct triples required')
    universe=sorted(set(itertools.chain.from_iterable(designs)));uindex={b:i for i,b in enumerate(universe)}
    blocks=np.array(universe,dtype=int);ids=np.array([[uindex[b] for b in d] for d in designs])
    M,D=len(names),len(designs);initial=np.asarray(initial_states,dtype=np.int8)
    if initial.shape==(n,):initial=np.broadcast_to(initial,(D,n))
    if initial.shape!=(D,n) or not np.isin(initial,[0,1]).all():raise ValueError('Initial states must be binary support bits')
    if tau not in (-1,1):raise ValueError('tau must be a global stance sign')
    totals=np.zeros((M,D,rounds));squares=np.zeros_like(totals)
    delta_totals=np.zeros_like(totals);delta_squares=np.zeros_like(totals)
    residual_totals=np.zeros((max(M-1,0),D,rounds));residual_squares=np.zeros_like(residual_totals)
    rng=np.random.default_rng(seed);count=0
    while count<draws:
        B=min(chunk_size,draws-count);state=np.broadcast_to(initial[None,:,None,:],(M,D,B,n)).copy()
        mi=np.arange(M)[:,None,None];di=np.arange(D)[None,:,None];bi=np.arange(B)[None,None,:]
        for r in range(rounds):
            priority=rng.random((B,len(universe)));uniform=rng.random((B,len(universe),3))
            order=np.argsort(priority[:,ids].transpose(1,0,2),axis=2)
            for step in range(ids.shape[1]):
                group_index=ids[np.arange(D)[:,None],order[:,:,step]]
                members=blocks[group_index]
                before=np.take_along_axis(state,members[None,:,:,:],axis=3)
                decisions=[]
                for pos in range(3):
                    others=[p for p in range(3) if p!=pos];j=members[:,:,pos];k=members[:,:,others[0]];l=members[:,:,others[1]]
                    index=4*before[:,:,:,pos]+2*before[:,:,:,others[0]]+before[:,:,:,others[1]]
                    y=caches[mi,j[None,:,:],k[None,:,:],l[None,:,:],index]
                    if not np.isfinite(y).all():raise ValueError('Missing dynamic group-state cache')
                    random=uniform[np.arange(B)[None,:],group_index,pos]
                    decisions.append(random[None,:,:]<sigmoid(y))
                for pos,decision in enumerate(decisions):state[mi,di,bi,members[None,:,:,pos]]=decision
            outcome=state.mean(axis=3)
            if tau==-1:outcome=1-outcome
            totals[:,:,r]+=outcome.sum(axis=2);squares[:,:,r]+=(outcome*outcome).sum(axis=2)
            delta=outcome-outcome[:,[0],:]
            delta_totals[:,:,r]+=delta.sum(axis=2);delta_squares[:,:,r]+=(delta*delta).sum(axis=2)
            residual=delta[[0],:,:]-delta[1:,:,:]
            residual_totals[:,:,r]+=residual.sum(axis=2);residual_squares[:,:,r]+=(residual*residual).sum(axis=2)
        count+=B
    def finish(sums,sq):
        means=sums/draws;var=np.maximum(0,(sq-sums*sums/draws)/max(1,draws-1))
        return {'means':means.tolist(),'Monte_Carlo_standard_errors':np.sqrt(var/draws).tolist()}
    return {'methods':names,'n_designs':D,'n_agents':n,'rounds':rounds,'draws':draws,'chunk_size':chunk_size,'seed':seed,'tau':tau,
      'common_random_numbers':'trajectory/round/canonical-triple/member; shared priorities and Bernoulli draws',
      'within_meeting_update':'simultaneous decisions from pre-meeting states; groups sequential',
      'outcomes':finish(totals,squares),'paired_difference_to_design0':finish(delta_totals,delta_squares),
      'paired_residual_U_measured_minus_other_methods':finish(residual_totals,residual_squares),
      'interval_scope':'Monte Carlo error conditional on caches; topic sampling uncertainty must be computed separately'}
