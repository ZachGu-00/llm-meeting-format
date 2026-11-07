"""Group-aware E4 predictors using only measured alpha/singleton effects.

These rules are not functions of the projected graph alone: the known meeting
composition remains an input. No E4 multi-peer response is used to fit them.
"""
import itertools
import numpy as np
from e4_simulation import sigmoid,single_round,single_round_batch
from run_e4_analysis import initial_state

RULES=('additive','present_mean','side_mean','side_max')


def rule_deltas(values,opponents):
    values=np.asarray(values,dtype=float);opponents=np.asarray(opponents,dtype=bool)
    if values.ndim!=1 or values.shape!=opponents.shape or not np.isfinite(values).all():raise ValueError('Invalid local singleton effects')
    if not len(values):return dict.fromkeys(RULES,0.)
    sides=[values[opponents],values[~opponents]]
    return {'additive':float(values.sum()),'present_mean':float(values.mean()),
            'side_mean':float(sum(side.mean() for side in sides if len(side))),
            'side_max':float(sum(side[np.abs(side).argmax()] for side in sides if len(side)))}


def local_rule_logits(cache,j,peers,state):
    peers=tuple(sorted(peers));sj=int(state[j]);alpha=cache['alpha'][j,sj]
    values=[cache['pair'][j,k,2*sj+int(state[k])]-alpha for k in peers]
    opponent=[int(state[k])!=sj for k in peers]
    return {rule:alpha+delta for rule,delta in rule_deltas(values,opponent).items()}


def triple_rule_caches(cache):
    n=cache['n'];out={rule:np.full_like(cache['triple'],np.nan) for rule in RULES}
    for j in range(n):
        for k,l in itertools.combinations([v for v in range(n) if v!=j],2):
            for sj,sk,sl in itertools.product((0,1),repeat=3):
                state={j:sj,k:sk,l:sl};index=4*sj+2*sk+sl
                for rule,value in local_rule_logits(cache,j,[k,l],state).items():out[rule][j,k,l,index]=value
    return out


def five_rule_outcomes(cache,group,state,tau):
    values={rule:[] for rule in RULES}
    for j in group:
        for rule,y in local_rule_logits(cache,j,[k for k in group if k!=j],state).items():values[rule].append(float(sigmoid(tau*y)))
    return {rule:float(np.mean(y)) for rule,y in values.items()}


def choose_e1_rule(e1):
    scores={rule:e1['point']['rules'][rule]['R2_within_item'] for rule in RULES}
    valid=[rule for rule in RULES if scores[rule] is not None and np.isfinite(scores[rule])]
    best=max(valid,key=lambda rule:scores[rule]) if valid else None
    return {'status':'selected_on_E1_only' if best else 'undefined_E1_rule_scores','rule':best,
            'metric':'R2_within_item over the E1 analysis multi-peer subsets','scores':scores,
            'tie_priority':list(RULES),'selection_uses_E4_outcomes':False,
            'negative_R2_values_retained':True}


def observed_and_primary(result):
    out={}
    out['E4a']={name:np.array([r['outcomes_left_right'][name] for r in result['E4a']['records']]) for name in ('measured','P0','PK')}
    out['E4b']={name:np.array([r['outcomes_mapping_by_system'][name] for r in result['E4b']['configurations']]) for name in ('measured','P0','PK')}
    out['E4c']={name:np.array([[r['five_person_outcomes']['PK_pair_truncated' if name=='PK' else name],r['all_three_person_outcomes'][name]]
                              for r in result['E4c']['records']]) for name in ('measured','P0','PK')}
    return out


def evaluate_topic(cache,result,meta,systems):
    predictions=triple_rule_caches(cache);out={part:{rule:[] for rule in RULES} for part in ('E4a','E4b','E4c')}
    for record in result['E4a']['records']:
        state=initial_state(cache['n'],record['n_target'],record['tau'],record['target_agent_ids'])
        for rule,logits in predictions.items():
            out['E4a'][rule].append([single_round(logits,meta[key],state,record['tau']) for key in ('g6_pasch_left','g6_pasch_right')])
    designs=[]
    for mapping in result['E4b']['point_to_agent_mappings']:
        for system in systems:designs.append([tuple(sorted(mapping[v] for v in block)) for block in system['blocks']])
    for record in result['E4b']['configurations']:
        state=initial_state(cache['n'],record['n_target'],record['tau'])
        for rule,logits in predictions.items():
            out['E4b'][rule].append(single_round_batch(logits,designs,state,record['tau']).reshape(result['E4b']['n_mappings'],len(systems)))
    for record in result['E4c']['records']:
        group=meta['five_person_sets'][record['set_index']];state=np.zeros(cache['n'],dtype=np.int8)
        state[group]=record['global_support_bits'];five=five_rule_outcomes(cache,group,state,record['tau'])
        blocks=list(itertools.combinations(group,3))
        for rule,logits in predictions.items():out['E4c'][rule].append([five[rule],single_round(logits,blocks,state,record['tau'])])
    primary=observed_and_primary(result)
    for part in out:
        out[part]={rule:np.asarray(values) for rule,values in out[part].items()}
        if not np.allclose(out[part]['additive'],primary[part]['P0'],rtol=1e-10,atol=1e-11):
            raise ValueError('Supplement additive rule does not reproduce primary P0: '+part)
        out[part]['PK']=primary[part]['PK'];out[part]['measured']=primary[part]['measured']
    return out


def method_statistics(part,observed,predicted):
    y=np.asarray(observed);pred=np.asarray(predicted)
    if y.shape!=pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():raise ValueError('Unaligned/nonfinite outcomes')
    extra={}
    if part=='E4b':
        if y.ndim!=3:raise ValueError('Expected configuration x mapping x system outcomes')
        indices=np.triu_indices(y.shape[-1],1)
        dy=(y[..., :,None]-y[...,None,:])[...,indices[0],indices[1]].ravel()
        dp=(pred[..., :,None]-pred[...,None,:])[...,indices[0],indices[1]].ravel()
        best=np.take_along_axis(y,np.argmax(pred,axis=-1)[...,None],axis=-1)[...,0]
        worst=np.take_along_axis(y,np.argmin(pred,axis=-1)[...,None],axis=-1)[...,0]
        from scipy.stats import rankdata
        yr=rankdata(y,axis=-1);pr=rankdata(pred,axis=-1);yr-=yr.mean(axis=-1,keepdims=True);pr-=pr.mean(axis=-1,keepdims=True)
        denominator=np.sqrt(np.sum(yr**2,axis=-1)*np.sum(pr**2,axis=-1))
        rho=np.divide(np.sum(yr*pr,axis=-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>0)
        extra={'selection_regret_maximize':float(np.mean(y.max(axis=-1)-best)),
               'selection_regret_minimize':float(np.mean(worst-y.min(axis=-1))),
               'selected_gain_over_uniform':float(np.mean(best-y.mean(axis=-1))),
               'Spearman_mean':float(np.mean(rho)),
               'undefined_mapping_Spearman_count':int((~np.isfinite(rho)).sum()),
               'observed_mean_within_mapping_range':float(np.ptp(y,axis=-1).mean()),
               'observed_mean_within_mapping_std':float(y.std(axis=-1).mean())}
    else:
        if y.ndim!=2 or y.shape[1]!=2:raise ValueError('Expected aligned two-design outcomes')
        dy=y[:,0]-y[:,1];dp=pred[:,0]-pred[:,1]
    residual=dy-dp
    return {'observed_MS':float(np.mean(dy**2)),'error_MS':float(np.mean(residual**2)),
            'absolute_error':float(np.mean(np.abs(residual))),
            'sign_accuracy_all_contrasts':float(np.mean(np.sign(dy)==np.sign(dp))),
            'observed_zero_contrast_count':int((dy==0).sum()),'n_contrasts':len(dy),**extra}


def estimands(stats):
    den=np.asarray(stats['observed_MS']);error=np.asarray(stats['error_MS'])
    fraction=np.divide(error,den,out=np.full(np.broadcast_shapes(error.shape,den.shape),np.nan),where=den>0)
    return {'R2_delta':1-fraction,'contrast_MSE':error,'contrast_RMSE':np.sqrt(error),
            'residual_fraction_relative_to_observed_contrasts':fraction,
            **{k:np.asarray(stats[k]) for k in ('absolute_error','sign_accuracy_all_contrasts','selection_regret_maximize',
                 'selection_regret_minimize','selected_gain_over_uniform','Spearman_mean') if k in stats}}
