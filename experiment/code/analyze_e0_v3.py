"""V3 E0 admission: operational readout checks, not desired scientific effects."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import reference_analyze_v2 as ref

ROOT=Path(__file__).resolve().parents[1]
LIMITS={'q_median':.70,'first_token_format':.90,'R2_concordance':.85,
        'slot_over_mean_abs_singleton':1.0,'ASI_halfwidth':.20,
        'saturation_fraction':.40,'primary_readout_gap':.25,'sign_deadzone':.10}


def flag(value,threshold,lower=True):
    valid=value is not None and math.isfinite(value)
    return {'value':value,'threshold':threshold,'comparison':'>=' if lower else '<=',
            'status':'incomplete' if not valid else 'pass' if (value>=threshold if lower else value<=threshold) else 'failed'}


def directions(left,right):
    exact=lambda v:None if v is None else 'positive' if v>0 else 'negative' if v<0 else 'zero'
    if left is None or right is None:
        status='undefined'
    elif abs(left)<=.10 or abs(right)<=.10:
        status='indeterminate_near_zero'
    else:
        status='consistent' if left*right>0 else 'clear_opposite'
    return {'R0':left,'R1':right,'R0_exact_sign':exact(left),'R1_exact_sign':exact(right),'status':status}


def admission(operational,r1_valid,direction,gaps,n_items):
    state=ref.overall(operational)
    large_gap=bool(gaps) and max(gaps.values())>.25
    opposite=any(x['status']=='clear_opposite' for x in direction.values())
    if state=='incomplete' or not r1_valid:
        tier='incomplete';primary=None
    elif state=='pass':
        primary='R1' if large_gap or opposite else 'R0'
        tier='R1_primary_R0_operational' if primary=='R1' else 'R0_primary'
    else:
        tier='R1_only';primary='R1'
    return {'tier':tier,'primary_readout':primary,'R0_operational':state,
            'R1_candidate_scoring_valid':r1_valid,'large_readout_gap':large_gap,
            'clear_nontrivial_opposite_sign':opposite,
            'development100_complete':n_items==100,
            'final_qualification':tier if n_items==100 else 'pilot_only',
            'R0_E4_primary_candidate':n_items==100 and tier=='R0_primary',
            'R0_E4_requires_readout_disagreement_flag':n_items==100 and tier=='R1_primary_R0_operational',
            'no_effect_direction_or_positive_singleton_requirement':True}


def analyze(inputs,r0,r1=None,draws=10000):
    original=ref.analyze(inputs,r0,r1,draws=draws)
    if original.get('n_items') is None:
        return {'schema':'voices_v3_E0_v1','qualification':{'final_qualification':'incomplete'},'original_v2_analysis':original}
    if set(original['subset_masks'])!=set(ref.SCREEN):
        raise ValueError('V3 E0 requires exactly the seven specified development subsets')
    panel=ref.prepare(inputs,r0)
    r1rows=r1 if r1 is not None else [r for r in r0 if r.get('R1_y_AB') is not None]
    p1=ref.prepare(inputs,r1rows,'R1')
    r1valid=p1['status']=='complete' and bool(np.isfinite(p1['oriented']).all())
    ops={name:flag(original['gates'][name].get('value'),threshold) for name,threshold in
         [('option_mass_median',.70),('first_token_format',.90),('R2_concordance',.85)]}
    slots,_,singles,_,_=ref.components(panel)
    mean_abs=float(np.abs(singles).mean())
    sd=float(slots[:,1:,:].std(axis=2,ddof=0).mean())
    slot_ratio=sd/mean_abs if mean_abs>0 else None
    gaps=original['gates']['R0_R1_primary_agreement'].get('value')
    if not isinstance(gaps,dict):gaps=None
    left=original['R0']['point'];right=original['R1_primary']
    ds={key:directions(left.get(key),right.get(key)) for key in ('w_same','w_diff')}
    diagnostics={'slot_effect_over_mean_abs_singleton':flag(slot_ratio,1.,False),
        'ASI_precision':flag(original['gates']['ASI_precision'].get('value'),.20,False),
        'empty_saturation':flag(original['gates']['empty_saturation'].get('value'),.40,False),
        'readout_primary_gap':flag(max(gaps.values()) if gaps else None,.25,False),
        'weight_directions':ds,
        'singleton_split_correlation':{'value':original['gates']['split_half_vA_Pearson'].get('value'),
            'interpretation':'structurally_identical_prompt_multisets_not_independent_reliability',
            'used_for_admission':False},
        'mean_absolute_opponent_singleton_effect':mean_abs,
        'nonpositive_or_unstable_signed_weight_denominators_are_not_exclusion':True}
    return {'schema':'voices_v3_E0_v1','n_items':original['n_items'],
        'n_topics':original['n_topic_clusters'],'n_rows':len(inputs),'bootstrap_draws':draws,
        'operational_gates':ops,'applicability_diagnostics':diagnostics,
        'qualification':admission(ops,r1valid,ds,gaps,original['n_items']),
        'thresholds':LIMITS,'strict_original_gates':original['gates'],
        'original_v2_full32_development_flag_not_used_for_v3_E0':True,
        'original_v2_analysis':original}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','r0','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--r1',type=Path);p.add_argument('--draws',type=int,default=10000)
    args=p.parse_args()
    r0,a0=ref.read_scored(args.r0,args.inputs)
    r1,a1=(ref.read_scored(args.r1,args.inputs) if args.r1 else (None,None))
    if a0['status']!='complete' or (a1 and a1['status']!='complete'):
        raise ValueError('E0 analysis requires a completed input-bound run')
    if a1:
        for key in ('engine','calibration_file_sha256','option_ids','batch_size'):
            if a0['run'][key]!=a1['run'][key]:raise ValueError('R0/R1 execution identity differs')
    if a0['run']['readout'] not in {'R0','R0_R1'} or (a1 and a1['run']['readout']!='R1'):
        raise ValueError('Incorrect readout files')
    result=analyze(ref.read_rows(args.inputs),r0,r1,args.draws)
    result['execution_integrity']={'R0':a0,'R1':a1 if a1 else 'embedded_in_R0_file'}
    sources=[Path(__file__),Path(ref.__file__),ROOT/'execution_protocol.txt',args.inputs,args.r0]
    if args.r1:sources.append(args.r1)
    result['file_sha256']={str(f.resolve()):ref.sha(f) for f in sources}
    if args.draws!=10000:
        result['qualification']['final_qualification']='nonstandard_draw_count_diagnostic_only'
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'n_items':result['n_items'],'qualification':result['qualification'],'operational_gates':result['operational_gates']}))


if __name__=='__main__':main()
