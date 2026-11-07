"""Describe a batch-only pilot comparison; no inherited v2 .05 rejection gate."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from analyze_v3 import read_closed,read_json,sha
V2=Path(__file__).resolve().parents[2]/'voices_nonadditivity_v2_20261030/code'
sys.path.insert(0,str(V2))
try:
    import analyze_v2 as v2
except ModuleNotFoundError:
    import reference_analyze_v2 as v2

def compare(inputs,reference,candidate):
    source=v2.read_rows(inputs);a=list(read_closed(inputs,reference));b=list(read_closed(inputs,candidate))
    runs=[read_json(str(p)+'.run.json') for p in (reference,candidate)]
    for key in ('engine','option_ids','calibration_file_sha256','input_file_sha256','readout'):
        if runs[0][key]!=runs[1][key]:raise ValueError('Pilot changed more than batch size: '+key)
    if [v2.row_id(r) for r in a]!=[v2.row_id(r) for r in b]:raise ValueError('Pilot record order changed')
    for x,y in zip(a,b):
        for key in ('actual_prompt_sha256','actual_prompt_ids_sha256','R1_actual_prompt_sha256','R1_actual_prompt_ids_sha256'):
            if x.get(key)!=y.get(key):raise ValueError('Actual prompt or token IDs changed: '+key)
    result={'n_rows':len(a),'batch_sizes':[r['batch_size'] for r in runs],
      'same_model_template_calibration_and_actual_prompt_tokens':True,'comparison_is_descriptive':True,
      'no_v2_point05_automatic_rejection':True,'readouts':{},'quality':[]}
    for rows in (a,b):
        checks=v2.generated_checks({'physical_rows':rows})
        result['quality'].append({'q_median':float(np.median([r['q'] for r in rows])),
            'q_below_point5_count':sum(r['q']<.5 for r in rows),'generated_checks':checks})
    for readout in ('R0','R1'):
        panels=[v2.prepare(source,rows,readout) for rows in (a,b)]
        if any(p['status']!='complete' for p in panels):raise ValueError('Incomplete numerical pilot')
        comps=[v2.components(p) for p in panels];points=[v2.primary_point(c[2],c[3],c[4]) for c in comps]
        denominators=[float(np.abs(c[2]).mean()) for c in comps]
        x=np.array([r[readout+'_y_AB'] for r in a]);y=np.array([r[readout+'_y_AB'] for r in b]);diff=np.abs(x-y)
        scale=denominators[0];metrics=('w_same','w_diff','ASI')
        changes={k:abs(points[1][k]-points[0][k]) if all(p[k] is not None for p in points) else None for k in metrics}
        result['readouts'][readout]={'reference_mean_absolute_singleton':scale,
          'candidate_mean_absolute_singleton':denominators[1],
          'mean_absolute_y_difference':float(diff.mean()),'mean_absolute_y_difference_over_reference_abs_singleton':float(diff.mean()/scale) if scale>0 else None,
          'max_absolute_y_difference':float(diff.max()),'p95_absolute_y_difference':float(np.percentile(diff,95)),
          'sign_agreement_including_equal_ties':float((np.sign(x)==np.sign(y)).mean()),
          'strict_same_nonzero_direction_fraction':float(((x*y)>0).mean()),
          'reference_ties':int((x==0).sum()),'candidate_ties':int((y==0).sum()),
          'reference_primary':{k:points[0][k] for k in metrics},'candidate_primary':{k:points[1][k] for k in metrics},
          'absolute_primary_changes':changes,
          'mean_absolute_label_slot_averaged_y_difference':float(np.abs(comps[0][1]-comps[1][1]).mean())}
    summaries=[read_json(str(p)+'.summary.json') for p in (reference,candidate)]
    seconds=[s['seconds_this_invocation'] for s in summaries]
    result['runtime']={'seconds':seconds,'reference_over_candidate_speedup':seconds[0]/seconds[1],
       'scope':'measured score invocation, not download or model-load wall time unless engine includes it'}
    result['within_batch_R0_R1_primary_absolute_differences']=[
        {k:abs(result['readouts']['R0'][which][k]-result['readouts']['R1'][which][k])
            if result['readouts']['R0'][which][k] is not None and result['readouts']['R1'][which][k] is not None else None
         for k in ('w_same','w_diff','ASI')} for which in ('reference_primary','candidate_primary')]
    paths=[inputs,reference,candidate,__file__,v2.__file__]
    paths += [str(p)+suffix for p in (reference,candidate) for suffix in ('.run.json','.summary.json')]
    result['file_sha256']={str(Path(p).resolve()):sha(p) for p in paths}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=compare(a.inputs,a.reference,a.candidate)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'batch_sizes':result['batch_sizes'],'runtime':result['runtime'],'readouts':result['readouts'],
      'quality':[{'q':q['q_median'],'format':q['generated_checks']['first_token_format']['value'],
                  'R2':q['generated_checks']['R2_concordance']['value']} for q in result['quality']]}))
