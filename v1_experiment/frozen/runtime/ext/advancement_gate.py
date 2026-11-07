"""Recorded resource-advancement rule; never relabel it as an RMS null test."""
import argparse
from datetime import datetime,timezone
import math
from pathlib import Path
import sys
from common import load,sha,verify_bindings,write_new

def saturated(e1):
    value=e1['point']['C'];ci=e1['bootstrap']['intervals']['C']
    return (e1['readout']=='R0' and value is not None and math.isfinite(value) and value<0 and
            ci['undefined_draws']==0 and ci['interval95'] is not None and ci['interval95'][1]<0)

def passes(metric,tolerance=1e-10):
    interval=metric['conditional_on_E1_point_kernel'];value=metric['estimate']
    return (value is not None and math.isfinite(value) and value>tolerance and interval['undefined_draws']==0
            and interval['interval95'] is not None and interval['interval95'][0]>tolerance)

def register(base,extension):
    base,root=Path(base).resolve(),Path(extension).resolve();target=root/'plans/advancement/plan.json'
    if target.exists():
        result=load(target);verify_bindings(result['source_sha256']);return result
    e9_path=root/'plans/e9_full/plan.json';e9=load(e9_path)
    e4_path=Path(e9['e4_predecessor']['score_plan']);e4=load(e4_path)
    sources={str(e9_path):sha(e9_path),str(e4_path):sha(e4_path),str(Path(__file__)):sha(__file__),
        str(root/'ADVANCEMENT_PROTOCOL.md'):sha(root/'ADVANCEMENT_PROTOCOL.md')}
    models=[]
    for model in e4['selected_models']:
        filename='confirmatory180_R0_analysis.json' if model['repo_id'] in ('Qwen/Qwen3.5-4B','LiquidAI/LFM2.5-1.2B-Instruct') else 'holdout200_R0_analysis.json'
        path=base/'results'/model['slug']/'e1'/filename;e1=load(path);verify_bindings(e1['file_sha256'])
        sources[str(path)]=sha(path);sources.update(e1['file_sha256'])
        identity=e1['execution_identity']['critical_identity']
        if identity['repo']!=model['repo_id'] or identity['revision']!=model['revision']:
            raise ValueError('E1 saturation label belongs to another E4 model')
        models.append({'repo_id':model['repo_id'],'slug':model['slug'],'revision':model['revision'],
            'E1_analysis':str(path),'E1_analysis_sha256':sha(path),'E1_saturation_label':saturated(e1),
            'E1_C':e1['point']['C'],'E1_C_interval':e1['bootstrap']['intervals']['C'],
            'E0_tier':model['tier'],'R0_operational':model['R0_operational']})
    result={'schema':'voices_v3_advancement_registration_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'base':str(base),'extension':str(root),'predecessor_plan':str(e9_path),
        'E4_expected':e9['e4_predecessor'],'models':models,'source_sha256':sources,
        'sections':['E4a','E4b'],'metric':'P0_residual_RMS','zero_tolerance':1e-10,
        'criterion':'any independently labelled saturated model has either section with finite topic-bootstrap lower endpoint above numerical zero',
        'PK_advantage_is_required_for_advancement':False,'statistical_H2_success_claimed_by_gate':False,
        'E6_always_required':True,'E7_if_gate_passes':True,'E3_E5':'excluded_by_user',
        'timing':'original E4 is already running; this rule is recorded before inspecting its effect estimates'}
    write_new(target,result);return result

def evaluate(registration,output):
    policy=load(registration);verify_bindings(policy['source_sha256']);base=Path(policy['base'])
    sys.path[:0]=[str(base/'setup'),str(base/'code')]
    from e9_queue import e4_closed
    e9=load(policy['predecessor_plan'])
    if not e4_closed(e9,deep=True):raise RuntimeError('Full frozen six-model/six-topic E4 is not closed')
    terminal_path=Path(policy['E4_expected']['history'])/'final_status.json';terminal=load(terminal_path)
    sources={str(registration):sha(registration),str(terminal_path):sha(terminal_path),**policy['source_sha256']}
    by_repo={}
    for path in terminal['single_round_outputs']['aggregate']:
        aggregate=load(path);verify_bindings(aggregate['file_sha256'])
        if aggregate['n_topics']!=6 or not aggregate['planned_six_topics_complete']:raise ValueError('Incomplete E4 topic panel')
        repo=aggregate['model_identity']['repo']
        if repo in by_repo:raise ValueError('Repeated E4 model aggregate')
        by_repo[repo]=aggregate;sources[str(path)]=sha(path);sources.update(aggregate['file_sha256'])
        artifact=Path(path).parent/aggregate['bootstrap_artifact']['filename']
        if sha(artifact)!=aggregate['bootstrap_artifact']['sha256']:raise ValueError('E4 bootstrap archive changed')
        sources[str(artifact)]=sha(artifact)
    if set(by_repo)!={m['repo_id'] for m in policy['models']}:raise ValueError('Missing selected E4 model')
    results=[]
    for model in policy['models']:
        aggregate=by_repo[model['repo_id']]
        if aggregate['model_identity']['revision']!=model['revision']:raise ValueError('E4 revision changed')
        sections={part:aggregate['single_round'][part]['metrics'][policy['metric']] for part in policy['sections']}
        gate_sections=[part for part,metric in sections.items() if passes(metric,policy['zero_tolerance'])]
        results.append({**model,'section_metrics':sections,'passing_sections':gate_sections,
            'advances_E7':bool(model['E1_saturation_label'] and gate_sections),
            'topic_sufficient_statistics':{part:aggregate['single_round'][part]['topic_sufficient_statistics'] for part in policy['sections']}})
    advance=any(x['advances_E7'] for x in results)
    result={'schema':'voices_v3_advancement_result_v1','status':'complete_verified','full_E4_verified':True,
        'models':results,'E7_advance':advance,'E6_scope':'v3 validation scope retained regardless of this feasibility gate',
        'E7_status':'authorized_by_v3_gate' if advance else 'not_advanced_under_v3_gate',
        'interpretation':'resource decision on deterministic cache effects; not a null test, practical-utility threshold or claim that H2/PK is confirmed',
        'source_sha256':sources}
    write_new(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--extension');p.add_argument('--plan');p.add_argument('--output')
    a=p.parse_args()
    r=evaluate(a.plan,a.output) if a.plan else register(a.base,a.extension)
    print({'status':r.get('status','registered'),'models':len(r['models']),'E7_advance':r.get('E7_advance')})
