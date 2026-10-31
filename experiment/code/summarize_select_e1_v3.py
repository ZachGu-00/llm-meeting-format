"""Verified E1 panel and explicit, predeclared absolute-C E4 selection.

Without --freeze this only reports results and eligibility. Frozen selection
requires every requested model's assigned E1 analyses to be locally verified.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from analyze_v3 import holm
import summarize_e0_panel_v3 as e0

ROOT=Path(__file__).resolve().parents[1]
MANDATORY=('Qwen/Qwen3.5-4B','LiquidAI/LFM2.5-1.2B-Instruct')
HASH_PREFIX='voices_v3_E4_absC_selection_v1|'


def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)


def architecture_class(config):
    text=config.get('text_config',config);types=set(text.get('layer_types',[]))
    if 'linear_attention' in types and 'full_attention' in types:kind='linear_and_full_attention'
    elif 'conv' in types and any('attention' in t for t in types):kind='convolution_and_attention'
    elif ('mamba' in types or text.get('mamba_d_state',0)>0 or text.get('mamba_num_heads',0)>0) and text.get('num_attention_heads',0)>0:
        kind='state_space_and_attention'
    elif types and all(t in {'attention','full_attention','sliding_attention'} for t in types):kind='attention'
    elif not types and text.get('model_type') in {'llama','granite','phi3','ministral3','smollm3','gemma4_text'}:kind='attention'
    else:kind='unclassified_'+str(text.get('model_type',config.get('model_type','unknown')))
    return {'class':kind,'text_model_type':text.get('model_type'),
            'layer_type_counts':dict(Counter(text.get('layer_types',[]))),
            'architectures':config.get('architectures',[]),
            'definition':'Text mixing mechanism from saved official config; full/sliding attention share one class. Qwen linear/full layers are a hybrid class.'}


class Verifier:
    def __init__(self,root,remote_root):self.root=Path(root);self.remote_root=remote_root;self.cache={};self.verified={}
    def path(self,value):return e0.local_path(value,self.root,self.remote_root)
    def digest(self,path):
        path=Path(path);stamp=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
        if stamp not in self.cache:self.cache[stamp]=e0.sha(path)
        return self.cache[stamp]
    def check(self,path,expected):
        path=Path(path)
        if self.digest(path)!=expected:raise ValueError('Source hash mismatch: '+str(path))
        self.verified[str(path.resolve())]=expected


def config_identity(root,repo,engine,verifier):
    slug=repo.replace('/','--');expected=engine['parent_loader']['metadata_sha256']['config.json']
    candidates=[root/'provenance/model_metadata_snapshot/models'/slug/'config.json',
                root/'models'/slug/'config.json',root/'provenance/architecture_configs'/(slug+'.json')]
    for path in candidates:
        if path.is_file() and verifier.digest(path)==expected:
            verifier.check(path,expected)
            return {**architecture_class(e0.read(path)),'family':repo.split('/')[0].casefold(),
                    'family_definition':'Hugging Face publisher namespace, a stable diversity tie-break proxy',
                    'config_path':str(path),'config_sha256':expected,'config_matches_execution':True}
    raise ValueError('No saved official config matches the execution config SHA for '+repo)


def item_ids(path):
    value=e0.read(path)
    if isinstance(value,dict):value=value['item_ids']
    return {item if isinstance(item,str) else item['item_id'] for item in value}


def expected_jobs(root,repo,r0_operational):
    fresh=repo in MANDATORY;jobs=[]
    def add(name,readout,items,role):jobs.append({'name':name,'readout':readout,'items':root/'data'/items,'role':role})
    if r0_operational:
        add('holdout200_R0_analysis','R0','e1_holdout200_items.json','descriptive' if fresh else 'primary')
        if fresh:add('confirmatory180_R0_analysis','R0','e1_remaining180_items.json','primary')
        add('semantic64_R0_analysis','R0','e1_semantic64_items.json','descriptive' if fresh else 'matched_R0')
        if fresh:add('matched44_R0_analysis','R0','e1_semantic_remaining44_items.json','matched_R0')
    add('semantic64_R1_analysis','R1','e1_semantic64_items.json','descriptive' if fresh else 'matched_R1')
    if fresh:add('matched44_R1_analysis','R1','e1_semantic_remaining44_items.json','matched_R1')
    return jobs


def verify_e1(path,job,repo,calibration,verifier):
    analysis=e0.read(path);expected_ids=item_ids(job['items'])
    if (analysis.get('readout')!=job['readout'] or analysis.get('full32') is not True or
        set(analysis.get('masks',[]))!=set(range(32)) or analysis.get('n_items')!=len(expected_ids) or
        set(analysis.get('item_ids',[]))!=expected_ids or len(analysis.get('item_ids',[]))!=len(expected_ids) or
        analysis.get('n_physical_rows')!=622*len(expected_ids)):
        raise ValueError('E1 readout/item partition/full32 panel differs from the planned analysis')
    if analysis.get('bootstrap',{}).get('draws')!=10000 or analysis['bootstrap'].get('unit')!='topic':
        raise ValueError('E1 analysis lacks the planned 10000 topic bootstrap')
    selection=analysis.get('selection',{})
    if selection.get('exclude_items') or selection.get('filters')!={}:raise ValueError('Unexpected E1 analysis exclusions/filters')
    if not analysis.get('file_sha256'):raise ValueError('E1 analysis has no source hashes')
    sources={}
    for source,digest in analysis['file_sha256'].items():
        local=verifier.path(source);verifier.check(local,digest);sources[str(local.resolve())]=digest
    raw=path.parent/('holdout200_R0.jsonl' if job['readout']=='R0' else 'semantic64_R1.jsonl')
    planned=verifier.root/'data'/('e1_holdout200_prompts.jsonl' if job['readout']=='R0' else 'e1_semantic64_prompts.jsonl')
    for required in (raw,planned,verifier.root/'code/analyze_v3.py'):
        if str(required.resolve()) not in sources:raise ValueError('E1 analysis does not bind its raw/input/analyzer source')
    run_path=Path(str(raw)+'.run.json');summary_path=Path(str(raw)+'.summary.json')
    run=e0.read(run_path);summary=e0.read(summary_path);n=124400 if job['readout']=='R0' else 39808
    if (summary.get('status')!='complete' or summary.get('exact_id_and_count_equality') is not True or
        summary['n_input']!=n or summary['n_output']!=n or run['n_input']!=n or
        summary['output_bytes']!=raw.stat().st_size or summary['output_sha256']!=verifier.digest(raw) or
        summary['run_file_sha256']!=verifier.digest(run_path) or run['input_file_sha256']!=verifier.digest(planned) or
        run['readout']!=job['readout'] or any(summary.get(k)!=v for k,v in run.items())):
        raise ValueError('E1 raw run is not a closed, matching planned score file')
    lock=calibration['value']
    if (run['engine']!=lock['engine'] or run['option_ids']!=lock['option_ids'] or
        run['calibration_file_sha256']!=calibration['sha256'] or run['batch_size']!=lock['batch_size'] or
        run['engine']['parent_loader']['repo_id']!=repo):raise ValueError('E1 engine/options/calibration differ from selected E0')
    executions=analysis['execution_identity']['runs']
    if (len(executions)!=1 or verifier.path(executions[0]['path']).resolve()!=raw.resolve() or
        executions[0]['run_sha256']!=verifier.digest(run_path) or executions[0]['engine']!=run['engine'] or
        executions[0]['batch_size']!=run['batch_size'] or executions[0]['calibration_sha256']!=run['calibration_file_sha256']):
        raise ValueError('E1 analysis execution receipt differs from its raw score file')
    for bound in (path,run_path,summary_path,job['items']):verifier.verified[str(bound.resolve())]=verifier.digest(bound)
    return analysis


def compact_analysis(analysis,path):
    return {'path':str(path),'sha256':e0.sha(path),'readout':analysis['readout'],'n_items':analysis['n_items'],
            'n_topics':analysis['n_topics'],'C':analysis['point']['C'],
            'C_interval95':analysis['bootstrap']['intervals']['C']['interval95'],
            'C_undefined_bootstrap_draws':analysis['bootstrap']['intervals']['C']['undefined_draws'],
            'rules':analysis['point']['rules'],'relation_kernels':analysis['point']['relation_kernels'],
            'H1':analysis['H1'],'execution_identity':analysis['execution_identity']['critical_identity']}


def direction(value):return 'undefined' if not finite(value) else 'positive' if value>0 else 'negative' if value<0 else 'zero'


def summarize_model(root,repo,e0row,verifier):
    row={'repo_id':repo,'E0_status':e0row.get('status'),'E0_qualification':e0row.get('qualification'),
         'E1_status':'pending_E0','E1_complete':False,'E4_eligible':False,'primary':None,
         'analyses':{},'missing_or_invalid_analyses':[]}
    if (not e0row.get('closed_development100') or e0row.get('integrity',{}).get('status')!='verified_local'):
        row['E0_integrity']=e0row.get('integrity');return row
    statepath=root/'results'/repo.replace('/','--')/'e0/e0_driver_state.json';state=e0.read(statepath)
    qualification=e0row['qualification'];row['E0_tier']=qualification['final_qualification']
    if qualification!=state['qualification'] or not qualification['R1_candidate_scoring_valid']:
        raise ValueError('E0 qualification/state inconsistent or R1 scoring invalid')
    row['R0_operational']=qualification['R0_operational']=='pass'
    row['R0_readout_disagreement_flag']=row['E0_tier']=='R1_primary_R0_operational'
    row['conditional_option_mass_admission']=bool(qualification.get('conditional_option_mass_admission',False))
    calpath=verifier.path(state['selected_calibration']);calibration={'value':e0.read(calpath),'sha256':verifier.digest(calpath)}
    engine=calibration['value']['engine'];row['revision']=engine['parent_loader']['revision']
    row['structure']=config_identity(root,repo,engine,verifier)
    row['E0_state_sha256']=verifier.digest(statepath);row['E0_analysis_sha256']=e0row['analysis_sha256']
    row['E0_protocol_sha256']=state.get('execution_protocol_sha256')
    row['E0_source_snapshot_substitutions']=e0row['integrity'].get('source_snapshot_substitutions',[])
    row['calibration_sha256']=calibration['sha256'];verifier.verified[str(calpath.resolve())]=calibration['sha256']
    verifier.verified[str(statepath.resolve())]=row['E0_state_sha256']
    analyses={}
    for job in expected_jobs(root,repo,row['R0_operational']):
        path=root/'results'/repo.replace('/','--')/'e1'/(job['name']+'.json')
        try:
            analysis=verify_e1(path,job,repo,calibration,verifier)
            row['analyses'][job['name']]=compact_analysis(analysis,path);analyses[job['role']]=row['analyses'][job['name']]
        except (OSError,ValueError,KeyError,TypeError) as error:
            row['missing_or_invalid_analyses'].append({'name':job['name'],'reason':str(error)})
    row['E1_complete']=not row['missing_or_invalid_analyses']
    row['E1_status']='complete_verified' if row['E1_complete'] else 'incomplete_or_not_synced'
    row['primary']=analyses.get('primary')
    if 'matched_R0' in analyses and 'matched_R1' in analyses:
        x,y=analyses['matched_R0'],analyses['matched_R1'];c0,c1=x['C'],y['C']
        row['matched_readout_diagnostic']={'n_items':x['n_items'],'R0_C':c0,'R1_C':c1,
            'R0_interval95':x['C_interval95'],'R1_interval95':y['C_interval95'],
            'R0_direction':direction(c0),'R1_direction':direction(c1),
            'same_exact_direction':direction(c0)==direction(c1) if finite(c0) and finite(c1) else None,
            'absolute_C_difference':abs(c0-c1) if finite(c0) and finite(c1) else None,
            'diagnostic_only_not_a_selection_filter':True}
    elif not row['R0_operational']:
        row['R1_only_exploratory']=analyses.get('matched_R1')
    c=row['primary']['C'] if row['primary'] else None
    row['E4_eligible']=row['E1_complete'] and row['R0_operational'] and finite(c)
    row['E4_ineligibility_reason']=None if row['E4_eligible'] else 'R1_only' if not row['R0_operational'] else 'E1_incomplete' if not row['E1_complete'] else 'C_undefined'
    return row


def select_six(rows):
    """Call only after explicit freeze; target quantiles are fixed before removal."""
    eligible=[r for r in rows if r['E4_eligible']]
    if len(eligible)<6:raise ValueError('Fewer than six eligible models; expand the panel')
    byrepo={r['repo_id']:r for r in eligible};selected=[byrepo[r] for r in MANDATORY if r in byrepo]
    unavailable=[r for r in MANDATORY if r not in byrepo];pool=[r for r in eligible if r['repo_id'] not in MANDATORY]
    slots=6-len(selected);probabilities=np.linspace(0,1,slots)
    targets=np.quantile([abs(r['primary']['C']) for r in pool],probabilities,method='linear')
    steps=[{'repo_id':r['repo_id'],'reason':'mandatory_when_R0_operational_and_C_finite','signed_C':r['primary']['C']} for r in selected]
    for probability,target in zip(probabilities,targets):
        architectures=Counter(r['structure']['class'] for r in selected);families=Counter(r['structure']['family'] for r in selected)
        def key(row):return (abs(abs(row['primary']['C'])-target),architectures[row['structure']['class']],
                             families[row['structure']['family']],hashlib.sha256((HASH_PREFIX+row['repo_id']).encode()).hexdigest())
        chosen=min(pool,key=key);criterion=key(chosen)
        steps.append({'repo_id':chosen['repo_id'],'reason':'nearest_absolute_C_quantile','quantile_probability':float(probability),
            'target_absolute_C':float(target),'signed_C':chosen['primary']['C'],'absolute_distance':float(criterion[0]),
            'already_selected_architecture_count':criterion[1],'already_selected_family_count':criterion[2],'tie_hash':criterion[3]})
        selected.append(chosen);pool.remove(chosen)
    return {'selected_repos':[r['repo_id'] for r in selected],'steps':steps,'unavailable_mandatory_models':unavailable,
        'quantile_probabilities':probabilities.tolist(),'quantile_targets':targets.tolist(),
        'quantile_pool_excludes_mandatory':True,'quantile_targets_recomputed_after_selection':False,
        'generalized_fallback':'r=6-number_of_available_mandatory; r equally spaced absolute-C quantiles',
        'selected_R0_readout_disagreement_flags':{r['repo_id']:r['R0_readout_disagreement_flag'] for r in selected},
        'selected_conditional_option_mass_flags':{r['repo_id']:r.get('conditional_option_mass_admission',False) for r in selected}}


def h1_family(rows):
    byrepo={r['repo_id']:r for r in rows};tests={};pvalues={}
    for repo,field,hypothesis in [(MANDATORY[0],'C_left_tail_bootstrap_p','C < 0'),
                                 (MANDATORY[1],'C_equivalence_bootstrap_p','abs(C) < 0.1')]:
        row=byrepo.get(repo);primary=row.get('primary') if row else None
        complete=bool(row and row['E1_complete'] and primary)
        p=primary['H1'].get(field) if complete else None
        if not finite(p):p=None
        tests[repo]={'hypothesis':hypothesis,'prespecified':True,'fresh_R0_items_required':180,
                     'raw_p':p,'status':'resolved' if p is not None else 'unresolved',
                     'C':primary['C'] if complete else None}
        pvalues[repo]=p
    adjusted=holm(pvalues)
    for repo,value in adjusted.items():tests[repo]['Holm_adjusted_p']=value
    return {'family_size':2,'tests':tests,'method':'prespecified Qwen/LFM centered topic-bootstrap tests with Holm correction',
            'other_models':'exploratory C estimates only; no same-data type assignment followed by a confirmatory type test'}


def build(root=ROOT,remote_root='/workspace/voices_nonadditivity_v3',models=None,freeze=False):
    root=Path(root).resolve();base=e0.build(root,remote_root);all_rows={r['repo_id']:r for r in base['models']}
    requested=list(models) if models else [r['repo_id'] for r in e0.read(root/'setup/download_queue.json')]
    if len(set(requested))!=len(requested) or not requested:raise ValueError('A nonempty unique requested model panel is required')
    scope_additions=[];terminal_unavailable=[]
    for repo in MANDATORY:
        state=all_rows.get(repo,{})
        terminal=(state.get('closed_development100') and state.get('integrity',{}).get('status')=='verified_local'
                  and state.get('qualification',{}).get('R0_operational')=='failed')
        if terminal:terminal_unavailable.append(repo)
        elif repo not in requested:
            # Operational mandatory models cannot be filtered away; unknown E0
            # remains pending instead of being silently treated as unavailable.
            requested.append(repo);scope_additions.append(repo)
    verifier=Verifier(root,remote_root);rows=[]
    for repo in requested:
        try:rows.append(summarize_model(root,repo,all_rows.get(repo,{}),verifier))
        except (OSError,ValueError,KeyError,TypeError) as error:
            rows.append({'repo_id':repo,'E1_status':'integrity_or_sync_error','E1_complete':False,'E4_eligible':False,'primary':None,'error':str(error)})
    complete=all(r['E1_complete'] for r in rows);eligible=sum(r['E4_eligible'] for r in rows)
    selection={'status':'not_requested','performed':False,'eligible_count':eligible,'required_count':6}
    if freeze:
        if not complete:selection.update(status='blocked_incomplete_requested_E1')
        elif eligible<6:selection.update(status='needs_more_eligible_models')
        else:selection.update(status='frozen',performed=True,**select_six(rows))
    protocol=root/'execution_protocol.txt';queue=root/'setup/download_queue.json'
    return {'schema':'voices_v3_E1_panel_and_E4_selection_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'E1 results and E4 model selection only; does not imply E2 or E4 experiments completed',
        'requested_models':requested,'all_requested_E1_complete':complete,'models':rows,
        'unrequested_mandatory_hypotheses_retained_as_unresolved':[repo for repo in MANDATORY if repo not in requested],
        'operational_or_pending_mandatory_models_added_to_scope':scope_additions,
        'terminal_R0_unavailable_mandatory_models':terminal_unavailable,
        'counts':{'requested':len(rows),'E1_verified_complete':sum(r['E1_complete'] for r in rows),'E4_eligible':eligible},
        'H1':h1_family(rows),'E4_selection':selection,
        'selection_policy':'Mandatory Qwen4/LFM when R0 operational and finite fresh180 C. Remaining r slots use fixed remaining-pool |C| quantiles linspace(0,1,r), without replacement; distance, fewer represented architecture, fewer publisher family, fixed SHA256 tie-break.',
        'R1_primary_R0_operational_models_eligible_but_flagged':True,'R1_only_models_ineligible':True,
        'signed_C_preserved':True,'readout_disagreement_not_hidden_or_used_as_selection_veto':True,
        'protocol_sha256':e0.sha(protocol),'queue_sha256':e0.sha(queue),'script_sha256':e0.sha(__file__),
        'E0_summarizer_sha256':base['summarizer_sha256'],'E0_protocol_versions':base['protocol_versions'],
        'verified_source_sha256':verifier.verified}


def fmt(value):return '—' if value is None else f'{value:.4g}' if isinstance(value,(float,int)) else str(value).replace('|','/')


def markdown(report):
    counts=report['counts'];lines=['# E1 面板与 E4 选模','',
        f"已完整校验 E1：{counts['E1_verified_complete']}/{counts['requested']}；E4 可用模型：{counts['E4_eligible']}。",
        f"选模状态：{report['E4_selection']['status']}。本报告不代表 E2 或 E4 实验已完成。",'',
        '| 模型 | 结构 | E0 读数层级 | 主 C / 95% CI | 主样本 | 匹配 R0 / R1 C | E1 状态 | E4 可用 |',
        '|---|---|---|---|---:|---|---|---|']
    for row in report['models']:
        primary=row.get('primary') or {};ci=primary.get('C_interval95');diagnostic=row.get('matched_readout_diagnostic',{})
        value=fmt(primary.get('C'))+(' ['+', '.join(fmt(v) for v in ci)+']' if ci else '')
        tier=row.get('E0_tier')
        if tier and row.get('conditional_option_mass_admission'):tier+=' [conditional-q]'
        cells=[row['repo_id'],row.get('structure',{}).get('class'),tier,value,primary.get('n_items'),
               fmt(diagnostic.get('R0_C'))+' / '+fmt(diagnostic.get('R1_C')),row['E1_status'],'是' if row['E4_eligible'] else row.get('E4_ineligibility_reason','未完成')]
        lines.append('| '+' | '.join(fmt(c) for c in cells)+' |')
    lines+=['','Qwen4/LFM 的主 C 来自未见的 180 条，匹配读数诊断来自 44 条；其他模型对应 200/64。'
            'R1-primary 且 R0 操作门通过的模型保留资格并带标记；R1-only 不进入 E4。', '',
            '仅两个事先提出的 H1 假设做 Holm 校正；其他模型的 C 及类型解释属于探索性结果。', '',
            '| 预设模型 | 假设 | 原 p | Holm p | 状态 |','|---|---|---:|---:|---|']
    for repo,test in report['H1']['tests'].items():lines.append('| '+' | '.join(fmt(x) for x in (repo,test['hypothesis'],test['raw_p'],test['Holm_adjusted_p'],test['status']))+' |')
    if report['E4_selection']['performed']:
        lines+=['','E4 冻结名单：','']+[f"- {repo}" for repo in report['E4_selection']['selected_repos']]
    else:lines+=['','尚未冻结 E4 名单；只有显式 `--freeze` 且全部请求的 E1 已闭合、至少六个模型可用时才执行选模。']
    lines+=['','结构依据已保存且与运行 SHA 一致的官方 config；Qwen3.5 的 linear/full attention 混合层不会误标成纯注意力。',
            '未完成、未同步、未定义 C 和运行完整性问题详列于同名 JSON；不会据此伪造完成状态。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--remote-root',default='/workspace/voices_nonadditivity_v3');p.add_argument('--models',nargs='+')
    p.add_argument('--output-prefix',type=Path,required=True);p.add_argument('--freeze',action='store_true')
    args=p.parse_args();json_path=Path(str(args.output_prefix)+'.json');md_path=Path(str(args.output_prefix)+'.md')
    frozen_path=Path(str(args.output_prefix)+'_E4_selection.json')
    if json_path.exists() or md_path.exists() or (args.freeze and frozen_path.exists()):raise FileExistsError('Refusing to overwrite panel/frozen-selection artifacts')
    report=build(args.root,args.remote_root,args.models,args.freeze);json_path.parent.mkdir(parents=True,exist_ok=True)
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    md_path.write_text(markdown(report),encoding='utf-8')
    if report['E4_selection']['performed']:
        frozen={'schema':'voices_v3_frozen_E4_model_selection_v1','status':'frozen',
                'created_utc':report['created_utc'],**report['E4_selection'],
                'requested_models':report['requested_models'],'selection_policy':report['selection_policy'],
                'panel_path':str(json_path),'panel_sha256':e0.sha(json_path),'protocol_sha256':report['protocol_sha256'],
                'selector_sha256':report['script_sha256'],'all_requested_E1_complete':True}
        with frozen_path.open('x',encoding='utf-8') as f:json.dump(frozen,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'counts':report['counts'],'selection_status':report['E4_selection']['status'],'panel':str(json_path)}))
    if args.freeze and not report['E4_selection']['performed']:raise SystemExit(2)


if __name__=='__main__':main()
