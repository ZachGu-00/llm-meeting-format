"""Single-round E4 supplement: three prespecified, unfitted singleton rules."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np

from analyze_v3 import read_json,sha
from e4_simulation import load_cache
from e4_rule_baselines_v3 import RULES,choose_e1_rule,evaluate_topic,method_statistics,estimands

ROOT=Path(__file__).resolve().parents[1]
NOTE=ROOT/'provenance/E4_pairwise_rule_supplement_pre_E4_20261031.txt'
OLD_MODELS={'Qwen/Qwen3.5-4B','LiquidAI/LFM2.5-1.2B-Instruct'}


def interval(point,draws):
    draws=np.asarray(draws);invalid=int((~np.isfinite(draws)).sum())
    return {'estimate':float(point) if np.isfinite(point) else None,
            'interval95':None if invalid else np.quantile(draws,[.025,.975]).tolist(),
            'undefined_draws':invalid,'undefined_draws_not_dropped':True}


def paired_improvements(baseline,pk):
    b,p=estimands(baseline),estimands(pk)
    result={'contrast_MSE_improvement':b['contrast_MSE']-p['contrast_MSE'],
            'contrast_RMSE_improvement':b['contrast_RMSE']-p['contrast_RMSE'],
            'R2_delta_improvement':p['R2_delta']-b['R2_delta'],
            'sign_accuracy_improvement':p['sign_accuracy_all_contrasts']-b['sign_accuracy_all_contrasts']}
    for metric in ('selection_regret_maximize','selection_regret_minimize'):
        if metric in b:result[metric+'_improvement']=b[metric]-p[metric]
    for metric in ('selected_gain_over_uniform','Spearman_mean'):
        if metric in b:result[metric+'_improvement']=p[metric]-b[metric]
    return result


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def weighted_topic_means(weights,values):
    values=np.asarray(values);invalid=~np.isfinite(values)
    result=weights@np.where(invalid,0.,values)/weights.shape[1]
    if invalid.any():result[(weights[:,invalid]>0).any(axis=1)]=np.nan
    return result


def summarize_topics(per_topic,topic_weights,selected_rule):
    output={};draws={}
    for part in ('E4a','E4b','E4c'):
        point={};samples={};reports={}
        for method in (*RULES,'PK'):
            records=[topic[part][method] for topic in per_topic]
            keys=[k for k in records[0] if not k.endswith('_count') and k!='n_contrasts']
            matrix={key:np.array([r[key] for r in records]) for key in keys}
            point[method]={key:values.mean() for key,values in matrix.items()}
            samples[method]={key:weighted_topic_means(topic_weights,values) for key,values in matrix.items()}
            pm,bm=estimands(point[method]),estimands(samples[method])
            reports[method]={metric:interval(value,bm[metric]) for metric,value in pm.items()}
            for metric,value in bm.items():draws[f'{part}_{method}_{metric}']=value
        comparisons={}
        for rule in RULES:
            p=paired_improvements(point[rule],point['PK']);b=paired_improvements(samples[rule],samples['PK'])
            comparisons[rule]={metric:interval(value,b[metric]) for metric,value in p.items()}
        output[part]={'methods':reports,'PK_improvement_over_each_rule':comparisons,
                      'positive_improvement_means_PK_better':True,
                      'E1_selected_rule':selected_rule,
                      'PK_improvement_over_E1_selected_rule':comparisons.get(selected_rule),
                      'selected_rule_is_alias_not_an_independent_fifth_rule':True,
                      'PK_scope':'pair-truncated' if part=='E4c' else 'main E1 relation-kernel prediction'}
    return output,draws


def build(topic_paths,e1_path,manifest_path,sts_path,draws=10000,seed=20261031):
    if draws<2:raise ValueError('At least two bootstrap draws required')
    e1=read_json(e1_path);repo=e1['execution_identity']['critical_identity']['repo']
    if e1.get('readout')!='R0' or e1.get('n_items')!=(180 if repo in OLD_MODELS else 200):
        raise ValueError('Choose the primary fresh180/200 R0 E1 analysis, matching the primary E4 kernel source')
    # This decision is made before inspecting any E4 response values.
    choice=choose_e1_rule(e1)
    source_paths=[Path(e1_path),Path(manifest_path),Path(sts_path),Path(__file__),NOTE,
                  Path(__file__).with_name('e4_rule_baselines_v3.py'),Path(__file__).with_name('e4_simulation.py'),
                  Path(__file__).with_name('analyze_v3.py')]
    source_hashes={str(p.resolve()):sha(p) for p in source_paths}
    manifest={t['topic_id']:t for t in read_json(manifest_path)['topics']};systems=read_json(sts_path)['systems']
    topics=[(Path(path),read_json(path)) for path in topic_paths];topics.sort(key=lambda p:p[1]['topic_id'])
    ids=[r['topic_id'] for _,r in topics];names=[r['topic'] for _,r in topics]
    if len(ids)<2 or len(set(ids))!=len(ids) or len(set(names))!=len(names):raise ValueError('At least two distinct E4 topics required')
    if set(names)&set(e1['topics']):raise ValueError('E1/E4 topics overlap')
    statistics=[];artifacts={};identities=[]
    for path,result in topics:
        if result['E1_execution_identity']!=e1['execution_identity']:raise ValueError('Different E1 kernel/source execution identity')
        if result['execution_identity']['critical_identity']!=e1['execution_identity']['critical_identity']:raise ValueError('E1/E4 model identity mismatch')
        for required in (e1_path,manifest_path,sts_path):
            if sha(required) not in result['input_hashes'].values():raise ValueError('Primary source provenance changed: '+str(required))
        for execution in result['execution_identity']['runs']:
            identities.append((execution['engine'],execution['batch_size'],execution['calibration_sha256']))
        cache_path=path.parent/result['cache_artifact']['filename']
        if sha(cache_path)!=result['cache_artifact']['sha256']:raise ValueError('Primary cache artifact changed')
        outcomes=evaluate_topic(load_cache(cache_path),result,manifest[result['topic_id']],systems)
        topic={part:{name:method_statistics(part,values['measured'],pred) for name,pred in values.items() if name!='measured'} for part,values in outcomes.items()}
        statistics.append(topic)
        for part,values in outcomes.items():
            for rule in RULES:artifacts[result['topic_id']+'_'+part+'_'+rule]=values[rule]
        source_hashes[str(path.resolve())]=sha(path);source_hashes[str(cache_path.resolve())]=sha(cache_path)
    if any(identity!=identities[0] for identity in identities):raise ValueError('Different E4 execution policies across topics')
    weights=np.random.default_rng(seed).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=draws)
    aggregated,draw_arrays=summarize_topics(statistics,weights,choice['rule'])
    artifacts.update(draw_arrays);artifacts['E4_topic_weights']=weights
    result={'schema':'voices_v3_E4_single_round_pairwise_rule_supplement_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'repo_id':repo,'topic_ids':ids,'topics':names,'n_topics':len(ids),'planned_six_topics_complete':len(ids)==6,
        'primary_P0_PK_U_definitions_unchanged':True,'E1_best_rule':choice,'E1_analysis_sha256':sha(e1_path),
        'supplement_provenance_sha256':sha(NOTE),'E4_fit_parameters':0,'new_LLM_calls':0,'dynamic_rules_added':False,
        'rule_definitions':{'additive':'alpha + sum(v)', 'present_mean':'alpha + mean(all present v)',
            'side_mean':'alpha + sum of mean(v) over each nonempty prescribed stance side',
            'side_max':'alpha + sum of signed maximum-absolute v over each nonempty prescribed stance side; exact ties use ascending fixed peer ID'},
        'orientation':'Local alpha/v are global-support logits. All four delta rules are odd under common sign reversal, so direct global evaluation equals conversion through focal-opposition orientation. Side membership is prescribed stance relative to the current focal, never sign(v).',
        'group_sizes':'Three-person meetings use two peer effects; five-person meetings use four. Baseline alpha is added once.',
        'single_round':aggregated,'per_topic_sufficient_statistics':finite_json(dict(zip(ids,statistics))),
        'bootstrap':{'draws':draws,'seed':seed,'unit':'topic','same_topic_draws_for_all_methods':True,
                     'all_mappings_states_and_tau_reversals_kept_inside_topics':True,
                     'ratios_recomputed_from_equal_topic_mean_squared_contrast_and_error':True,
                     'scope':'Conditional on measured caches, the fitted E1 PK kernels and the E1-selected rule. E1 kernel/rule-selection uncertainty is not propagated in this supplementary comparison.',
                     'pointwise_95_intervals_not_simultaneous':True},
        'small_topic_panel_caveat':'Six independent topics give approximate intervals. Mappings, states and design pairs do not increase the independent topic sample size.',
        'claim_boundary':'Projection-only functions cannot distinguish equal projections. Group-aware predictors supplied H and alpha/v can nevertheless predict regrouping effects through nonlinear aggregation. Beating additive P0 is not a proof that all pairwise-data-informed algorithms fail; beating these rules supports only superiority over this prespecified finite family.',
        'file_sha256':source_hashes}
    return result,artifacts


def cell(metric):
    if metric['estimate'] is None:return '未定义'
    ci=metric['interval95'];value=f"{metric['estimate']:.4g}"
    return value+(' ['+', '.join(f'{x:.4g}' for x in ci)+']' if ci is not None else ' [区间未定义]')


def markdown(report):
    lines=['# E4 零参数聚合规则补充分析','',report['repo_id'], '',
           f"议题数：{report['n_topics']}。E1 预选规则：{report['E1_best_rule']['rule']}。不新增 LLM 调用，不使用 E4 多人读数拟合参数。",'',
           '| 实验 | 规则 | R²Δ | PK 相对该规则的配对 MSE 改进 |','|---|---|---:|---:|']
    for part,values in report['single_round'].items():
        for name in (*RULES,'PK'):
            diff=values['PK_improvement_over_each_rule'].get(name,{}).get('contrast_MSE_improvement')
            lines.append('| '+part+' | '+name+' | '+cell(values['methods'][name]['R2_delta'])+' | '+(cell(diff) if diff else '—')+' |')
    lines+=['','正的配对改进表示 PK 更好；负值不裁剪。E4c 的 PK 仍是已声明的 pair-truncated 版本。',
            '区间按议题共同重采样，条件于已估计的 E1 核及已选规则；这张补充表不传播 E1 参数或规则选择不确定性。',
            'P0、PK、U 的主定义和主检验保持不变。单独优于加性 P0 不能证明所有使用两两读数、同时知道分组的算法都不足。'
            '如果均值规则解释了某模型的结果，应按该机制报告，不强行归因于不可约的多人信息。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--topics',action='append',required=True)
    p.add_argument('--e1-analysis',required=True);p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--cache-manifest',default=str(ROOT/'data/e4_cache_manifest.json'))
    p.add_argument('--sts',default=str(ROOT/'data/e4_graphs/sts15_all80.json'));p.add_argument('--draws',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261031);args=p.parse_args()
    paths=[Path(str(args.output_prefix)+suffix) for suffix in ('.json','.md','_outcomes_and_bootstrap.npz')]
    if any(path.exists() for path in paths):raise FileExistsError('Refusing to overwrite supplementary artifacts')
    result,arrays=build(args.topics,args.e1_analysis,args.cache_manifest,args.sts,args.draws,args.seed)
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    with paths[2].open('xb') as f:np.savez_compressed(f,**arrays)
    result['outcomes_and_bootstrap_artifact']={'filename':paths[2].name,'sha256':sha(paths[2])}
    with paths[0].open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    with paths[1].open('x',encoding='utf-8') as f:f.write(markdown(result))
    print(json.dumps({'repo_id':result['repo_id'],'n_topics':result['n_topics'],'E1_selected_rule':result['E1_best_rule']['rule'],'output':str(paths[0])}))


if __name__=='__main__':main()
