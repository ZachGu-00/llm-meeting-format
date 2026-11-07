"""Paired E1/E2 topic-bootstrap diagnostics on the same 64 items (+ fresh44).

Named-peer C is kept distinct from prefix-count contrasts. All conditions share
topic resamples; differences are computed inside each draw, never by subtracting
independently estimated confidence intervals.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from analyze_v3 import PAIRS,C_MASKS,read_json,sufficient
from summarize_select_e1_v3 import Verifier,MANDATORY,item_ids

ROOT=Path(__file__).resolve().parents[1]


def closed_analysis(path,verifier,expected_input,expected_rows):
    analysis=read_json(path)
    if not analysis.get('file_sha256'):raise ValueError('Analysis has no source hashes: '+str(path))
    sources={}
    for source,digest in analysis['file_sha256'].items():
        local=verifier.path(source);verifier.check(local,digest);sources[str(local.resolve())]=digest
    if str(expected_input.resolve()) not in sources:raise ValueError('Analysis does not bind its fixed input file')
    runs=analysis['execution_identity']['runs']
    if len(runs)!=1:raise ValueError('Paired diagnostics require one frozen execution per analysis')
    identity=runs[0];raw=verifier.path(identity['path']);run_path=Path(str(raw)+'.run.json');summary_path=Path(str(raw)+'.summary.json')
    if str(raw.resolve()) not in sources:raise ValueError('Analysis does not bind its raw output')
    run=read_json(run_path);summary=read_json(summary_path)
    if (summary.get('status')!='complete' or summary.get('exact_id_and_count_equality') is not True or
        summary['n_input']!=summary['n_output'] or summary['n_input']!=run['n_input'] or run['n_input']!=expected_rows or
        summary['output_sha256']!=verifier.digest(raw) or summary['output_bytes']!=raw.stat().st_size or
        summary['run_file_sha256']!=verifier.digest(run_path) or run['input_file_sha256']!=verifier.digest(expected_input) or
        run['readout']!='R0' or any(summary.get(k)!=v for k,v in run.items())):
        raise ValueError('Raw score file is not a closed matching R0 run')
    if (identity['run_sha256']!=verifier.digest(run_path) or identity['engine']!=run['engine'] or
        identity['batch_size']!=run['batch_size'] or identity['calibration_sha256']!=run['calibration_file_sha256']):
        raise ValueError('Analysis execution identity differs from its closed raw run')
    if (analysis.get('readout','R0')!='R0' or analysis.get('bootstrap',{}).get('draws')!=10000 or
        analysis['bootstrap'].get('unit')!='topic'):
        raise ValueError('Expected R0 analysis with the planned 10000-draw bootstrap')
    for file in (path,run_path,summary_path):verifier.verified[str(Path(file).resolve())]=verifier.digest(file)
    return analysis,(run['engine'],run['batch_size'],run['calibration_file_sha256'],run['option_ids'])


def reconstruct_named(analysis,ids,topics):
    original=analysis['item_ids'];lookup={item:i for i,item in enumerate(original)}
    if len(lookup)!=len(original) or set(original)!=set(ids):raise ValueError('Named analysis item set differs from semantic64')
    if any(analysis['topics'][lookup[item]]!=topic for item,topic in zip(ids,topics)):raise ValueError('Item topic changed across E1/E2')
    masks=analysis['masks'];y=np.full((len(ids),32),np.nan)
    for i,item in enumerate(ids):
        terms=analysis['mobius_by_item'][item]
        for mask in masks:
            subsets=[s for s in range(32) if s&mask==s]
            if any(str(s) not in terms for s in subsets):raise ValueError('Cannot reconstruct an observed cell from saved Mobius terms')
            y[i,mask]=sum(terms[str(s)] for s in subsets)
    panel={'item_ids':ids,'topics':topics,'masks':masks,'y':y}
    computed=sufficient(panel);order=[lookup[item] for item in ids]
    num=np.array(analysis['item_C_numerators'])[order];den=np.array(analysis['item_C_denominators'])[order]
    if not np.allclose(num,computed['C_num'],rtol=1e-10,atol=1e-11) or not np.allclose(den,computed['C_den'],rtol=1e-10,atol=1e-11):
        raise ValueError('Saved C components disagree with the saved Mobius/singleton data')
    v=np.array(analysis['item_singletons'])[order]
    if not np.allclose(v,computed['v'],rtol=1e-10,atol=1e-11):raise ValueError('Saved singleton data disagree with Mobius reconstruction')
    estimate=float(num.sum()/den.sum()) if den.sum()>0 else None
    recorded=analysis['point']['C']
    if (estimate is None)!=(recorded is None) or estimate is not None and not np.isclose(estimate,recorded,rtol=1e-10,atol=1e-11):
        raise ValueError('Reconstructed C does not reproduce the analysis point estimate')
    return {**panel,'C_num':num,'C_den':den,'kernels':computed['kernels']}


class PairedTopics:
    def __init__(self,topics,draws,seed):
        labels=np.asarray(topics);self.topics=sorted(set(topics));self.labels=labels;self.draws=draws
        self.groups=[np.flatnonzero(labels==topic) for topic in self.topics]
        self.weights=np.random.default_rng(seed).multinomial(len(self.topics),np.full(len(self.topics),1/len(self.topics)),size=draws)
        self.sample_sizes=self.weights@np.array([len(g) for g in self.groups])
    def summed(self,values):
        values=np.asarray(values);grouped=np.stack([values[g].sum(axis=0) for g in self.groups])
        return self.weights@grouped
    def mean(self,values):
        values=np.asarray(values);den=self.sample_sizes.reshape((-1,)+(1,)*(values.ndim-1))
        return values.mean(axis=0),self.summed(values)/den
    def ratio(self,numerator,denominator,absolute_aggregate_denominator=False):
        num=float(np.sum(numerator));den=float(np.sum(denominator));bn=self.summed(numerator);bd=self.summed(denominator)
        if absolute_aggregate_denominator:den=abs(den);bd=np.abs(bd)
        point=num/den if den>0 else np.nan
        return point,np.divide(bn,bd,out=np.full(self.draws,np.nan),where=bd>0)


def summarize_metric(metric):
    point,draws=metric;draws=np.asarray(draws);undefined=int((~np.isfinite(draws)).sum())
    interval=None if undefined else np.quantile(draws,[.025,.975]).tolist()
    return {'estimate':float(point) if np.isfinite(point) else None,'interval95':interval,
            'undefined_draws':undefined,'undefined_draws_not_dropped':True,
            'interval_entirely_positive':bool(interval and interval[0]>0),
            'interval_entirely_negative':bool(interval and interval[1]<0)}


def difference(left,right):return left[0]-right[0],left[1]-right[1]


def named_metrics(panel,context):
    metrics={'C':context.ratio(panel['C_num'],panel['C_den'])}
    for name,(num,den) in panel['kernels'].items():metrics['kernel_'+name]=context.ratio(num,den)
    masks=set(panel['masks']);y=panel['y'];alpha=y[:,0]
    for no,na in itertools.product(range(4),range(3)):
        matched=[m for m in masks if (m&7).bit_count()==no and (m&24).bit_count()==na]
        if matched:metrics[f'count_delta_{no}_{na}']=context.mean(y[:,matched].mean(axis=1)-alpha)
        transitions=[(mask,mask|(1<<ally)) for mask in masks if (mask&7).bit_count()==no and (mask&24).bit_count()==na
                     for ally in (3,4) if not mask>>ally&1 and mask|(1<<ally) in masks]
        if transitions:
            change=np.mean([y[:,b]-y[:,a] for a,b in transitions],axis=0)
            metrics[f'add_ally_{no}_{na}_to_{na+1}']=context.mean(change)
    return metrics


def prefix_panel(panel):
    conditions=list(itertools.product(range(4),range(3)))
    masks=[((1<<n)-1)|(((1<<m)-1)<<3) for n,m in conditions]
    return {'conditions':conditions,'delta':panel['y'][:,masks]-panel['y'][:,[0]],
            'scope':'Same representative first-n opponent and first-m ally roles as E2d/e in the shared 0..3 by 0..2 grid'}


def power_from_curves(conditions,curves,max_opponents=None):
    indices=[i for i,(n,m) in enumerate(conditions) if m==0 and n>0 and (max_opponents is None or n<=max_opponents)]
    x=np.log(np.array([conditions[i][0] for i in indices],dtype=float));values=np.atleast_2d(curves)[:,indices]
    slopes=np.full(len(values),np.nan);r2=np.full(len(values),np.nan)
    if len(x)<2:return slopes,r2
    good=np.isfinite(values).all(axis=1)&(values>0).all(axis=1)
    logy=np.log(values[good]);xc=x-x.mean();slope=(logy@xc)/np.sum(xc**2);slopes[good]=slope
    centered=logy-logy.mean(axis=1,keepdims=True);den=np.sum(centered**2,axis=1)
    error=np.sum((centered-slope[:,None]*xc)**2,axis=1)
    r2[good]=np.divide(den-error,den,out=np.full(len(den),np.nan),where=den>0)
    return slopes,r2


def count_metrics(panel,context):
    conditions=[tuple(c) for c in panel['conditions']];values=np.asarray(panel['delta'])
    point,draws=context.mean(values);metrics={f'count_delta_{n}_{m}':(point[i],draws[:,i]) for i,(n,m) in enumerate(conditions)}
    for i,(n,m) in enumerate(conditions):
        if (n,m+1) in conditions:metrics[f'add_ally_{n}_{m}_to_{m+1}']=context.mean(values[:,conditions.index((n,m+1))]-values[:,i])
    if {(2,0),(0,2),(1,1),(1,0)}<=set(conditions):
        numerator=values[:,conditions.index((2,0))]+values[:,conditions.index((0,2))]-2*values[:,conditions.index((1,1))]
        metrics['count_grouping_contrast']=context.ratio(numerator,values[:,conditions.index((1,0))],True)
    for suffix,limit in [('native_range',None),('common_1_to_3',3)]:
        t,r2=power_from_curves(conditions,np.vstack([point,draws]),limit)
        metrics['power_t_'+suffix]=(t[0],t[1:]);metrics['power_log_R2_'+suffix]=(r2[0],r2[1:])
    return metrics


def subset_named(panel,indices):
    return {**panel,'y':panel['y'][indices],'C_num':panel['C_num'][indices],'C_den':panel['C_den'][indices],
            'kernels':{k:(n[indices],d[indices]) for k,(n,d) in panel['kernels'].items()}}


def compare_cohort(named,counts,ids,topics,keep,draws,seed):
    indices=np.array([i for i,item in enumerate(ids) if item in keep]);selected_ids=[ids[i] for i in indices]
    selected_topics=[topics[i] for i in indices]
    if len(indices)!=len(keep):raise ValueError('Requested cohort contains missing item IDs')
    context=PairedTopics(selected_topics,draws,seed)
    panels={name:subset_named(panel,indices) for name,panel in named.items()}
    raw={name:named_metrics(panel,context) for name,panel in panels.items()}
    pointwise={name:{k:summarize_metric(v) for k,v in metrics.items()} for name,metrics in raw.items()}
    comparisons={}
    for name,metrics in raw.items():
        if name=='E1':continue
        comparisons[name+'_minus_E1']={k:summarize_metric(difference(v,raw['E1'][k])) for k,v in metrics.items() if k in raw['E1']}
    for n in (1,2,3):
        name='E2a_neutral'+str(n);base='E2a_neutral0'
        comparisons[name+'_minus_neutral0']={k:summarize_metric(difference(v,raw[base][k])) for k,v in raw[name].items() if k in raw[base]}
    count_panels={'E1_prefix':prefix_panel(panels['E1'])}
    count_panels.update({name:{**p,'delta':p['delta'][indices]} for name,p in counts.items()})
    count_raw={name:count_metrics(panel,context) for name,panel in count_panels.items()}
    count_results={name:{'conditions':p['conditions'],'scope':p['scope'],
                        'metrics':{k:summarize_metric(v) for k,v in count_raw[name].items()}} for name,p in count_panels.items()}
    count_differences={}
    for name,metrics in count_raw.items():
        if name=='E1_prefix':continue
        # Native power ranges differ for E2e (1..6) and E1 (1..3).
        keys=[k for k in metrics if k in count_raw['E1_prefix'] and ('native_range' not in k or name=='E2d')]
        count_differences[name+'_minus_E1_matched_prefix']={k:summarize_metric(difference(metrics[k],count_raw['E1_prefix'][k])) for k in keys}
    return {'n_items':len(indices),'n_topics':len(context.topics),'item_ids':selected_ids,'topics':selected_topics,
            'bootstrap':{'draws':draws,'seed':seed,'unit':'topic','paired_all_conditions':True,
                         'resample_entire_topics_then_recompute_item_weighted_ratios':True},
            'named_peer_results':pointwise,'paired_named_peer_differences':comparisons,
            'count_grid_results':count_results,'paired_count_grid_differences':count_differences,
            'adding_ally_interpretation':'Positive change is an output-layer log-odds shift toward opposition to the focal. Named-grid increments average all matched one-ally additions; prefix-grid increments follow the fixed next ally.',
            'count_contrast_not_C':True,
            'power_t_definition':'Pooled no-ally mean deltas: log(delta)=intercept+t*log(n). Undefined whenever any included mean delta is nonpositive; no dropping or sign correction.',
            'pointwise_intervals_are_exploratory_not_multiplicity_adjusted':True}


def build(e1_path,e2_dir,root=ROOT,remote_root='/workspace/voices_nonadditivity_v3',draws=10000,seed=20261031):
    root=Path(root).resolve();e1_path=Path(e1_path);e2_dir=Path(e2_dir);verifier=Verifier(root,remote_root)
    core=read_json(root/'data/core_data_manifest.json');manifest=read_json(root/'data/e2_manifest.json')
    entry=core['outputs']['e1_holdout200_prompts.jsonl'];source=root/entry['path'];verifier.check(source,entry['sha256'])
    e1,identity=closed_analysis(e1_path,verifier,source,entry['n_rows'])
    ids=sorted(e1['item_ids']);expected=item_ids(root/'data/e1_semantic64_items.json')
    if len(ids)!=64 or set(ids)!=expected or not e1['full32']:raise ValueError('Pass the E1 semantic64 R0 analysis, not the full200 analysis')
    topic_by_id=dict(zip(e1['item_ids'],e1['topics']));topics=[topic_by_id[item] for item in ids]
    analyses={'E1':e1};counts={}
    for name,filename,manifest_key,neutral in [*(('E2a_neutral'+str(n),f'e2a_neutral{n}_R0_analysis.json','E2a',n) for n in range(4)),
            ('E2b','e2b_R0_analysis.json','E2b',None),('E2c','e2c_R0_analysis.json','E2c',None),
            ('E2d','e2d_R0_analysis.json','E2d',None),('E2e','e2e_R0_analysis.json','E2e',None)]:
        entry=manifest['outputs'][manifest_key];source=root/entry['path'];verifier.check(source,entry['sha256'])
        analysis,execution=closed_analysis(e2_dir/filename,verifier,source,entry['n_rows'])
        if execution!=identity:raise ValueError('E1/E2 model, readout engine, batch or calibration changed')
        if len(analysis['item_ids'])!=64 or set(analysis['item_ids'])!=set(ids):raise ValueError('E2 does not contain exactly the matched semantic64 items')
        if name in ('E2d','E2e'):
            indices=[analysis['item_ids'].index(item) for item in ids]
            delta=np.array(analysis['item_count_deltas'])[indices]
            if not np.isfinite(delta).all():raise ValueError('Nonfinite count delta')
            conditions=[tuple(c) for c in analysis['conditions']]
            expected_grid=set(itertools.product(range(4 if name=='E2d' else 7),range(3 if name=='E2d' else 4)))
            if set(conditions)!=expected_grid or len(conditions)!=len(expected_grid) or delta.shape!=(64,len(conditions)):
                raise ValueError('Incomplete or changed E2 count grid')
            for value,(n,m) in zip(delta.mean(axis=0),conditions):
                if not np.isclose(value,analysis['point'][f'delta_{n}_{m}'],rtol=1e-10,atol=1e-11):
                    raise ValueError('Item count deltas do not reproduce saved count-curve estimates')
            counts[name]={'conditions':[tuple(c) for c in analysis['conditions']],'delta':delta,
                'scope':'Fixed prefix-count grid; not the complete named-peer lattice; '+('original five-role representative prefixes' if name=='E2d' else 'nine slots, six opponents/three allies; six cyclic positions, not full position balance')}
        else:
            required_masks=C_MASKS if neutral is not None else set(range(32))
            if set(analysis['masks'])!=required_masks:raise ValueError('Unexpected E2 named subset support')
            if neutral is not None and analysis.get('selection',{}).get('filters')!={'neutral_count':neutral}:raise ValueError('Wrong neutral-background filter')
            analyses[name]=analysis
    named={name:reconstruct_named(analysis,ids,topics) for name,analysis in analyses.items()}
    repo=e1['execution_identity']['critical_identity']['repo']
    cohorts={'semantic64_exploratory':compare_cohort(named,counts,ids,topics,set(ids),draws,seed)}
    if repo in MANDATORY:
        keep=item_ids(root/'data/e1_semantic_remaining44_items.json')
        if len(keep)!=44:raise ValueError('Expected the frozen fresh44 sensitivity cohort')
        cohorts['fresh44_sensitivity']=compare_cohort(named,counts,ids,topics,keep,draws,seed+1)
    for path in [e1_path,root/'data/core_data_manifest.json',root/'data/e2_manifest.json',
                 root/'data/e1_semantic64_items.json',Path(__file__),Path(__file__).with_name('analyze_v3.py'),
                 Path(__file__).with_name('summarize_select_e1_v3.py')]:verifier.verified[str(path.resolve())]=verifier.digest(path)
    if repo in MANDATORY:
        path=root/'data/e1_semantic_remaining44_items.json';verifier.verified[str(path.resolve())]=verifier.digest(path)
    return {'schema':'voices_v3_paired_E1_E2_diagnostics_v1','repo_id':repo,'readout':'R0',
        'status':'complete_paired_diagnostics','model_identity':e1['execution_identity']['critical_identity'],
        'cohorts':cohorts,'primary_scope':'semantic64 exploratory paired ablation diagnostics; fresh44 is an additional sensitivity for Qwen4/LFM',
        'C_difference_direction':'E2 minus E1; neutral incremental comparisons use E2a neutral-n minus neutral0',
        'no_desired_effect_selection_or_sign_clipping':True,'independent_CI_subtraction_used':False,
        'file_sha256':verifier.verified}


def cell(metric):
    if metric['estimate'] is None:return '未定义'
    value=f"{metric['estimate']:.4g}";ci=metric['interval95']
    return value+(' ['+', '.join(f'{x:.4g}' for x in ci)+']' if ci is not None else ' [区间未定义]')


def markdown(report):
    primary=report['cohorts']['semantic64_exploratory']
    lines=['# E1/E2 配对诊断','',report['repo_id'], '',
        f"主诊断使用相同 {primary['n_items']} 条、{primary['n_topics']} 个议题，共同按议题 bootstrap {primary['bootstrap']['draws']} 次。以下为点估计及点状 95% 区间，均属探索性诊断。",'',
        '| 变体 | C | C 相对 E1 的配对变化 |','|---|---:|---:|']
    for name,metrics in primary['named_peer_results'].items():
        diff=primary['paired_named_peer_differences'].get(name+'_minus_E1',{}).get('C')
        lines.append('| '+name+' | '+cell(metrics['C'])+' | '+(cell(diff) if diff else '—')+' |')
    lines+=['','增加一名同盟后的 R0 读数变化：正值表示朝原反对方移动；负值仍完整保留。', '',
            '| 初始反对者数/同盟数 | E1 | E2b | E2c | E2d 固定前缀 | E2e 固定前缀 |','|---|---:|---:|---:|---:|---:|']
    for n,m in itertools.product(range(4),range(2)):
        key=f'add_ally_{n}_{m}_to_{m+1}'
        metrics=[primary['named_peer_results'][name][key] for name in ('E1','E2b','E2c')]
        metrics += [primary['count_grid_results'][name]['metrics'][key] for name in ('E2d','E2e')]
        lines.append('| '+f'{n}/{m}'+' | '+' | '.join(cell(v) for v in metrics)+' |')
    lines+=['','计数网格单独汇报，不把计数对比称为命名成员 C。', '',
            '| 网格 | 计数对比 | 无同盟原生范围 t | 共同 1–3 人范围 t |','|---|---:|---:|---:|']
    for name,values in primary['count_grid_results'].items():
        metrics=values['metrics'];lines.append('| '+name+' | '+' | '.join(cell(metrics[key]) for key in ('count_grouping_contrast','power_t_native_range','power_t_common_1_to_3'))+' |')
    if 'fresh44_sensitivity' in report['cohorts']:
        lines+=['','未见 44 条敏感性：','', '| 变体 | 配对 C 变化（相对 E1） |','|---|---:|']
        for name,metrics in report['cohorts']['fresh44_sensitivity']['paired_named_peer_differences'].items():
            if name.endswith('_minus_E1'):lines.append('| '+name+' | '+cell(metrics['C'])+' |')
    lines+=['','配对变化是在每次共同重采样内重算比值再相减；未采用两个独立区间相减。'
            '中性背景增量、三类关系核、全部计数曲线及配对同盟效应差异见 JSON。',
            't 的拟合若遇到任一非正平均增量即保留为未定义，不删点或更改符号。E2e 原生 t 使用 1–6 名反对者，不能与 1–3 人范围直接混同。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--e1-semantic64',type=Path,required=True)
    p.add_argument('--e2-dir',type=Path,required=True);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--remote-root',default='/workspace/voices_nonadditivity_v3');p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=20261031);args=p.parse_args()
    if args.draws<2:raise ValueError('At least two bootstrap draws are required')
    outputs=[Path(str(args.output_prefix)+suffix) for suffix in ('.json','.md')]
    if any(path.exists() for path in outputs):raise FileExistsError('Refusing to overwrite paired diagnostic reports')
    result=build(args.e1_semantic64,args.e2_dir,args.root,args.remote_root,args.draws,args.seed)
    outputs[0].parent.mkdir(parents=True,exist_ok=True)
    for path,text in zip(outputs,[json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',markdown(result)]):
        with path.open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'status':result['status'],'repo_id':result['repo_id'],'cohorts':list(result['cohorts']),'output':str(outputs[0])}))


if __name__=='__main__':main()
