"""Paired old20 E9 diagnostic: original10, complementary10 and balanced20.

Only two-peer outcomes are newly scored. Original baseline/singleton averages
are reused after exact source-message multiset validation. No main experiment,
model selection, higher-order estimate or rule R-squared is changed here.
"""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path

import numpy as np
import analyze_v3 as io
import build_e9_affine_balance_v3 as materials

ROOT=Path(__file__).resolve().parents[1]
PAIR_ROLES=list(itertools.combinations(range(5),2))
PAIR_MASKS=[(1<<j)|(1<<k) for j,k in PAIR_ROLES]
RELATIONS={'opponent_opponent':[i for i,(j,k) in enumerate(PAIR_ROLES) if k<3],
    'opponent_ally':[i for i,(j,k) in enumerate(PAIR_ROLES) if j<3<=k],
    'ally_ally':[i for i,(j,k) in enumerate(PAIR_ROLES) if j>=3]}
PANELS=('old10','additional10','combined20')
METRICS=('C',*('kernel_'+r for r in RELATIONS))


def scalar(value):return float(value) if np.isfinite(value) else None


def ratio(num,den):
    return np.divide(num,den,out=np.full(np.broadcast_shapes(np.shape(num),np.shape(den)),np.nan),where=np.asarray(den)>0)


def metric(point,draws):
    values=np.asarray(draws);bad=int((~np.isfinite(values)).sum())
    return {'estimate':scalar(point),'interval95':None if bad else np.quantile(values,[.025,.975]).tolist(),
        'undefined_draws':bad,'undefined_draws_not_dropped':True}


def components(alpha,single_y,pair_y):
    """Direct two-person C contrast and original-singleton standardized kernels."""
    alpha=np.asarray(alpha);single_y=np.asarray(single_y);pair_y=np.asarray(pair_y)
    if single_y.shape!=(len(alpha),5) or pair_y.shape!=(len(alpha),10):raise ValueError('Need baseline, five singletons and ten pairs for every item')
    if not all(np.isfinite(value).all() for value in (alpha,single_y,pair_y)):raise ValueError('Nonfinite item readout; no effect-based exclusions are allowed')
    v=single_y-alpha[:,None]
    interactions=np.stack([pair_y[:,i]-single_y[:,j]-single_y[:,k]+alpha for i,(j,k) in enumerate(PAIR_ROLES)],axis=1)
    means={relation:pair_y[:,ix].mean(axis=1) for relation,ix in RELATIONS.items()}
    c_direct=means['opponent_opponent']+means['ally_ally']-2*means['opponent_ally']
    nums={relation:interactions[:,ix].mean(axis=1) for relation,ix in RELATIONS.items()}
    c_interactions=nums['opponent_opponent']+nums['ally_ally']-2*nums['opponent_ally']
    if not np.allclose(c_direct,c_interactions,rtol=1e-11,atol=1e-11):raise ValueError('Direct pair-y C does not equal the interaction expression')
    denominators={relation:np.stack([.5*(np.abs(v[:,PAIR_ROLES[i][0]])+np.abs(v[:,PAIR_ROLES[i][1]])) for i in ix],axis=1).mean(axis=1)
        for relation,ix in RELATIONS.items()}
    return {'numerators':np.stack([c_direct,*(nums[r] for r in RELATIONS)],axis=1),
        'denominators':np.stack([np.abs(v[:,:3]).mean(axis=1),*(denominators[r] for r in RELATIONS)],axis=1),
        'pair_interactions':interactions,'singleton_effects':v,
        'direct_vs_interaction_max_abs_difference':float(np.max(np.abs(c_direct-c_interactions)))}


def geometry_proof(original):
    """A stance-blind linear-adjacency term is a counterexample, not model data."""
    orders={'old10':original.permutations(),'additional10':[materials.affine_order(a,b) for a in (2,3) for b in range(5)]}
    orders['combined20']=orders['old10']+orders['additional10'];result={}
    for name,perms in orders.items():
        probabilities=np.array([np.mean([abs(p.index(materials.ROLES[j])-p.index(materials.ROLES[k]))==1 for p in perms]) for j,k in PAIR_ROLES])
        value=components(np.zeros(1),np.ones((1,5)),2+probabilities[None,:])
        result[name]={'pair_linear_adjacency_probabilities':probabilities.tolist(),
            'class_linear_adjacency_probabilities':{r:float(probabilities[ix].mean()) for r,ix in RELATIONS.items()},
            'C_per_unit_lambda':float(value['numerators'][0,0])}
    expected={'old10':.8,'additional10':-.8,'combined20':0.}
    if any(not np.isclose(result[k]['C_per_unit_lambda'],v,atol=1e-12) for k,v in expected.items()):raise ValueError('Frozen geometry counterexample does not reproduce its algebra')
    return {'synthetic_only':True,'formula':'y(S)=|S|+lambda*number_of_active_linearly_adjacent_pairs',
        'panels':result,'conclusion':'Identifies a possible stance-blind pair-position contribution, not proof of an actual model artifact.'}


def verify_materials(root,manifest_path):
    root=Path(root);manifest=io.read_json(manifest_path);sources={}
    if (manifest.get('schema')!='voices_v3_E9_affine_pair_balance_materials_v1' or
        manifest.get('main_E1_E2_selection_and_PK_unchanged') is not True or
        manifest.get('old_permutations')!=list(range(10)) or manifest.get('new_permutations')!=list(range(10,20))):
        raise ValueError('Unexpected E9 supplementary materials contract')
    specs=[manifest['builder'],manifest['amendment'],manifest['items'],manifest['additional_prompts'],*manifest['original_sources'].values()]
    for spec in specs:
        path=root/spec['path']
        if io.sha(path)!=spec['sha256'] or path.stat().st_size!=spec['bytes']:raise ValueError('E9 material source changed: '+str(path))
        sources[str(path.resolve())]=spec['sha256']
    if io.sha(materials.__file__)!=manifest['builder']['sha256']:raise ValueError('Loaded supplementary builder differs from its manifest')
    if manifest['amendment']['path']!='provenance/E9_affine_pair_balance_20261031.txt':raise ValueError('Wrong E9 amendment binding')
    if manifest['additional_prompts']['n_rows']!=4000:raise ValueError('Expected the fixed 4000 additional pair rows')
    core=io.read_json(root/manifest['original_sources']['core_manifest']['path'])
    ids=manifest['item_ids'];items=io.read_json(root/manifest['items']['path'])
    if (ids!=sorted(core['reused20_ids']) or len(ids)!=len(set(ids)) or len(ids)!=20 or
        [item['item_id'] for item in items]!=ids):raise ValueError('Diagnostic must retain exactly the original fixed twenty items')
    original=materials.renderer(root/manifest['original_sources']['renderer']['path'])
    coverage=materials.coverage_proof(original);reuse=materials.reuse_proof(original,items)
    if (coverage!=manifest['proofs']['pair_slot_coverage'] or reuse!=manifest['proofs']['baseline_singleton_message_multisets']):
        raise ValueError('Independent renderer replay disagrees with pair balance/singleton reuse proofs')
    for path in (Path(__file__),Path(io.__file__),Path(manifest_path)):
        sources[str(path.resolve())]=io.sha(path)
    return manifest,items,original,sources,{'pair_slot_coverage':coverage,'baseline_singleton_message_multisets':reuse}


def run_contract(raw,required_readouts,expected_rows,sources):
    raw=Path(raw);run_path=Path(str(raw)+'.run.json');summary_path=Path(str(raw)+'.summary.json')
    run,summary=io.read_json(run_path),io.read_json(summary_path)
    available={'R0','R1'} if run.get('readout')=='R0_R1' else {run.get('readout')}
    if (not set(required_readouts)<=available or run.get('n_input')!=expected_rows or
        summary.get('exact_id_and_count_equality') is not True or summary.get('output_bytes')!=raw.stat().st_size or
        any(summary.get(key)!=value for key,value in run.items())):raise ValueError('Raw readout/count/closed-summary identity differs')
    for path in (raw,run_path,summary_path):sources[str(path.resolve())]=io.sha(path)
    return run


def load_original(source,raw,readout,items,original,sources,expected_rows):
    run=run_contract(raw,[readout],expected_rows,sources);ids=[x['item_id'] for x in items];index={x:i for i,x in enumerate(ids)}
    by_id=dict(zip(ids,items));alpha=np.full((20,2),np.nan);singles=np.full((20,5,10,2),np.nan);pairs=np.full((20,10,10,2),np.nan)
    seen=set();q=[];retained=0;selected_all=0
    for row in io.read_closed(source,raw):
        item=row['item_id']
        if item not in index:continue
        selected_all+=1;mask=row['subset_mask']
        if mask not in materials.REUSE_MASKS+PAIR_MASKS:continue
        p,label=row['permutation'],row['label_map'];key=(item,mask,p,label)
        if key in seen or label not in (0,1) or p not in range(10) or (mask==0 and p!=0):raise ValueError('Duplicate/unexpected original low-order cell')
        expected=original.render(by_id[item],mask,p,label)
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Original row differs from frozen renderer')
        value=float(row[readout+'_y_AB'])*(1 if row['positive_letter']=='A' else -1)
        if not np.isfinite(value):raise ValueError('Nonfinite original readout')
        i=index[item]
        if mask==0:alpha[i,label]=value
        elif mask.bit_count()==1:singles[i,mask.bit_length()-1,p,label]=value
        else:pairs[i,PAIR_MASKS.index(mask),p,label]=value
        if readout=='R0':q.append(float(row['q']))
        seen.add(key);retained+=1
    if retained!=6040 or selected_all!=12440 or not all(np.isfinite(x).all() for x in (alpha,singles,pairs)):
        raise ValueError('Original same20 baseline/singleton/pair panel is incomplete')
    return {'alpha':alpha.mean(axis=1),'single_y':singles.mean(axis=3).mean(axis=2),'pair_y':pairs.mean(axis=3).mean(axis=2),
        'run':run,'retained_low_order_physical_rows':retained,'selected_full32_physical_rows':selected_all,
        'q_median_retained_R0':float(np.median(q)) if q else None}


def load_additional(source,raw,readouts,items,original,sources):
    run=run_contract(raw,readouts,4000,sources)
    requested='R0_R1' if readouts==['R0','R1'] else 'R1'
    if run['readout']!=requested:raise ValueError('All additional readouts must be analyzed; do not omit an available branch')
    ids=[x['item_id'] for x in items];index={x:i for i,x in enumerate(ids)};by_id=dict(zip(ids,items))
    values={readout:np.full((20,10,10,2),np.nan) for readout in readouts};seen=set();q=[]
    for row in io.read_closed(source,raw):
        item,mask,p,label=row['item_id'],row['subset_mask'],row['permutation'],row['label_map']
        key=(item,mask,p,label)
        if item not in index or mask not in PAIR_MASKS or p not in range(10,20) or label not in (0,1) or key in seen:
            raise ValueError('Duplicate/unexpected supplementary pair cell')
        a,b=2+(p-10)//5,(p-10)%5
        expected=materials.additional_row(original,by_id[item],mask,a,b,label)
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Supplementary row differs from fixed affine renderer')
        for readout in readouts:
            value=float(row[readout+'_y_AB'])*(1 if row['positive_letter']=='A' else -1)
            if not np.isfinite(value):raise ValueError('Nonfinite supplementary readout')
            values[readout][index[item],PAIR_MASKS.index(mask),p-10,label]=value
        if 'R0' in readouts:q.append(float(row['q']))
        seen.add(key)
    if len(seen)!=4000 or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Missing supplementary pair/label/order cells')
    return {r:v.mean(axis=3).mean(axis=2) for r,v in values.items()},run,float(np.median(q)) if q else None


def paired_summary(original,new_pairs,topics,draws=10000,seed=20261031):
    if draws<2:raise ValueError('At least two topic bootstrap draws required')
    names=sorted(set(topics));labels=np.asarray(topics)
    weights=np.random.default_rng(seed).multinomial(len(names),np.full(len(names),1/len(names)),size=draws)
    output={};per_item={};artifacts={'topic_weights':weights};point_cache={};draw_cache={}
    for readout,old in original.items():
        pair_panels={'old10':old['pair_y'],'additional10':new_pairs[readout],
            'combined20':(old['pair_y']+new_pairs[readout])/2}
        records={};statistics={};point_cache[readout]={};draw_cache[readout]={}
        for panel,pairs in pair_panels.items():
            comp=components(old['alpha'],old['single_y'],pairs);num,den=comp['numerators'],comp['denominators']
            tn=np.stack([num[labels==t].sum(axis=0) for t in names]);td=np.stack([den[labels==t].sum(axis=0) for t in names])
            point=ratio(num.sum(axis=0),den.sum(axis=0));samples=ratio(weights@tn,weights@td)
            point_cache[readout][panel]=point;draw_cache[readout][panel]=samples
            records[panel]={name:metric(point[i],samples[:,i]) for i,name in enumerate(METRICS)}
            statistics[panel]={'pair_y':pairs.tolist(),'metric_numerators':num.tolist(),
                'metric_denominators':den.tolist(),'pair_interactions':comp['pair_interactions'].tolist(),
                'direct_vs_interaction_max_abs_difference':comp['direct_vs_interaction_max_abs_difference']}
            artifacts[readout+'_'+panel+'_metrics']=samples
        p,d=point_cache[readout],draw_cache[readout]
        if not np.allclose(d['combined20'],(d['old10']+d['additional10'])/2,equal_nan=True,atol=1e-12):
            raise ValueError('Balanced panel is not equal-weight old/new under the shared denominator')
        differences={}
        for later,earlier in [('additional10','old10'),('combined20','old10'),('combined20','additional10')]:
            key=later+'_minus_'+earlier;points=p[later]-p[earlier];samples=d[later]-d[earlier]
            differences[key]={name:metric(points[i],samples[:,i]) for i,name in enumerate(METRICS)}
            artifacts[readout+'_'+key+'_metrics']=samples
        output[readout]={'panels':records,'paired_differences':differences}
        per_item[readout]={'original_alpha':old['alpha'].tolist(),'original_singleton_y':old['single_y'].tolist(),
            'original_singleton_effects':(old['single_y']-old['alpha'][:,None]).tolist(),'panels':statistics}
    cross={}
    if set(original)=={'R0','R1'}:
        for panel in PANELS:
            points=point_cache['R1'][panel]-point_cache['R0'][panel];samples=draw_cache['R1'][panel]-draw_cache['R0'][panel]
            cross[panel+'_R1_minus_R0']={name:metric(points[i],samples[:,i]) for i,name in enumerate(METRICS)}
            artifacts[panel+'_R1_minus_R0_metrics']=samples
        for panel in ('additional10','combined20'):
            points=(point_cache['R1'][panel]-point_cache['R1']['old10'])-(point_cache['R0'][panel]-point_cache['R0']['old10'])
            samples=(draw_cache['R1'][panel]-draw_cache['R1']['old10'])-(draw_cache['R0'][panel]-draw_cache['R0']['old10'])
            cross[panel+'_change_R1_minus_R0']={name:metric(points[i],samples[:,i]) for i,name in enumerate(METRICS)}
    return output,per_item,cross,artifacts,names


def build(original_r0,original_r1,additional,manifest_path,root=ROOT,draws=10000,seed=20261031):
    root=Path(root).resolve();manifest,items,renderer,sources,proofs=verify_materials(root,manifest_path)
    readouts=['R0','R1'] if original_r0 else ['R1'];paths=([Path(original_r0)] if original_r0 else [])+[Path(original_r1),Path(additional)]
    identity=io.execution_identity(paths,allow_mixed=False);original={}
    core=io.read_json(root/manifest['original_sources']['core_manifest']['path'])
    for readout,raw,key in [('R0',original_r0,'e1_holdout200_prompts'),('R1',original_r1,'e1_semantic64_prompts')]:
        if raw is None:continue
        source=root/manifest['original_sources'][key]['path'];count=core['outputs'][source.name]['n_rows']
        original[readout]=load_original(source,raw,readout,items,renderer,sources,count)
    new,new_run,new_q=load_additional(root/manifest['additional_prompts']['path'],additional,readouts,items,renderer,sources)
    topics=[x['topic'] for x in items];estimates,statistics,cross,artifacts,unique=paired_summary(original,new,topics,draws,seed)
    result={'schema':'voices_v3_E9_affine_pair_diagnostic_v1','status':'complete_diagnostic','created_utc':datetime.now(timezone.utc).isoformat(),
        'repo_id':identity['critical_identity']['repo'],'readouts':readouts,'n_items':20,'n_topics':len(unique),
        'item_ids':[x['item_id'] for x in items],'topics':topics,'metric_order':METRICS,'pair_role_indices':PAIR_ROLES,
        'pair_masks':PAIR_MASKS,'estimates':estimates,'paired_readout_comparisons':cross,'per_item':statistics,
        'bootstrap':{'draws':draws,'seed':seed,'unit':'topic','topic_names':unique,'shared_weights_across_panels_and_readouts':True,
            'ratio_recomputed_from_resampled_sums':True,'intervals':'pointwise exploratory percentile95; not multiplicity corrected',
            'undefined_draws_not_dropped':True,'small_topic_sample_caveat':'Only the topics represented in the same old20 items are independent clusters.'},
        'estimands':{'C_numerator':'mean(yOO)+mean(yAA)-2*mean(yOA); exactly equals the original mean pair-interaction contrast',
            'C_denominator':'mean absolute original opponent singleton effects; ratio of item sums, never mean of item ratios',
            'pair_interaction':'mean_pair_y-original_single_y_j-original_single_y_k+original_alpha',
            'relation_kernel':'class mean pair interaction divided by class mean 0.5*(abs(original_v_j)+abs(original_v_k))',
            'balanced20':'0.5*(old10_pair_mean+additional10_pair_mean), original baseline/singleton means unchanged'},
        'geometry_identification_proof':geometry_proof(renderer),'material_proofs_replayed':proofs,
        'reuse_policy':manifest['reuse_policy'],'main_E1_E2_selection_and_PK_unchanged':True,
        'scope':{'same_previously_used20_only':True,'fresh_confirmation':False,'adaptive_measurement_diagnostic':True,
            'higher_order_estimated':False,'full32_rule_R2_estimated':False,'new_physical_rows':4000,
            'original_retained_low_order_rows_per_readout':6040,'original_selected_full32_rows_per_readout':12440,
            'new_baseline_or_singleton_calls':0,'no_effect_based_item_or_model_filter':True},
        'readout_quality':{'original_retained_R0_q_median':original.get('R0',{}).get('q_median_retained_R0'),
            'additional_R0_q_median':new_q,'no_quality_threshold_used_to_filter_diagnostic_results':True},
        'execution_identity':identity,'additional_readout_policy':new_run['readout'],
        'limitations':['A toy counterexample establishes nonidentification under the original pair-position sampling, not that any observed model effect is an artifact.',
            'Balanced pair positions remove class imbalance for a stance/role-independent pair-position term; argument-by-position and higher-order interactions can remain.',
            'Changed C diagnoses sensitivity to positional sampling, not an internal attention mechanism. Unchanged C supports only robustness to this particular counterbalance.',
            'R0/R1 comparisons use identical twenty items and topic draws; readout scales differ and no fresh180 or semantic64 confirmation is claimed.'],
        'file_sha256':sources}
    return result,artifacts


def cell(value):
    if value is None or value.get('estimate') is None:return '未定义'
    ci=value['interval95'];text=f"{value['estimate']:.4g}"
    return text+(' ['+', '.join(f'{x:.4g}' for x in ci)+']' if ci else ' [区间未定义]')


def markdown(result):
    lines=['# E9 配对位置平衡诊断','',result['repo_id'],'',
        f"同一批既有 20 条材料、{result['n_topics']} 个议题；新增 4000 条两位同伴条件读数。不是 fresh180 确证。",'',
        '| 读出 / 指标 | 原 10 排列 | 新增 10 排列 | 合并 20 排列 | 新增−原 | 合并−原 |',
        '|---|---:|---:|---:|---:|---:|']
    for readout,values in result['estimates'].items():
        for name in METRICS:
            columns=[readout+' / '+name,*[cell(values['panels'][p][name]) for p in PANELS],
                cell(values['paired_differences']['additional10_minus_old10'][name]),cell(values['paired_differences']['combined20_minus_old10'][name])]
            lines.append('| '+' | '.join(columns)+' |')
    lines+=['',f"区间使用同一组 {result['bootstrap']['draws']} 次议题配对重采样，在每次重采样内重新计算比值；原单人/空组均值及分母固定复用，不产生新的独立重复。",
        '原排列平衡单个角色位置，但保留角色对的循环距离。立场无关的相邻位置 toy 可产生原 C=0.8λ、新 C=−0.8λ、合并 C=0。这个证明说明识别限制，不代表真实模型结果必然为位置伪效应。',
        '主 E1/E2、|C| 选模及主 E4 PK 均不改变。保留非零、负值与未定义结果；不由这些低阶补充条件声称完整高阶交互、规则 R² 或内部机制。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-r0');parser.add_argument('--original-r1',required=True)
    parser.add_argument('--additional',required=True);parser.add_argument('--manifest',type=Path,default=ROOT/'data/e9_affine_balance/manifest.json')
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--draws',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    paths=[Path(str(args.output_prefix)+suffix) for suffix in ('.json','.md','_bootstrap.npz')]
    if any(p.exists() for p in paths):raise FileExistsError('Refusing to overwrite an E9 diagnostic snapshot')
    result,arrays=build(args.original_r0,args.original_r1,args.additional,args.manifest,args.root,args.draws,args.seed)
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    with paths[2].open('xb') as stream:np.savez_compressed(stream,**arrays)
    result['bootstrap_artifact']={'filename':paths[2].name,'sha256':io.sha(paths[2])}
    for path,text in [(paths[0],json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),(paths[1],markdown(result))]:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
    print(json.dumps({'status':result['status'],'repo_id':result['repo_id'],'readouts':result['readouts'],'output':str(paths[0])}))


if __name__=='__main__':main()
