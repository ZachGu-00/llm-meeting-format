"""Concise cross-model E9 affine-position report; missing models stay pending."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import analyze_e9_affine_pairs_v3 as analysis
from summarize_e0_panel_v3 import local_path

ROOT=Path(__file__).resolve().parents[1]


def build(paths,root=ROOT,expected_models=None):
    root=Path(root).resolve();queue_path=root/'setup/download_queue.json'
    expected=list(expected_models) if expected_models else [r['repo_id'] for r in analysis.io.read_json(queue_path)]
    if not expected or len(expected)!=len(set(expected)):raise ValueError('Use an explicit nonempty unique model scope')
    verified={};cache={};byrepo={};errors=[]
    def check(path,digest):
        p=Path(path);key=(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
        if key not in cache:cache[key]=analysis.io.sha(p)
        if cache[key]!=digest:raise ValueError('E9 source SHA mismatch: '+str(p))
        verified[str(p.resolve())]=digest
    if expected_models is None:check(queue_path,analysis.io.sha(queue_path))
    for path in paths:
        path=Path(path).resolve()
        try:
            result=analysis.io.read_json(path);repo=result['repo_id']
            if repo not in expected or repo in byrepo:raise ValueError('Duplicate/out-of-scope model input')
            if (result.get('schema')!='voices_v3_E9_affine_pair_diagnostic_v1' or result.get('status')!='complete_diagnostic' or
                result.get('n_items')!=20 or result.get('scope',{}).get('fresh_confirmation') is not False or
                result.get('main_E1_E2_selection_and_PK_unchanged') is not True or
                result['bootstrap'].get('draws')!=10000 or result['bootstrap'].get('unit')!='topic' or
                result['bootstrap'].get('shared_weights_across_panels_and_readouts') is not True or
                result['readouts'] not in (['R0','R1'],['R1']) or set(result['estimates'])!=set(result['readouts'])):
                raise ValueError('Not a closed same20 paired-topic diagnostic at the declared budget')
            if not result.get('file_sha256'):raise ValueError('Missing diagnostic source hashes')
            for source,digest in result['file_sha256'].items():check(local_path(source,root),digest)
            artifact=result['bootstrap_artifact'];sidecar=path.parent/artifact['filename']
            if sidecar.resolve().parent!=path.parent:raise ValueError('Bootstrap artifact is outside its result directory')
            check(sidecar,artifact['sha256']);check(path,analysis.io.sha(path))
            byrepo[repo]={'repo_id':repo,'status':'complete_verified','path':str(path),'sha256':analysis.io.sha(path),
                'n_topics':result['n_topics'],'item_ids':result['item_ids'],'readouts':result['readouts'],
                'estimates':result['estimates'],'paired_readout_comparisons':result['paired_readout_comparisons']}
        except (ValueError,KeyError,TypeError,OSError) as error:errors.append({'path':str(path),'error':str(error)})
    item_sets={tuple(row['item_ids']) for row in byrepo.values()}
    if len(item_sets)>1:raise ValueError('Models do not use identical fixed old20 items')
    rows=[byrepo.get(repo,{'repo_id':repo,'status':'pending'}) for repo in expected]
    return {'schema':'voices_v3_E9_affine_panel_report_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete_requested_diagnostic_panel' if len(byrepo)==len(expected) and not errors else 'partial_or_pending',
        'requested_models':expected,'n_verified_models':len(byrepo),'models':rows,'input_errors':errors,
        'main_E1_E2_selection_and_PK_unchanged':True,'fresh_confirmation':False,
        'interpretation':'Changed C shows sensitivity to original pair-position sampling; the geometry counterexample proves an identification limitation, not that actual effects are artifacts. Nonzero or unchanged effects are retained.',
        'no_positive_effect_or_nonsignificance_gate':True,'verified_source_sha256':verified,
        'reporter_sha256':analysis.io.sha(__file__)}


def markdown(result):
    lines=['# E9 配对位置平衡：模型汇总','',
        f"已核验 {result['n_verified_models']}/{len(result['requested_models'])} 个模型；状态：{result['status']}。",'',
        '| 模型 / 读出 | 原 10 排列 C | 新增 10 排列 C | 合并 20 排列 C | 新增−原（配对95%） | 合并−原（配对95%） |',
        '|---|---:|---:|---:|---:|---:|']
    for row in result['models']:
        if row['status']!='complete_verified':lines.append('| '+row['repo_id']+' | 待完成 | — | — | — | — |');continue
        for readout,values in row['estimates'].items():
            cells=[row['repo_id']+' / '+readout,*[analysis.cell(values['panels'][p]['C']) for p in analysis.PANELS],
                analysis.cell(values['paired_differences']['additional10_minus_old10']['C']),
                analysis.cell(values['paired_differences']['combined20_minus_old10']['C'])]
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','所有比较固定使用既有同一批 20 条材料。三类关系核及 R0/R1 配对比较保存在 JSON 和各模型报告。原单人均值固定复用，不增加独立样本；区间为 10000 次配对议题 bootstrap，未做多重校正。',
        '原排列没有平衡角色对的距离；补齐后每个角色对覆盖全部 20 种有序位置对。相邻位置 toy 的 C 为原 0.8λ、新 −0.8λ、合并 0，证明一种识别限制，不能据此断言真实模型效应都是位置伪效应。',
        'C 改变表示对原位置采样敏感；不变或仍非零同样保留。此处不判断内部机制，不改变主 E1/E2、|C| 选模或 E4 主 PK，也不作为 fresh180 确证或完整高阶实验。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',action='append',default=[])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--expected-models',nargs='+')
    parser.add_argument('--output-prefix',type=Path,required=True);args=parser.parse_args()
    paths=[Path(str(args.output_prefix)+suffix) for suffix in ('.json','.md')]
    if any(path.exists() for path in paths):raise FileExistsError('Refusing to overwrite an E9 panel report')
    report=build(args.analysis,args.root,args.expected_models);paths[0].parent.mkdir(parents=True,exist_ok=True)
    for path,text in [(paths[0],json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),(paths[1],markdown(report))]:
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
    print(json.dumps({'status':report['status'],'n_verified_models':report['n_verified_models'],'output':str(paths[0])}))


if __name__=='__main__':main()
