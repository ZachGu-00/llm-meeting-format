"""Pre-unsealing E1 leave-one-topic kernels and development-only content fits."""
import argparse
from pathlib import Path
import sys
import numpy as np
from diagnostics_d3_d4 import features,fit_ridge


def build(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path[:0]=[str(ext),str(base/'code')]
    from common import load,sha,write_new,verify_bindings
    from analyze_v3 import read_closed,aggregate,sufficient,point
    from e4_simulation import load_cache
    old_path=base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json';old=load(old_path)
    initial_path=base/'setup/e1_e2_queue_history/20261031T053546212590Z/plan.json';initial=load(initial_path)
    g1_path=root/'G1_plan.json';g1=load(g1_path)
    target=root/'gate1/frozen_predictors.json'
    if target.exists():
        result=load(target);verify_bindings(result['source_sha256']);return result
    original_topics=load(base/'data/e4_agents.json');new_topics=load(root/'data/e4_agents_new.json')
    topics={t['topic_id']:t for t in original_topics+new_topics};pairs=load(root/'data/M2/pair_features.json')
    source_paths=[old_path,initial_path,g1_path,root/'G2_G3_plan.json',root/'data/M2/manifest.json',
                  root/'data/M2/pair_features.json',root/'data/M1_manifest.json',Path(__file__),root/'diagnostics_d3_d4.py',
                  root/'diagnostics_d1_d2.py',base/'code/analyze_v3.py',base/'code/e4_simulation.py']
    sources={str(p):sha(p) for p in source_paths};records={}
    models={m['repo_id']:m for m in old['selected_models']}
    models.update({m['repo_id']:m for m in g1['models']})
    for repo,model in models.items():
        expanded=Path(model['model_dir']).is_relative_to(ext/'expanded_models')
        if expanded:
            task_path=ext/'plans/expansion/models'/model['slug']/'tasks.json';tasks=load(task_path)['jobs'];sources[str(task_path)]=sha(task_path)
        else:tasks=[j for j in initial['jobs'] if j['model']==repo]
        record={'model':repo,'revision':model['revision'],'readouts':{}}
        for readout in ('R0','R1'):
            job=next(j for j in tasks if j['name']=='E1_'+readout)
            if readout=='R0':
                analysis=Path(model['kernel_analysis']) if 'kernel_analysis' in model else Path(next(a['output'] for a in job['analyses'] if a['name']=='holdout200_R0_analysis'))
            else:
                name='matched44_R1_analysis' if model.get('old20_excluded_from_confirmation') else 'semantic64_R1_analysis'
                analysis=Path(next(a['output'] for a in job['analyses'] if a['name']==name))
            fitted=load(analysis);verify_bindings(fitted['file_sha256'])
            panel=aggregate(read_closed(job['input'],job['output']),readout=readout,keep_ids=set(fitted['item_ids']))
            stats=sufficient(panel);full=point(stats)['relation_kernels']
            if any(full[k] is None or not np.isclose(full[k],fitted['point']['relation_kernels'][k],rtol=0,atol=1e-12) for k in full):
                raise ValueError('Reconstructed E1 reference kernels differ')
            kernels={}
            for tid,topic in topics.items():
                indices=[i for i,t in enumerate(panel['topics']) if t!=topic['topic']]
                if not indices:raise ValueError('No E1 data remains after leaving the target topic out')
                values=point(stats,indices)['relation_kernels']
                if any(v is None or not np.isfinite(v) for v in values.values()):raise ValueError('Undefined LOTO E1 kernel')
                kernels[tid]={'relation_kernels':values,'excluded_topic':topic['topic'],
                              'excluded_items':len(panel['item_ids'])-len(indices),'training_items':len(indices),
                              'training_item_ids':[panel['item_ids'][i] for i in indices],
                              'training_topics':sorted({panel['topics'][i] for i in indices})}
            record['readouts'][readout]={'reference_analysis':str(analysis),'full_relation_kernels':full,'by_topic':kernels}
            for p in [str(analysis),job['input'],job['output'],job['output']+'.run.json',job['output']+'.summary.json']:
                sources[p]=sha(p)
        if repo in old['selected_repos']:
            cases=[]
            for job in old['jobs']:
                if job['model']!=repo:continue
                p=Path(job['deferred_analysis']['output_dir'])/(job['topic_id']+'_caches.npz');sources[str(p)]=sha(p)
                cases.append(features(load_cache(p),topics[job['topic_id']],pairs[job['topic_id']]))
            y=np.concatenate([f['residual'] for f in cases])
            record['content_coefficients']=fit_ridge(np.concatenate([f['content'] for f in cases]),y).tolist()
            record['matched_relation_only_coefficients']=fit_ridge(np.concatenate([f['kernel'] for f in cases]),y).tolist()
            record['training_E4_topics']=[t['topic_id'] for t in original_topics]
            d3_path=root/'diagnostics/D3_D4/report.json';sources[str(d3_path)]=sha(d3_path)
            record['fixed_loss_normalizer']=next(m for m in load(d3_path)['models'] if m['model']==repo)['methods']['P0']['contrast_MSE']
        records[repo]=record
        print(repo+' E1 kernels and eligible development predictors frozen',flush=True)
    result={'schema':'voices_v31_preunsealing_predictors_v1','models':records,'source_sha256':sources,
            'sealed_outputs_read':False,'E1_leave_one_topic_out':True,'content_training':'original six E4 topics only',
            'R1_content_sensitivity':'Transfer the frozen R0-trained content coefficients to R1 low-order features; report this limitation explicitly'}
    write_new(target,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();r=build(a.base,a.extension);print({'status':'predictors_frozen_before_unsealing','models':len(r['models'])})
