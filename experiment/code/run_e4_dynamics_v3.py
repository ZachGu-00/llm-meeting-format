"""Dedicated E4 dynamics driver: 100k STS draws split over 20 mapping strata."""
import argparse
import functools
import itertools
import json
import os
from pathlib import Path
import platform
import time
import numpy as np

from analyze_v3 import read_json,sha
from e4_simulation import load_cache,predicted_caches,simulate
from e4_dynamic_stratified import simulate_sts_stratified,simulate_pasch

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topic-analysis',required=True);p.add_argument('--kernel-analysis',required=True)
    p.add_argument('--output',required=True);p.add_argument('--stages',choices=['a','b','a,b'],default='a,b')
    p.add_argument('--cache-manifest',default=str(ROOT/'data/e4_cache_manifest.json'))
    p.add_argument('--sts',default=str(ROOT/'data/e4_graphs/sts15_all80.json'))
    p.add_argument('--total-draws-per-design',type=int,default=100000)
    p.add_argument('--pasch-draws',type=int,default=100000);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--chunk-size',type=int,default=128);p.add_argument('--backend',choices=['numpy','torch'],default='numpy')
    p.add_argument('--device',default='cpu');p.add_argument('--rng-backend',choices=['numpy','torch'],default='numpy')
    args=p.parse_args();out=Path(args.output)
    if str(args.device).startswith('cuda') and not os.environ.get('STY'):
        raise RuntimeError('GPU Monte Carlo must run inside GNU screen')
    if out.exists():raise FileExistsError('Refusing to overwrite completed dynamics')
    if args.rounds<1 or args.total_draws_per_design<2 or args.pasch_draws<2:raise ValueError('Positive rounds and at least two draws required')
    point=read_json(args.topic_analysis);kernel=read_json(args.kernel_analysis)
    if point['E1_execution_identity']!=kernel['execution_identity']:raise ValueError('Different E1 kernel identity')
    for path in (args.kernel_analysis,args.cache_manifest,args.sts):
        if sha(path) not in point['input_hashes'].values():raise ValueError('Different point-analysis provenance: '+path)
    cache_path=Path(args.topic_analysis).parent/point['cache_artifact']['filename']
    if sha(cache_path)!=point['cache_artifact']['sha256']:raise ValueError('Changed cache')
    caches=predicted_caches(load_cache(cache_path),kernel['point']['relation_kernels'])
    meta=next(t for t in read_json(args.cache_manifest)['topics'] if t['topic_id']==point['topic_id'])
    systems=read_json(args.sts)['systems'];runner=simulate
    versions={'python':platform.python_version(),'numpy':np.__version__}
    code_paths=[Path(__file__),Path(__file__).with_name('e4_dynamic_stratified.py'),Path(__file__).with_name('e4_simulation.py')]
    if args.backend=='torch':
        import torch
        from e4_simulation_torch import simulate as torch_simulate
        runner=functools.partial(torch_simulate,device=args.device,rng_backend=args.rng_backend)
        versions.update(torch=torch.__version__,cuda=torch.version.cuda)
        code_paths.append(Path(__file__).with_name('e4_simulation_torch.py'))
    elif args.device!='cpu' or args.rng_backend!='numpy':raise ValueError('numpy backend only supports CPU/numpy RNG')
    identity={'source_E4_analysis_sha256':sha(args.topic_analysis),'E1_analysis_sha256':sha(args.kernel_analysis),
              'cache_sha256':sha(cache_path),'backend':args.backend,'device':args.device,'rng_backend':args.rng_backend,
              'runtime_versions':versions,
              'code_sha256':{p.name:sha(p) for p in code_paths}}
    partial=out.with_name(out.stem+'_parts');partial.mkdir(parents=True,exist_ok=True)
    def saved_call(name,kwargs,call):
        path=partial/(name+'.json');expected={'identity':identity,'simulation_arguments':kwargs}
        if path.exists():
            item=read_json(path)
            if item['binding']!=expected:raise ValueError('Cannot resume changed simulation: '+str(path))
            return item['result']
        started=time.perf_counter();result=call();item={'binding':expected,'result':result,'seconds':time.perf_counter()-started}
        temporary=path.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8') as f:json.dump(item,f,allow_nan=False)
        temporary.replace(path)
        print(json.dumps({'completed':name,'seconds':item['seconds']}),flush=True)
        return result
    result={'topic_id':point['topic_id'],'topic':point['topic'],'identity':identity,'MC_completed':True,
            'requested_stages':args.stages,'source_E4_analysis_sha256':sha(args.topic_analysis),
            'E1_kernel_parameter_uncertainty_propagated':False}
    if 'a' in args.stages:
        result['E4a_MC']=simulate_pasch(caches,meta,draws=args.pasch_draws,rounds=args.rounds,
            chunk_size=args.chunk_size,simulator=runner,on_condition=lambda name,kw,call:saved_call('E4a_'+name,kw,call))
    if 'b' in args.stages:
        result['E4b_MC']={}
        for n_target,tau in itertools.product((3,5,7),(-1,1)):
            name=f'm{n_target}_tau{tau}'
            result['E4b_MC'][name]=simulate_sts_stratified(caches,systems,point['E4b']['point_to_agent_mappings'],
                point['topic_id'],n_target,tau,total_draws=args.total_draws_per_design,rounds=args.rounds,
                chunk_size=args.chunk_size,simulator=runner,
                on_mapping=lambda mi,kw,call:saved_call(f'E4b_{name}_map{mi:02d}',kw,call))
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,allow_nan=False)
    print(json.dumps({'output':str(out),'completed_stages':args.stages,'STS_total_draws_per_design':args.total_draws_per_design}))


if __name__=='__main__':main()
