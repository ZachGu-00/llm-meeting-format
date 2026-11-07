"""Development-only D1 repeat stability and D2 n3 decomposition on original E4."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

SEED = 20261101


def split_caches(rows, n=15):
    """Balanced diagonal slot/label halves, with disjoint two-label baselines.

    Original baseline cells have two labels, not four slot/label repetitions.
    Allocate baseline labels by focal-id/stance parity, balancing labels overall.
    This quantifies deterministic rendering stability, not independent draws.
    """
    arrays = [{k: np.full(shape, np.nan) for k, shape in
               [('alpha', (n, 2)), ('pair', (n, n, 4)), ('triple', (n, n, n, 8))]} for _ in range(2)]
    cells = {}; count = 0
    for r in rows:
        count += 1
        if r['cache_kind'] == 'five': continue
        j = r['focal_agent_id']; peers = tuple(r['peer_agent_ids'])
        bits = tuple(int(s == 1) for s in [r['focal_stance'], *r['peer_stances']])
        position, label = r['permutation'], r['label_map']
        half = position ^ label ^ ((j + bits[0]) % 2)
        key = (r['cache_kind'], j, peers, bits, half)
        val = float(r['R0_y_AB']) * (1 if r['positive_letter'] == 'A' else -1)
        if not np.isfinite(val): raise ValueError('Nonfinite original E4 score')
        cells.setdefault(key, []).append(val)
    for (kind, j, peers, bits, half), values in cells.items():
        if len(values) != (1 if kind == 'baseline' else 2): raise ValueError('Missing repeat-half cells')
        index = sum(b << (len(bits)-1-i) for i, b in enumerate(bits)); value = np.mean(values)
        if kind == 'baseline': arrays[half]['alpha'][j, bits[0]] = value
        elif kind == 'pair': arrays[half]['pair'][j, peers[0], index] = value
        elif kind == 'triple': arrays[half]['triple'][j, *peers, index] = value
    for a in arrays:
        a.update(n=n, five={}, orientation='global_support')
        if not np.isfinite(a['alpha']).all(): raise ValueError('Baseline half incomplete')
        for j in range(n):
            for k in range(n):
                if k != j and not np.isfinite(a['pair'][j,k]).all(): raise ValueError('Pair half incomplete')
            for k, l in itertools.combinations([x for x in range(n) if x != j], 2):
                if not np.isfinite(a['triple'][j,k,l]).all(): raise ValueError('Triple half incomplete')
    return arrays, count


def centered(a): return a - a.mean(axis=-1, keepdims=True)


def moment(a, b):
    # Mean product over all unordered pair contrasts within each shared mapping.
    a, b = centered(np.asarray(a)), centered(np.asarray(b)); n = a.shape[-1]
    return float((a*b).mean() * (2*n/(n-1)))


def d1_statistics(y, p0, half_y, half_p0, rng, permutations=1000):
    ua, ub = half_y[0]-half_p0[0], half_y[1]-half_p0[1]
    ya, yb = half_y
    values = {'observed_contrast_variance': moment(y,y), 'observed_U_variance': moment(y-p0,y-p0),
              'cross_half_U_covariance': moment(ua,ub), 'cross_half_Y_covariance': moment(ya,yb),
              'half_U_A_variance': moment(ua,ua), 'half_U_B_variance': moment(ub,ub),
              'half_Y_A_variance': moment(ya,ya), 'half_Y_B_variance': moment(yb,yb)}
    a, b = centered(ua), centered(ub); null = np.empty(permutations)
    for i in range(permutations):
        permuted = rng.permuted(b, axis=1)
        null[i] = (a*permuted).mean() * (2*a.shape[-1]/(a.shape[-1]-1))
    return values, null


def d2_statistics(y, p0, n3):
    rows = []
    for index, (m, tau) in enumerate(itertools.product((3,5,7),(-1,1))):
        x = centered(n3[index*20:(index+1)*20]); denominator = float((x*x).sum())
        result = {'n_target':m,'tau':tau,'n_mappings':20,'fits':{}}
        for name, v in [('measured',y),('P0',p0),('U',y-p0)]:
            v = centered(v[index*20:(index+1)*20]); slope = float((x*v).sum()/denominator) if denominator else 0.
            error = float(((v-slope*x)**2).sum()); total = float((v*v).sum())
            slopes = [float((xx*vv).sum()/(xx*xx).sum()) if (xx*xx).sum()>0 else None for xx,vv in zip(x,v)]
            result['fits'][name] = {'pooled_within_mapping_slope':slope,'R2':1-error/total if total else None,
                                    'slope_by_mapping':slopes}
        rows.append(result)
    return rows


def finish_moments(values):
    v = dict(values)
    for name in ('U','Y'):
        denominator = np.sqrt(v[f'half_{name}_A_variance']*v[f'half_{name}_B_variance'])
        r = v[f'cross_half_{name}_covariance']/denominator if denominator else None
        v[f'{name}_half_correlation'] = r
        v[f'{name}_Spearman_Brown'] = 2*r/(1+r) if r is not None and r > -1 else None
    v['reliable_U_fraction_of_observed_contrast_variance'] = (v['cross_half_U_covariance']/v['observed_contrast_variance']
                                                            if v['observed_contrast_variance'] else None)
    return v


def run(base, extension):
    base, ext = Path(base), Path(extension); root = ext/'v31'; out = root/'diagnostics/D1_D2'
    sys.path[:0] = [str(ext),str(base/'code')]
    from common import load, sha, verify_bindings, write_new, atomic_status
    from analyze_v3 import read_closed
    from e4_simulation import predicted_caches, single_round_batch
    from run_e4_analysis import initial_state
    from e4_graphs import validate_sts
    score_plan = base/'setup/e4_score_queue_history/20261031T154754913456Z/plan.json'; plan = load(score_plan)
    manifest = base/'data/e4_cache_manifest.json'; sts_path = base/'data/e4_graphs/sts15_all80.json'
    source = {str(p):sha(p) for p in [score_plan,manifest,sts_path,Path(__file__),root/'research_plan_v31_user.md',
                                    base/'code/analyze_v3.py',base/'code/e4_simulation.py',base/'code/run_e4_analysis.py']}
    sources = dict(source)
    for job in plan['jobs']:
        for p in [job['input'],job['output'],job['output']+'.run.json',job['output']+'.summary.json']:
            sources[p] = sha(p)
        folder=Path(job['deferred_analysis']['output_dir']);tid=job['topic_id']
        point=folder/(tid+'_E4_analysis.json');cache=folder/(tid+'_caches.npz')
        if load(point)['cache_artifact']['sha256']!=sha(cache):raise ValueError('Original E4 cache binding changed')
        sources[str(point)]=sha(point);sources[str(cache)]=sha(cache)
    protocol = {'schema':'voices_v31_D1_D2_development_protocol_v1','source_sha256':sources,
                'scope':'original36 E4 tasks only; never G1/G2','seed':SEED,'bootstrap_draws':10000,'permutation_draws':1000,
                'split':'diagonal slot/label halves; baseline has only two labels, assigned disjointly by focal-id/stance parity',
                'reliable_variance_estimator':'cross-half pair-contrast covariance, without clipping; fraction denominator is full observed contrast second moment',
                'null':'permute half-B design labels separately within mapping and initial condition',
                'assumptions':'Deterministic slot/label variations are systematic perturbations, not independent random measurements. Spearman-Brown and covariance correction are conditional diagnostic estimates.',
                'D2':'within-mapping centered regression on n3, separated by k and tau; diagnostic topic fitting, not predictive confirmation'}
    write_new(out/'protocol.json',protocol)
    sts = load(sts_path)['systems']; records = []
    for ji, job in enumerate(plan['jobs']):
        target = out/job['model'].replace('/','--')/(job['topic_id']+'.json')
        if target.exists():
            record=load(target);verify_bindings(record['source_sha256']);records.append(record);continue
        atomic_status(out/'status.json',{'status':'running_original_E4_D1_D2','job':ji+1,'of':len(plan['jobs']),
                                        'model':job['model'],'topic':job['topic_id'],'GPU_used':False})
        halves,count=split_caches(read_closed(job['input'],job['output']))
        if count != 49020: raise ValueError('Original topic count differs')
        zero={k:0. for k in ('opponent_opponent','ally_ally','opponent_ally')}
        full={k:(halves[0][k]+halves[1][k])/2 for k in ('alpha','pair','triple')};full.update(n=15,five={})
        cached=Path(job['deferred_analysis']['output_dir'])/(job['topic_id']+'_caches.npz')
        with np.load(cached,allow_pickle=False) as old:
            for new_key,old_key in [('alpha','alpha'),('pair','pair'),('triple','measured')]:
                if not np.allclose(full[new_key],old[old_key],rtol=0,atol=1e-12,equal_nan=True):
                    raise ValueError('Half reconstruction differs from frozen original full E4 cache')
        predictions=[predicted_caches(c,zero) for c in [full,*halves]]
        seed=int(hashlib.sha256(('voices_v3_E4_mapping_v1|'+job['topic_id']).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed);mappings=[rng.permutation(15) for _ in range(20)]
        designs=[]
        for mapping in mappings:
            for system in sts:
                blocks=sorted(tuple(sorted(mapping[v] for v in b)) for b in system['blocks']);validate_sts(blocks);designs.append(blocks)
        blocks=np.array(designs);values=[{'measured':[],'P0':[]} for _ in predictions];n3=[]
        for m,tau in itertools.product((3,5,7),(-1,1)):
            state=initial_state(15,m,tau)
            target_bits=state==int(tau==1);number=target_bits[blocks].sum(axis=2)
            counts=np.stack([(number==i).sum(axis=1) for i in range(4)],axis=1)
            if not (np.all(counts[:,2]==m*(m-1)//2-3*counts[:,3]) and
                    np.all(counts[:,1]==m*(15-m)//2-counts[:,2]) and np.all(counts.sum(axis=1)==35)):
                raise ValueError('STS n3 composition identities fail')
            n3.extend(counts[:,3].reshape(20,80))
            for pred,destination in zip(predictions,values):
                for method in destination:
                    destination[method].extend(single_round_batch(pred[method],designs,state,tau).reshape(20,80))
        values=[{k:np.asarray(v) for k,v in item.items()} for item in values];n3=np.array(n3)
        d1,null=d1_statistics(values[0]['measured'],values[0]['P0'],[v['measured'] for v in values[1:]],
                              [v['P0'] for v in values[1:]],np.random.default_rng(SEED+ji))
        record={'model':job['model'],'topic_id':job['topic_id'],'D1_moments':d1,'D1':finish_moments(d1),
                'D1_permutation_null':null.tolist(),'D2':d2_statistics(values[0]['measured'],values[0]['P0'],n3),
                'source_sha256':{str(out/'protocol.json'):sha(out/'protocol.json')},'sealed_data_read':False}
        write_new(target,record);records.append(record)
    models=[];rng=np.random.default_rng(SEED);indices=rng.integers(0,6,size=(10000,6))
    for repo in dict.fromkeys(j['model'] for j in plan['jobs']):
        rows=[r for r in records if r['model']==repo]
        if len(rows)!=6:raise ValueError('Incomplete original six topics')
        mean={k:float(np.mean([r['D1_moments'][k] for r in rows])) for k in rows[0]['D1_moments']}
        covariance=np.array([r['D1_moments']['cross_half_U_covariance'] for r in rows])
        denominator=np.array([r['D1_moments']['observed_contrast_variance'] for r in rows])
        b_cov=covariance[indices].mean(axis=1);b_ratio=b_cov/denominator[indices].mean(axis=1)
        null=np.mean([r['D1_permutation_null'] for r in rows],axis=0)
        models.append({'model':repo,'topics':6,'D1':finish_moments(mean),
                       'cross_half_U_covariance_topic95CI':np.quantile(b_cov,[.025,.975]).tolist(),
                       'reliable_U_fraction_topic95CI':np.quantile(b_ratio,[.025,.975]).tolist(),
                       'permutation_one_sided_p':float((1+(null>=covariance.mean()).sum())/(len(null)+1))})
    result={'schema':'voices_v31_D1_D2_development_results_v1','status':'complete_development_diagnostics',
            'models':models,'original_tasks':36,'sealed_data_read':False,'confirmatory_claims':False,
            'limitations':protocol['assumptions'],'source_sha256':{str(out/'protocol.json'):sha(out/'protocol.json')}}
    for job in plan['jobs']:
        p=out/job['model'].replace('/','--')/(job['topic_id']+'.json');result['source_sha256'][str(p)]=sha(p)
    write_new(out/'report.json',result);atomic_status(out/'status.json',{'status':'complete_D1_D2_original36','GPU_used':False})
    print(json.dumps({'status':result['status'],'models':models}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();run(a.base,a.extension)
