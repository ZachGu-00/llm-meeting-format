"""CPU checks for kernel propagation and the within-mapping E4 estimand."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import analyze_v3 as a
from aggregate_e4_v3 import kernel_bootstrap,statistics,compile_topic,topic_aggregate,metric_values
from e4_simulation import predicted_caches,save_cache,load_cache
from run_e4_analysis import e4a,e4b,e4c
from e4_dynamic_stratified import combine_independent_estimates,simulate_sts_stratified,simulate_pasch

ROOT=Path(__file__).resolve().parents[1]


def kernel_analysis():
    v=np.array([1.,1.,1.,-1.,-1.]);y=[]
    for theta in (.1,.2,.3):
        row=[]
        for mask in range(32):
            peers=[j for j in range(5) if mask>>j&1]
            value=v[peers].sum()
            for relation,pairs in a.PAIRS.items():
                scale={'opponent_opponent':1,'opponent_ally':2,'ally_ally':-1}[relation]
                value+=sum(theta*scale for j,k in pairs if j in peers and k in peers)
            row.append(value)
        y.append(row)
    return a.analyze({'y':np.array(y),'masks':list(range(32)),'item_ids':['a','b','c'],
                      'topics':['x','y','z'],'n_physical_rows':1,'q_median':1.},draws=16)


def oracle_cache(meta):
    n=15;rng=np.random.default_rng(7);alpha=rng.normal(size=(n,2));pair=np.zeros((n,n,4))
    for j,k in itertools.permutations(range(n),2):
        for sj,sk in itertools.product((0,1),repeat=2):
            pair[j,k,2*sj+sk]=alpha[j,sj]+rng.normal()
    cache={'n':n,'alpha':alpha,'pair':pair,'triple':np.zeros((n,n,n,8)),'five':{}}
    observed=dict(zip(a.PAIRS,[.4,.1,-.2]));cache['triple']=predicted_caches(cache,observed)['PK']
    for group in meta['five_person_sets']:
        for bits in itertools.product((0,1),repeat=5):
            state=dict(zip(group,bits))
            for j in group:
                peers=tuple(sorted(k for k in group if k!=j));sj=state[j]
                vs={k:pair[j,k,2*sj+state[k]]-alpha[j,sj] for k in peers}
                value=alpha[j,sj]+sum(vs.values())+.1*(sum(bits)-2)
                for k,l in itertools.combinations(peers,2):
                    rel='opponent_opponent' if state[k]!=sj and state[l]!=sj else 'ally_ally' if state[k]==sj and state[l]==sj else 'opponent_ally'
                    value+=(1-2*sj)*observed[rel]*(abs(vs[k])+abs(vs[l]))/2
                cache['five'][j,peers,(sj,*(state[k] for k in peers))]=value
    return cache


class PropagationTests(unittest.TestCase):
    def test_kernel_covariance_and_ratio_reconstruction(self):
        point,samples,weights,topics=kernel_bootstrap(kernel_analysis(),128,17)
        np.testing.assert_allclose(point,[.2,.4,-.2],atol=1e-12)
        np.testing.assert_allclose(samples[:,1],2*samples[:,0],atol=1e-12)
        np.testing.assert_allclose(samples[:,2],-samples[:,0],atol=1e-12)
        self.assertGreater(samples[:,0].std(),0)
        np.testing.assert_array_equal(weights.sum(axis=1),3)

    def test_within_mapping_contrasts_survive_cancelling_mapping_means(self):
        y=np.array([[[0.,1.,2.,3.],[3.,2.,1.,0.]]]);p0=.5*y;pk=.75*y
        stats=statistics('E4b',y,p0,pk[...,None])
        self.assertEqual(float(y.mean(axis=1).std()),0.)
        self.assertGreater(stats['observed_mean_square_contrast'][0],0.)
        pairs=np.triu_indices(4,1);differences=[]
        for mapping in y[0]:differences.extend((mapping[:,None]-mapping[None,:])[pairs])
        self.assertAlmostEqual(stats['observed_mean_square_contrast'][0],np.mean(np.square(differences)))
        shifted=statistics('E4b',y+1000,p0+2000,pk[...,None]-5000)
        for key in stats:np.testing.assert_allclose(stats[key],shifted[key],atol=1e-12)

    def test_topic_ratios_recomputed_instead_of_averaging_ratios(self):
        topics=[{'observed_mean_square_contrast':1.,'P0_mean_square_error':1.,'PK_mean_square_error':.5},
                {'observed_mean_square_contrast':9.,'P0_mean_square_error':1.,'PK_mean_square_error':.5}]
        metric=metric_values(topic_aggregate(topics,np.array([[1,1],[2,0]])))
        np.testing.assert_allclose(metric['P0_unexplained_fraction_U'],[.2,1.])

    def test_compiler_reproduces_all_three_experiments_and_cache_roundtrip(self):
        meta=a.read_json(ROOT/'data/e4_cache_manifest.json')['topics'][0]
        systems=a.read_json(ROOT/'data/e4_graphs/sts15_all80.json')['systems']
        cache=oracle_cache(meta);kernels=dict(zip(a.PAIRS,[.3,-.2,.1]));pred=predicted_caches(cache,kernels)
        b,_=e4b(cache,pred,systems,meta['topic_id'],n_mappings=2)
        result={'E4a':e4a(cache,pred,meta),'E4b':b,'E4c':e4c(cache,pred,kernels,meta)}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'cache.npz';save_cache(path,cache,pred);restored=load_cache(path)
            self.assertEqual(restored['five'],cache['five'])
            compiler,sections,expected=compile_topic(restored,result,meta,systems)
            values=compiler.evaluate([[0.,0.,0.],list(kernels.values())])
            for part,(sl,shape) in sections.items():
                computed=values[sl].reshape(*shape,2)
                np.testing.assert_allclose(computed[...,0],expected[part]['P0'],atol=2e-14)
                np.testing.assert_allclose(computed[...,1],expected[part]['PK'],atol=2e-14)
                stats=statistics(part,expected[part]['measured'],expected[part]['P0'],computed[...,1,None])
                old=result[part]['summary']
                self.assertAlmostEqual(stats['observed_mean_square_contrast'][0],old['observed_delta_square_sum']/old['n_contrasts'])
                self.assertAlmostEqual(stats['PK_mean_square_error'][0],old['PK_error_square_sum']/old['n_contrasts'])

    def test_independent_mapping_SE_is_not_mapping_variation(self):
        result=combine_independent_estimates([
            {'means':[0.],'Monte_Carlo_standard_errors':[.3]},
            {'means':[1.],'Monte_Carlo_standard_errors':[.4]}])
        self.assertAlmostEqual(result['means'][0],.5)
        self.assertAlmostEqual(result['Monte_Carlo_standard_errors'][0],.25)
        self.assertNotAlmostEqual(result['between_mapping_SD_of_estimated_means'][0],.25)

    def test_stratified_budget_and_paired_null(self):
        systems=a.read_json(ROOT/'data/e4_graphs/sts15_all80.json')['systems'][:2]
        mappings=[list(range(15)),list(reversed(range(15)))];cache=np.zeros((15,15,15,8))
        result=simulate_sts_stratified({'measured':cache,'P0':cache,'PK':cache},systems,mappings,
            'test-only',3,1,total_draws=16,rounds=1,chunk_size=4)
        self.assertEqual(result['draws'],8);self.assertEqual(result['total_draws_per_design'],16)
        self.assertEqual(len(set(result['mapping_seeds'])),2)
        self.assertTrue(np.all(np.array(result['paired_residual_U_measured_minus_other_methods']['means'])==0))
        self.assertTrue(np.all(np.array(result['paired_residual_U_measured_minus_other_methods']['Monte_Carlo_standard_errors'])==0))

    def test_G6_dynamics_excludes_nine_irrelevant_agents(self):
        meta=a.read_json(ROOT/'data/e4_cache_manifest.json')['topics'][0]
        cache=np.full((15,15,15,8),1000.)
        result=simulate_pasch({'measured':cache,'P0':cache,'PK':cache},meta,draws=4,rounds=1,chunk_size=4)
        self.assertEqual(result['n_conditions'],70);self.assertEqual(result['outcome_denominator'],6)
        for row in result['records']:
            self.assertEqual(row['simulation']['n_agents'],6)
            expected=1. if row['tau']==1 else 0.
            np.testing.assert_array_equal(np.array(row['simulation']['outcomes']['means']),expected)


if __name__=='__main__':unittest.main()
