import copy
import itertools
from pathlib import Path
import unittest
import numpy as np

import analyze_v3 as a
from e4_simulation import predicted_caches,sigmoid
from run_e4_analysis import e4a,e4b,e4c
from e4_rule_baselines_v3 import RULES,rule_deltas,local_rule_logits,triple_rule_caches,five_rule_outcomes,choose_e1_rule,evaluate_topic,method_statistics
from supplement_e4_rules_v3 import summarize_topics,finite_json,weighted_topic_means
from test_e4_aggregate_v3 import oracle_cache

ROOT=Path(__file__).resolve().parents[1]


class RuleBaselineTests(unittest.TestCase):
    def test_exact_E1_definitions_for_all_nonempty_role_subsets(self):
        v=np.array([-3.,2.,.7,.5,-.5])
        for mask in range(1,32):
            peers=[j for j in range(5) if mask>>j&1]
            current=rule_deltas(v[peers],[j<3 for j in peers])
            expected=a.rule_predictions(v[None,:],[mask])
            for name in RULES:self.assertAlmostEqual(current[name],expected[name][0,0])

    def test_global_orientation_conversion_and_signed_max_tie(self):
        values=np.array([-2.,2.,.6,-.4]);opponents=[True,True,False,False]
        positive=rule_deltas(values,opponents);negative=rule_deltas(-values,opponents)
        for rule in RULES:self.assertAlmostEqual(positive[rule],-negative[rule])
        self.assertAlmostEqual(positive['side_max'],-1.4)

    def test_three_peer_side_mean_reduces_to_mean_or_additive(self):
        values=[1.,3.]
        self.assertEqual(rule_deltas(values,[True,True])['side_mean'],2.)
        self.assertEqual(rule_deltas(values,[True,False])['side_mean'],4.)

    def test_four_peers_mean_and_alpha_once(self):
        alpha=np.full((5,2),2.);pair=np.full((5,5,4),4.)
        cache={'alpha':alpha,'pair':pair};state=np.zeros(5,dtype=int)
        logits=local_rule_logits(cache,0,[1,2,3,4],state)
        self.assertEqual(logits['present_mean'],4.);self.assertEqual(logits['additive'],10.)
        result=five_rule_outcomes(cache,list(range(5)),state,1)
        self.assertAlmostEqual(result['present_mean'],sigmoid(4.))

    def test_E1_selection_includes_additive_and_keeps_negative_scores(self):
        model={'point':{'rules':{rule:{'R2_within_item':-1.} for rule in RULES}}}
        self.assertEqual(choose_e1_rule(model)['rule'],'additive')
        model['point']['rules']['side_max']['R2_within_item']=-.2
        self.assertEqual(choose_e1_rule(model)['rule'],'side_max')
        for rule in RULES:model['point']['rules'][rule]['R2_within_item']=None
        self.assertIsNone(choose_e1_rule(model)['rule'])

    def test_averaging_oracle_has_P0_residual_but_is_exactly_pairwise_predictable(self):
        meta=a.read_json(ROOT/'data/e4_cache_manifest.json')['topics'][0]
        systems=a.read_json(ROOT/'data/e4_graphs/sts15_all80.json')['systems']
        cache=oracle_cache(meta);rules=triple_rule_caches(cache);cache['triple']=rules['present_mean']
        for group in meta['five_person_sets']:
            for bits in itertools.product((0,1),repeat=5):
                state=dict(zip(group,bits))
                for j in group:
                    peers=tuple(sorted(k for k in group if k!=j));key=(j,peers,(state[j],*(state[k] for k in peers)))
                    cache['five'][key]=local_rule_logits(cache,j,peers,state)['present_mean']
        kernels=dict.fromkeys(a.PAIRS,0.);primary=predicted_caches(cache,kernels)
        b,_=e4b(cache,primary,systems,meta['topic_id'],n_mappings=2)
        result={'E4a':e4a(cache,primary,meta),'E4b':b,'E4c':e4c(cache,primary,kernels,meta)}
        output=evaluate_topic(cache,result,meta,systems);topic={}
        for part,values in output.items():
            np.testing.assert_allclose(values['present_mean'],values['measured'],atol=1e-14)
            exact=method_statistics(part,values['measured'],values['present_mean'])
            additive=method_statistics(part,values['measured'],values['additive'])
            self.assertLess(exact['error_MS'],1e-27);self.assertGreater(additive['error_MS'],1e-9)
            topic[part]={name:method_statistics(part,values['measured'],pred) for name,pred in values.items() if name!='measured'}
        summaries,_=summarize_topics([topic,copy.deepcopy(topic)],np.array([[1,1],[2,0],[0,2]]),'present_mean')
        for part in summaries:
            self.assertAlmostEqual(summaries[part]['methods']['present_mean']['R2_delta']['estimate'],1.)
            # PK equals additive in this oracle, so its improvement over exact averaging must be negative.
            self.assertLess(summaries[part]['PK_improvement_over_each_rule']['present_mean']['contrast_MSE_improvement']['estimate'],0.)
        import json
        json.dumps(finite_json(summaries),allow_nan=False)

    def test_undefined_topic_metric_only_invalidates_draws_containing_it(self):
        result=weighted_topic_means(np.array([[2,0],[1,1],[0,2]]),np.array([.5,np.nan]))
        self.assertAlmostEqual(result[0],.5);self.assertTrue(np.isnan(result[1:]).all())


if __name__=='__main__':unittest.main()
