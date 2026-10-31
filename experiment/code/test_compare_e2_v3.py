import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import analyze_v3 as a
import compare_e2_v3 as c
from test_v3_analysis import panel_for


def named_fixture():
    panel=panel_for('additive')
    for mask in range(32):
        n=(mask&7).bit_count();m=(mask&24).bit_count()
        panel['y'][:,mask]=n-m+2*n*m+np.array([0.,.2,-.3])
    analysis=a.analyze(panel,draws=16)
    return panel,c.reconstruct_named(analysis,panel['item_ids'],panel['topics'])


class PairedDiagnosticTests(unittest.TestCase):
    def test_reconstructs_full_and_low_order_observations(self):
        panel=panel_for('mean');analysis=a.analyze(panel,draws=16)
        rebuilt=c.reconstruct_named(analysis,panel['item_ids'],panel['topics'])
        np.testing.assert_allclose(rebuilt['y'],panel['y'],atol=1e-12)
        low={**panel,'masks':sorted(a.C_MASKS)};low_analysis=a.analyze(low,draws=16)
        rebuilt=c.reconstruct_named(low_analysis,panel['item_ids'],panel['topics'])
        np.testing.assert_allclose(rebuilt['y'][:,low['masks']],panel['y'][:,low['masks']],atol=1e-12)
        with self.assertRaises(ValueError):c.reconstruct_named(analysis,panel['item_ids'],['wrong','t2','t3'])

    def test_identical_conditions_have_exact_zero_paired_interval(self):
        panel,named=named_fixture();context=c.PairedTopics(panel['topics'],128,7)
        left=c.named_metrics(named,context);right=c.named_metrics(named,context)
        for key in left:
            result=c.summarize_metric(c.difference(left[key],right[key]))
            self.assertEqual(result['estimate'],0.);self.assertEqual(result['interval95'],[0.,0.])

    def test_cluster_ratios_preserve_all_items_and_joint_weights(self):
        context=c.PairedTopics(['topicA','topicA','topicB'],32,4)
        numerator=np.array([1.,2.,9.]);denominator=np.array([1.,3.,2.])
        point,draws=context.ratio(numerator,denominator)
        self.assertAlmostEqual(point,2.)
        expected=(context.weights@np.array([3.,9.]))/(context.weights@np.array([4.,2.]))
        np.testing.assert_allclose(draws,expected)

    def test_adding_ally_keeps_positive_and_negative_changes(self):
        panel,named=named_fixture();metrics=c.named_metrics(named,c.PairedTopics(panel['topics'],32,5))
        self.assertAlmostEqual(metrics['add_ally_0_0_to_1'][0],-1.)
        self.assertAlmostEqual(metrics['add_ally_1_0_to_1'][0],1.)
        self.assertAlmostEqual(metrics['add_ally_3_1_to_2'][0],5.)

    def test_prefix_comparison_uses_same_role_cells_not_count_average(self):
        panel=panel_for('additive');analysis=a.analyze(panel,draws=16)
        named=c.reconstruct_named(analysis,panel['item_ids'],panel['topics']);prefix=c.prefix_panel(named)
        col=prefix['conditions'].index((1,0))
        np.testing.assert_allclose(prefix['delta'][:,col],np.array([1.,2.,.5]))
        metrics=c.count_metrics(prefix,c.PairedTopics(panel['topics'],32,6))
        self.assertNotIn('C',metrics);self.assertIn('count_grouping_contrast',metrics)

    def test_nonpositive_power_draws_are_not_dropped(self):
        conditions=[(0,0),(1,0),(2,0),(3,0)]
        panel={'conditions':conditions,'delta':np.array([[0.,1.,2.,3.],[0.,-.5,-1.,-1.5]])}
        metrics=c.count_metrics(panel,c.PairedTopics(['a','b'],128,8))
        result=c.summarize_metric(metrics['power_t_native_range'])
        self.assertAlmostEqual(result['estimate'],1.)
        self.assertGreater(result['undefined_draws'],0);self.assertIsNone(result['interval95'])

    def test_full_paired_cohort_is_json_safe_and_excludes_native_range_mismatch(self):
        panel,named=named_fixture();names={'E1':named,'E2b':named,'E2c':named}
        low_panel={**panel,'masks':sorted(a.C_MASKS)};low=c.reconstruct_named(a.analyze(low_panel,draws=16),panel['item_ids'],panel['topics'])
        names.update({'E2a_neutral'+str(n):low for n in range(4)})
        prefix=c.prefix_panel(named)
        grid=list(itertools.product(range(7),range(4)))
        extended={'conditions':grid,'delta':np.array([[n-m+2*n*m for n,m in grid]]*3),'scope':'synthetic nine-role test only'}
        result=c.compare_cohort(names,{'E2d':prefix,'E2e':extended},panel['item_ids'],panel['topics'],set(panel['item_ids']),32,10)
        json.dumps(result,allow_nan=False)
        self.assertNotIn('power_t_native_range',result['paired_count_grid_differences']['E2e_minus_E1_matched_prefix'])
        self.assertEqual(result['paired_named_peer_differences']['E2b_minus_E1']['C']['interval95'],[0.,0.])

    def test_closed_source_binding_rejects_modified_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.jsonl';raw=root/'output.jsonl'
            source.write_text('{"id":1}\n{"id":2}\n');raw.write_text('{"id":1,"y":1}\n{"id":2,"y":2}\n')
            run={'n_input':2,'input_file_sha256':a.sha(source),'readout':'R0','engine':{'test_only':True},'batch_size':16,'calibration_file_sha256':'test-only','option_ids':{'A':1,'B':2}}
            rp=Path(str(raw)+'.run.json');rp.write_text(json.dumps(run))
            summary={**run,'status':'complete','exact_id_and_count_equality':True,'n_output':2,'output_sha256':a.sha(raw),'output_bytes':raw.stat().st_size,'run_file_sha256':a.sha(rp)}
            Path(str(raw)+'.summary.json').write_text(json.dumps(summary))
            analysis={'readout':'R0','bootstrap':{'draws':10000,'unit':'topic'},'file_sha256':{str(source):a.sha(source),str(raw):a.sha(raw)},'execution_identity':{'runs':[{'path':str(raw),'run_sha256':a.sha(rp),'engine':run['engine'],'batch_size':16,'calibration_sha256':'test-only'}]}}
            ap=root/'analysis.json';ap.write_text(json.dumps(analysis));v=c.Verifier(root,'/unused-test-root')
            c.closed_analysis(ap,v,source,2)
            raw.write_text(raw.read_text()+'changed\n')
            with self.assertRaises(ValueError):c.closed_analysis(ap,v,source,2)


if __name__=='__main__':unittest.main()
