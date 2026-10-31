"""Independent algebra, shared-bootstrap and actual-renderer diagnostic checks."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import analyze_e9_affine_pairs_v3 as a
import summarize_e9_affine_pairs_v3 as summary

ROOT=Path(__file__).resolve().parents[1]


def panel(scales,numerators):
    scales=np.asarray(scales,dtype=float);single=np.repeat(scales[:,None],5,axis=1)
    pairs=2*np.repeat(scales[:,None],10,axis=1)
    pairs[:,a.RELATIONS['opponent_opponent']]+=np.asarray(numerators)[:,None]
    return {'alpha':np.zeros(len(scales)),'single_y':single,'pair_y':pairs}


class AffineAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.items,cls.render,_,_=a.verify_materials(ROOT,ROOT/'data/e9_affine_balance/manifest.json')

    def test_direct_pair_formula_matches_expanded_three_opponent_contrasts(self):
        rng=np.random.default_rng(10);alpha=rng.normal(size=7);single=rng.normal(size=(7,5));pairs=rng.normal(size=(7,10))
        result=a.components(alpha,single,pairs)
        def m(j,k):
            idx=a.PAIR_ROLES.index(tuple(sorted((j,k))))
            return pairs[:,idx]-single[:,j]-single[:,k]+alpha
        expanded=np.mean([m(j,k)+m(3,4)-.5*(m(j,3)+m(k,4)+m(j,4)+m(k,3))
            for j,k in itertools.combinations(range(3),2)],axis=0)
        np.testing.assert_allclose(result['numerators'][:,0],expanded,atol=1e-14)
        np.testing.assert_allclose(result['denominators'][:,0],np.abs(single[:,:3]-alpha[:,None]).mean(axis=1))

    def test_stance_blind_geometry_toy_and_arbitrary_pair_position_term(self):
        proof=a.geometry_proof(self.render)
        for name,want in [('old10',.8),('additional10',-.8),('combined20',0.)]:
            self.assertAlmostEqual(proof['panels'][name]['C_per_unit_lambda'],want)
        orders=self.render.permutations()+[a.materials.affine_order(x,b) for x in (2,3) for b in range(5)]
        geometry=np.random.default_rng(2).normal(size=(5,5))
        pairs=np.array([[2+np.mean([geometry[p.index(a.materials.ROLES[j]),p.index(a.materials.ROLES[k])] for p in orders])
            for j,k in a.PAIR_ROLES]])
        self.assertAlmostEqual(a.components(np.zeros(1),np.ones((1,5)),pairs)['numerators'][0,0],0.)

    def test_additive_role_position_effect_cancels_without_false_confound(self):
        effects=np.random.default_rng(3).normal(size=(5,5))
        for orders in [self.render.permutations(),[a.materials.affine_order(x,b) for x in (2,3) for b in range(5)]]:
            single=np.array([[np.mean([effects[j,p.index(a.materials.ROLES[j])] for p in orders]) for j in range(5)]])
            pairs=np.array([[np.mean([effects[j,p.index(a.materials.ROLES[j])]+effects[k,p.index(a.materials.ROLES[k])] for p in orders]) for j,k in a.PAIR_ROLES]])
            np.testing.assert_allclose(a.components(np.zeros(1),single,pairs)['pair_interactions'],0,atol=1e-15)

    def test_genuine_relation_component_can_survive_balancing(self):
        old=panel([1.,1.],[.4,.4]);new=old['pair_y'].copy()
        result,_,_,_,_=a.paired_summary({'R0':old},{'R0':new},['x','y'],draws=64)
        for name in a.PANELS:self.assertAlmostEqual(result['R0']['panels'][name]['C']['estimate'],.4)
        self.assertEqual(result['R0']['paired_differences']['combined20_minus_old10']['C']['estimate'],0.)

    def test_ratio_of_sums_joint_resampling_and_readout_pairing(self):
        scales=np.array([1.,2.,4.]);old=panel(scales,[.1,.2,1.]);new=old['pair_y'].copy()
        new[:,a.RELATIONS['opponent_opponent']]+=.3*scales[:,None]
        second={k:1.7*v for k,v in old.items()};new_second=1.7*new
        result,items,cross,draws,topics=a.paired_summary({'R0':old,'R1':second},{'R0':new,'R1':new_second},['x','y','z'],draws=1000)
        self.assertAlmostEqual(result['R0']['panels']['old10']['C']['estimate'],1.3/7)
        self.assertNotAlmostEqual(1.3/7,np.mean(np.array([.1,.2,1.])/scales))
        self.assertGreater(np.std(draws['R0_old10_metrics'][:,0]),.01)
        np.testing.assert_allclose(draws['R0_additional10_minus_old10_metrics'][:,0],.3,atol=1e-14)
        np.testing.assert_allclose(draws['R0_combined20_minus_old10_metrics'][:,0],.15,atol=1e-14)
        np.testing.assert_allclose(draws['old10_R1_minus_R0_metrics'][:,0],0,atol=1e-14)
        self.assertEqual(items['R0']['panels']['old10']['metric_denominators'],items['R0']['panels']['additional10']['metric_denominators'])

    def test_undefined_denominator_draws_and_negative_changes_retained(self):
        old=panel([0.,1.],[1.,.2]);new=old['pair_y'].copy();new[1,a.RELATIONS['opponent_opponent']]-=1
        result,_,_,_,_=a.paired_summary({'R1':old},{'R1':new},['x','y'],draws=1000)
        value=result['R1']['panels']['old10']['C']
        self.assertGreater(value['undefined_draws'],0);self.assertIsNone(value['interval95'])
        delta=result['R1']['paired_differences']['additional10_minus_old10']['C']
        self.assertAlmostEqual(delta['estimate'],-1.);self.assertGreater(delta['undefined_draws'],0)

    def additional_rows(self,omit_last=False):
        count=0
        for item,mask,x,b,label in itertools.product(self.items,a.materials.PAIR_MASKS,(2,3),range(5),(0,1)):
            count+=1
            if omit_last and count==4000:return
            row=a.materials.additional_row(self.render,item,mask,x,b,label)
            j,k=[z for z in range(5) if mask>>z&1];order=row['slot_role_order']
            value=2+float(abs(order.index(a.materials.ROLES[j])-order.index(a.materials.ROLES[k]))==1)
            sign=1 if row['positive_letter']=='A' else -1
            row.update(R0_y_AB=value*sign,R1_y_AB=2*value*sign,q=.8)
            yield row

    def test_new4000_renderer_alignment_orientation_and_missing_label(self):
        with patch.object(a,'run_contract',return_value={'readout':'R0_R1'}),patch.object(a.io,'read_closed',side_effect=lambda *args:self.additional_rows()):
            pairs,_,_=a.load_additional('unused','unused',['R0','R1'],self.items,self.render,{})
        expected=a.geometry_proof(self.render)['panels']['additional10']['pair_linear_adjacency_probabilities']
        np.testing.assert_allclose(pairs['R0'],np.tile(2+np.array(expected),(20,1)))
        np.testing.assert_allclose(pairs['R1'],2*pairs['R0'])
        with patch.object(a,'run_contract',return_value={'readout':'R0_R1'}),patch.object(a.io,'read_closed',side_effect=lambda *args:self.additional_rows(True)):
            with self.assertRaisesRegex(ValueError,'Missing'):a.load_additional('unused','unused',['R0','R1'],self.items,self.render,{})
        with patch.object(a,'run_contract',return_value={'readout':'R0_R1'}):
            with self.assertRaisesRegex(ValueError,'omit'):a.load_additional('unused','unused',['R1'],self.items,self.render,{})

    def test_all_original_duplicate_singletons_are_averaged(self):
        def rows():
            for item in self.items:
                for mask in range(32):
                    for p in ([0] if mask==0 else range(10)):
                        for label in (0,1):
                            if mask not in a.materials.REUSE_MASKS+a.PAIR_MASKS:
                                yield {'item_id':item['item_id'],'subset_mask':mask};continue
                            row=self.render.render(item,mask,p,label)
                            # Deliberately distinct duplicate scores; the entire old mean must survive.
                            value=mask.bit_count()+.01*p
                            row.update(R0_y_AB=value*(1 if row['positive_letter']=='A' else -1),q=.9)
                            yield row
        with patch.object(a,'run_contract',return_value={'readout':'R0'}),patch.object(a.io,'read_closed',side_effect=lambda *args:rows()):
            value=a.load_original('unused','unused','R0',self.items,self.render,{},124400)
        np.testing.assert_allclose(value['single_y'],1.045)
        np.testing.assert_allclose(value['pair_y'],2.045);np.testing.assert_array_equal(value['alpha'],0.)
        self.assertEqual(value['retained_low_order_physical_rows'],6040)

    def test_cross_model_report_keeps_missing_models_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            report=summary.build([],Path(directory),['fixture/A','fixture/B'])
            self.assertEqual(report['status'],'partial_or_pending');self.assertEqual(report['n_verified_models'],0)
            self.assertIn('待完成',summary.markdown(report))

    def test_actual_closed_row_contract_then_source_output_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);source=path/'input.jsonl';raw=path/'result.jsonl'
            row={'id':'synthetic_one','messages':[]}
            source.write_text(json.dumps(row)+'\n',encoding='utf-8')
            run={'readout':'R1','n_input':1,'input_file_sha256':a.io.sha(source)}
            run_path=Path(str(raw)+'.run.json');run_path.write_text(json.dumps(run),encoding='utf-8')
            scored={**row,'R1_y_AB':1.,'status':'ok','run_file_sha256':a.io.sha(run_path)}
            raw.write_text(json.dumps(scored)+'\n',encoding='utf-8')
            receipt={**run,'status':'complete','n_output':1,'exact_id_and_count_equality':True,
                'output_bytes':raw.stat().st_size,'output_sha256':a.io.sha(raw),'run_file_sha256':a.io.sha(run_path)}
            Path(str(raw)+'.summary.json').write_text(json.dumps(receipt),encoding='utf-8')
            a.run_contract(raw,['R1'],1,{})
            self.assertEqual(len(list(a.io.read_closed(source,raw))),1)
            scored['R1_y_AB']=9.;raw.write_text(json.dumps(scored)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed'):list(a.io.read_closed(source,raw))

    def test_cross_model_verified_receipt_and_tampered_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);source=path/'synthetic_source.txt';source.write_text('fixture only')
            bootstrap=path/'diagnostic_bootstrap.npz';np.savez(bootstrap,fixture=np.zeros(1))
            old=panel(np.ones(20),np.full(20,.2));estimates,_,cross,_,_=a.paired_summary({'R1':old},{'R1':old['pair_y']},['a']*10+['b']*10,draws=10000)
            record={'schema':'voices_v3_E9_affine_pair_diagnostic_v1','status':'complete_diagnostic','repo_id':'fixture/A',
                'n_items':20,'n_topics':2,'item_ids':[item['item_id'] for item in self.items],
                'readouts':['R1'],'scope':{'fresh_confirmation':False},'main_E1_E2_selection_and_PK_unchanged':True,
                'bootstrap':{'draws':10000,'unit':'topic','shared_weights_across_panels_and_readouts':True},
                'estimates':estimates,'paired_readout_comparisons':cross,
                'file_sha256':{str(source):a.io.sha(source)},'bootstrap_artifact':{'filename':bootstrap.name,'sha256':a.io.sha(bootstrap)}}
            result=path/'diagnostic.json';result.write_text(json.dumps(record),encoding='utf-8')
            full=summary.build([result],path,['fixture/A']);self.assertEqual(full['status'],'complete_requested_diagnostic_panel')
            partial=summary.build([result],path,['fixture/A','fixture/B']);self.assertEqual(partial['n_verified_models'],1)
            bootstrap.write_bytes(b'changed')
            invalid=summary.build([result],path,['fixture/A']);self.assertEqual(invalid['n_verified_models'],0)
            self.assertEqual(len(invalid['input_errors']),1)


if __name__=='__main__':unittest.main()
