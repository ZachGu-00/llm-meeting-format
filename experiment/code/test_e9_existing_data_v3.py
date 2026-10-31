"""CPU-only known-effect, pairing and exact empty-baseline semantics checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import e9_existing_data_v3 as e9


def panel(interactions=None):
    values=np.zeros((4,32,10,2))
    for i in range(4):
        for mask in range(32):
            linear=.2+sum((1,.7,1.3,-.5,-.6)[j] for j in range(5) if mask&(1<<j))
            for p in range(10):
                for label in range(2):
                    coefficient=0 if interactions is None else interactions(i,p,label)
                    pairs=sum(bool(mask&(1<<a) and mask&(1<<b)) for a,b in ((0,1),(0,2),(1,2)))
                    values[i,mask,p,label]=linear+coefficient*pairs
    return {'item_ids':['a','b','c','d'],'topics':['T0','T0','T1','T1'],'values':values,'readout':'R0',
            'empty_baseline':{'synthetic_fixture':True}}


class ExistingE9Tests(unittest.TestCase):
    def definitions(self):return e9.group_definitions(e9.OriginalPromptContract.permutations())

    def test_neutral_zero_denominator_is_undefined_not_forced_zero_or_excluded(self):
        source=panel();source['values'][:]=0
        result=e9.diagnose(source,source['item_ids'],self.definitions(),200,1,'synthetic')
        self.assertEqual(result['n_items'],4)
        for group in result['groups']:
            self.assertIsNone(group['C']);self.assertIsNone(group['C_minus_reference'])
            self.assertEqual(group['C_interval']['undefined_draws'],200)

    def test_additive_zero_and_known_C_match_original_estimator(self):
        for coefficient in (0.,.4,-.3):
            source=panel(lambda i,p,l:coefficient)
            y=source['values'].mean(axis=(2,3));num,den=e9.components(y)
            original=e9.base.sufficient({'y':y,'masks':list(range(32))})
            np.testing.assert_allclose(num,original['C_num'],atol=1e-14)
            np.testing.assert_allclose(den,original['C_den'],atol=1e-14)
            result=e9.diagnose(source,source['item_ids'],self.definitions(),200,1,'synthetic')
            for group in result['groups']:
                self.assertAlmostEqual(group['C'],coefficient,places=12)
                self.assertAlmostEqual(group['C_minus_reference'],0.,places=12)

    def test_single_label_artifact_is_preserved_and_paired_against_overall(self):
        source=panel(lambda i,p,l:float(l))
        result=e9.diagnose(source,source['item_ids'],self.definitions(),500,2,'synthetic')
        groups={g['id']:g for g in result['groups']}
        self.assertAlmostEqual(groups['overall']['C'],.5)
        self.assertAlmostEqual(groups['label_0']['C'],0)
        self.assertAlmostEqual(groups['label_1']['C'],1)
        np.testing.assert_allclose(groups['label_1']['paired_difference_interval']['percentile_95'],[.5,.5],atol=1e-12)
        self.assertTrue(result['bootstrap']['paired_weights_shared_across_all_groups'])

    def test_five_actual_slot_sets_and_all_ten_permutations_are_explicit(self):
        source=panel(lambda i,p,l:p/10)
        result=e9.diagnose(source,source['item_ids'],self.definitions(),200,1,'synthetic')
        self.assertEqual(result['actual_availability']['observed_opponent_slot_sets'],5)
        groups={g['id']:g for g in result['groups']}
        self.assertEqual(groups['opponent_slots_1_2_3']['permutations'],[0,7])
        self.assertAlmostEqual(groups['opponent_slots_1_2_3']['C'],.35)
        self.assertAlmostEqual(groups['permutation_09']['C'],.9)

    def test_matched_readout_difference_does_not_compare_different_item_compositions(self):
        r0=panel(lambda i,p,l:.1 if i<2 else 1.)
        r1=panel(lambda i,p,l:.12 if i<2 else 3.)
        r1['item_ids']=r1['item_ids'][:2];r1['topics']=r1['topics'][:2];r1['values']=r1['values'][:2];r1['readout']='R1'
        result=e9.paired_readouts(r0,r1,r0['item_ids'],200,1)
        self.assertEqual(result['n_items'],2)
        self.assertAlmostEqual(result['R0']['C'],.1)
        self.assertAlmostEqual(result['R1']['C'],.12)
        self.assertAlmostEqual(result['paired_R1_minus_R0'],.02)

    @staticmethod
    def closed_fixture(root):
        item={'item_id':'a','topic':'A fixed proposal','focal_stance':-1,'focal_statement':'I oppose the proposal.',
              'roles':{r:{'stance':1 if j<3 else -1,'argument':f'Fixed argument {r}.'} for j,r in enumerate(e9.ROLES)}}
        rows=[]
        for mask in range(32):
            for p in ([0] if mask==0 else range(10)):
                for label in (0,1):
                    rows.append({'id':f'm{mask}p{p}l{label}','item_id':'a','subset_mask':mask,'permutation':p,'label_map':label,
                        **e9.OriginalPromptContract.render(item,mask,p,label)})
        inp=root/'input.jsonl';inp.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        raw=root/'raw.jsonl';rp=Path(str(raw)+'.run.json')
        run={'readout':'R0','n_input':622,'input_file_sha256':e9.base.sha(inp)};rp.write_text(json.dumps(run))
        results=[r|{'R0_y_AB':float(r['subset_mask'].bit_count()),'status':'ok','run_file_sha256':e9.base.sha(rp)} for r in rows]
        raw.write_text(''.join(json.dumps(r)+'\n' for r in results))
        summary=run|{'status':'complete','n_output':622,'exact_id_and_count_equality':True,
            'output_sha256':e9.base.sha(raw),'output_bytes':raw.stat().st_size,'run_file_sha256':e9.base.sha(rp)}
        Path(str(raw)+'.summary.json').write_text(json.dumps(summary))
        return inp,raw,{'a':item}

    def test_empty_broadcast_requires_exact_source_builder_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inp,raw,items=self.closed_fixture(root)
            result=e9.load_panel(inp,raw,items,e9.OriginalPromptContract,'R0',root,{})
            self.assertEqual(result['physical_rows'],622)
            self.assertEqual(result['empty_baseline']['checked_empty_prompt_equivalences'],20)
            np.testing.assert_array_equal(result['values'][:,0],np.repeat(result['values'][:,0,0:1],10,axis=1))
            class ChangedEmpty(e9.OriginalPromptContract):
                @staticmethod
                def render(item,mask,p,label):
                    result=e9.OriginalPromptContract.render(item,mask,p,label)
                    if mask==0 and p>0:result['messages'][-1]['content']+=' Position-dependent empty meaning.'
                    return result
            with self.assertRaisesRegex(ValueError,'position-dependent empty'):
                e9.load_panel(inp,raw,items,ChangedEmpty,'R0',root,{})


if __name__=='__main__':unittest.main()
