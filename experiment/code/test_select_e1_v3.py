import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_select_e1_v3 as s


def row(repo,c,architecture='attention',family=None,complete=True,eligible=True,flag=False):
    return {'repo_id':repo,'E1_complete':complete,'E1_status':'complete_verified' if complete else 'incomplete',
            'E4_eligible':eligible,'R0_readout_disagreement_flag':flag,
            'structure':{'class':architecture,'family':family or repo.split('/')[0].lower()},
            'primary':{'C':c,'H1':{'C_left_tail_bootstrap_p':.01,'C_equivalence_bootstrap_p':.02}}}


def panel():
    return [row(s.MANDATORY[0],-.4,'linear_and_full_attention'),
            row(s.MANDATORY[1],.02,'convolution_and_attention'),
            *[row('publisher'+str(i)+'/model',c) for i,c in enumerate([-.9,-.1,.3,.6,.8,1.2])]]


class SelectionTests(unittest.TestCase):
    def test_absolute_quantiles_and_mandatory_models(self):
        result=s.select_six(panel())
        self.assertEqual(result['selected_repos'][:2],list(s.MANDATORY))
        self.assertEqual(result['steps'][2]['signed_C'],-.1)
        self.assertEqual(result['steps'][-1]['signed_C'],1.2)
        self.assertEqual(len(set(result['selected_repos'])),6)
        self.assertEqual(result['quantile_probabilities'],[0.,1/3,2/3,1.])

    def test_ties_prefer_structure_then_publisher_then_fixed_hash(self):
        rows=panel()[:2]+[row('Qwen/extra',.5,'linear_and_full_attention'),
              row('Qwen/dense',.5),row('new/a',.5),row('new/b',.5),row('another/a',.5)]
        expected=min(['new/a','new/b','another/a'],key=lambda repo:s.hashlib.sha256((s.HASH_PREFIX+repo).encode()).hexdigest())
        self.assertEqual(s.select_six(rows)['selected_repos'][2],expected)
        self.assertEqual(s.select_six(rows)['selected_repos'],s.select_six(list(reversed(rows)))['selected_repos'])

    def test_generalized_fallback_and_disagreement_flag(self):
        rows=panel()[2:];rows[0]['R0_readout_disagreement_flag']=True
        result=s.select_six(rows)
        self.assertEqual(len(result['quantile_probabilities']),6)
        self.assertEqual(result['unavailable_mandatory_models'],list(s.MANDATORY))
        self.assertTrue(result['selected_R0_readout_disagreement_flags'][rows[0]['repo_id']])

    def test_ineligible_undefined_and_R1_only_models_not_selected(self):
        rows=panel()+[row('r1/only',None,eligible=False)]
        result=s.select_six(rows)
        self.assertNotIn('r1/only',result['selected_repos'])
        with self.assertRaises(ValueError):s.select_six(rows[:5])

    def test_Holm_has_only_two_prespecified_hypotheses(self):
        rows=panel();rows[1]['E1_complete']=False
        result=s.h1_family(rows)
        self.assertEqual(result['family_size'],2)
        self.assertEqual(set(result['tests']),set(s.MANDATORY))
        self.assertAlmostEqual(result['tests'][s.MANDATORY[0]]['Holm_adjusted_p'],.02)
        self.assertIsNone(result['tests'][s.MANDATORY[1]]['Holm_adjusted_p'])

    def test_official_config_classes(self):
        expected={'Qwen--Qwen3.5-4B':'linear_and_full_attention',
                  'LiquidAI--LFM2.5-1.2B-Instruct':'convolution_and_attention',
                  'ibm-granite--granite-4.0-h-micro':'state_space_and_attention',
                  'google--gemma-4-E2B-it':'attention'}
        for slug,kind in expected.items():
            config=s.e0.read(s.ROOT/'provenance/architecture_configs'/(slug+'.json'))
            self.assertEqual(s.architecture_class(config)['class'],kind)

    def test_no_selection_without_flag_or_with_incomplete_panel(self):
        rows=panel();models=[r['repo_id'] for r in rows]
        base={'models':rows,'summarizer_sha256':'test-only','protocol_versions':{}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'setup').mkdir();(root/'setup/download_queue.json').write_text(json.dumps([{'repo_id':r} for r in models]))
            (root/'execution_protocol.txt').write_text('test-only')
            lookup={r['repo_id']:r for r in rows}
            with patch.object(s.e0,'build',return_value=base),patch.object(s,'summarize_model',side_effect=lambda root,repo,*args:copy.deepcopy(lookup[repo])):
                with patch.object(s,'select_six',side_effect=AssertionError('Selection was not authorized')):
                    result=s.build(root,models=models,freeze=False)
                    self.assertEqual(result['E4_selection']['status'],'not_requested')
                    lookup[models[-1]]['E1_complete']=False
                    result=s.build(root,models=models,freeze=True)
                    self.assertEqual(result['E4_selection']['status'],'blocked_incomplete_requested_E1')
                lookup[models[-1]]['E1_complete']=True
                result=s.build(root,models=models,freeze=True)
                self.assertEqual(result['E4_selection']['status'],'frozen')
                json.dumps(result,allow_nan=False)

    def test_filter_cannot_hide_operational_or_pending_mandatory(self):
        rows=panel();models=[r['repo_id'] for r in rows[2:]];lookup={r['repo_id']:r for r in rows}
        global_e0=[]
        for r in rows:
            global_e0.append({**r,'closed_development100':True,'integrity':{'status':'verified_local'},
                              'qualification':{'R0_operational':'pass' if r['repo_id']==s.MANDATORY[0] else 'failed'}})
        base={'models':global_e0,'summarizer_sha256':'test-only','protocol_versions':{}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'setup').mkdir();(root/'setup/download_queue.json').write_text(json.dumps([{'repo_id':r['repo_id']} for r in rows]))
            (root/'execution_protocol.txt').write_text('test-only')
            with patch.object(s.e0,'build',return_value=base),patch.object(s,'summarize_model',side_effect=lambda root,repo,*args:copy.deepcopy(lookup[repo])):
                report=s.build(root,models=models,freeze=True)
                self.assertIn(s.MANDATORY[0],report['requested_models']);self.assertNotIn(s.MANDATORY[1],report['requested_models'])
                self.assertEqual(report['E4_selection']['status'],'frozen')
                global_e0[0]['closed_development100']=False;lookup[s.MANDATORY[0]]['E1_complete']=False
                report=s.build(root,models=models,freeze=True)
                self.assertIn(s.MANDATORY[0],report['requested_models'])
                self.assertEqual(report['E4_selection']['status'],'blocked_incomplete_requested_E1')


if __name__=='__main__':unittest.main()
