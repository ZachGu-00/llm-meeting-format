import unittest
from analyze_e0_v3 import admission,directions,flag


class AdmissionTests(unittest.TestCase):
    def ops(self,q=.8):return {'q':flag(q,.7),'format':flag(.92,.9),'R2':flag(.88,.85)}
    def test_near_zero_sign_reversal_is_not_exclusion(self):
        ds={'w_same':directions(-.02,.03),'w_diff':directions(.12,.16)}
        d=admission(self.ops(),True,ds,{'w_same':.05,'w_diff':.04,'ASI':.01},100)
        self.assertEqual(d['tier'],'R0_primary')
        self.assertEqual(ds['w_same']['status'],'indeterminate_near_zero')
    def test_clear_disagreement_retains_model_as_R1_primary(self):
        ds={'w_same':directions(-.12,.13)}
        d=admission(self.ops(),True,ds,{'w_same':.25},100)
        self.assertEqual(d['tier'],'R1_primary_R0_operational')
    def test_failed_R0_retains_valid_candidate_scoring(self):
        d=admission(self.ops(.1),True,{},None,100)
        self.assertEqual(d['tier'],'R1_only')
        self.assertFalse(d['R0_E4_primary_candidate'])
    def test_missing_diagnostics_and_pilot_never_claim_final_qualification(self):
        d=admission({'q':flag(None,.7)},True,{},None,100)
        self.assertEqual(d['tier'],'incomplete')
        d=admission(self.ops(),True,{},None,12)
        self.assertEqual(d['final_qualification'],'pilot_only')


if __name__=='__main__':unittest.main()
