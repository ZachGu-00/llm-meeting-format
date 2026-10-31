import itertools
import unittest
import numpy as np
import analyze_v3 as a
from e4_graphs import PASCH_LEFT,PASCH_RIGHT,pair_projection,validate_sts
from e4_simulation import predicted_caches,simulate

def panel_for(rule):
    v=np.array([[1,2,3,-1,-2],[2,1,4,-3,-1],[.5,1,1.5,-.4,-2]])
    alpha=np.array([-2.,.4,1.]);y=np.zeros((3,32));y[:,0]=alpha
    for mask in range(1,32):
        peers=[j for j in range(5) if mask>>j&1]
        y[:,mask]=alpha+(v[:,peers].sum(axis=1) if rule=='additive' else v[:,peers].mean(axis=1))
    return {'y':y,'masks':list(range(32)),'item_ids':['a','b','c'],'topics':['t1','t2','t3'],'n_physical_rows':1866,'q_median':.99}

class EstimandTests(unittest.TestCase):
    def test_additive_and_mean_have_zero_C_and_correct_rules(self):
        for name,rule in [('additive','additive'),('mean','present_mean')]:
            p=a.point(a.sufficient(panel_for(name)))
            self.assertAlmostEqual(p['C'],0,places=12)
            self.assertAlmostEqual(p['rules'][rule]['R2_within_item'],1,places=12)
    def test_positive_affine_invariance_and_exact_mobius(self):
        panel=panel_for('mean');panel['y'][:,7]+=1.7
        original=a.sufficient(panel);p=a.point(original)
        modified={**panel,'y':7.3*panel['y']+4.7};q=a.point(a.sufficient(modified))
        self.assertAlmostEqual(p['C'],q['C'])
        for k in p['relation_kernels']:self.assertAlmostEqual(p['relation_kernels'][k],q['relation_kernels'][k])
        coeff=a.mobius_partial(panel['y'],panel['masks'])
        restored=np.stack([sum(coeff[:,s] for s in range(32) if s&mask==s) for mask in range(32)],axis=1)
        np.testing.assert_allclose(restored,panel['y'],atol=1e-12)
    def test_C_uses_all_cross_pairs(self):
        panel=panel_for('additive')
        for mask in range(32):
            if mask&9==9:panel['y'][:,mask]+=2
        self.assertAlmostEqual(a.point(a.sufficient(panel))['C'],-(2/3)/a.sufficient(panel)['C_den'].mean())
    def test_side_max_keeps_measured_sign(self):
        v=np.array([[-3.,2,1,.4,-.2]])
        self.assertEqual(a.rule_predictions(v,[3])['side_max'][0,0],-3.)
    def test_holm_retains_missing_family_members(self):
        adjusted=a.holm({'a':.01,'b':.04,'c':None})
        self.assertAlmostEqual(adjusted['a'],.03);self.assertAlmostEqual(adjusted['b'],.08);self.assertIsNone(adjusted['c'])

class GraphAndDynamicTests(unittest.TestCase):
    def test_projection_exact(self):
        self.assertEqual(pair_projection(PASCH_LEFT),pair_projection(PASCH_RIGHT))
        self.assertEqual(pair_projection([tuple(range(5))]*3),pair_projection(list(itertools.combinations(range(5),3))))
    def test_PK_orientation_follows_focal(self):
        n=3;alpha=np.zeros((n,2));pair=np.zeros((n,n,4))
        for j,k in itertools.permutations(range(n),2):
            for sj,sk in itertools.product((0,1),repeat=2):pair[j,k,2*sj+sk]=(1-2*sj) if sj!=sk else 0
        cache={'n':n,'alpha':alpha,'pair':pair,'triple':np.zeros((n,n,n,8))}
        pred=predicted_caches(cache,dict.fromkeys(a.PAIRS,-1))
        self.assertEqual(pred['PK'][0,1,2,3],1)
        self.assertEqual(pred['PK'][0,1,2,4],-1)
    def test_common_random_numbers_and_target_complement(self):
        cache=np.zeros((6,6,6,8));designs=[PASCH_LEFT,PASCH_LEFT]
        kwargs=dict(rounds=2,draws=128,chunk_size=32,seed=42)
        x=simulate({'measured':cache,'P0':cache},designs,[0,0,0,1,1,1],**kwargs)
        y=simulate({'measured':cache,'P0':cache},designs,[0,0,0,1,1,1],tau=-1,**kwargs)
        means=np.array(x['outcomes']['means'])
        np.testing.assert_array_equal(means[0],means[1]);np.testing.assert_array_equal(means[:,0],means[:,1])
        np.testing.assert_allclose(means+np.array(y['outcomes']['means']),1,atol=1e-12)
        self.assertTrue(np.all(np.array(x['paired_residual_U_measured_minus_other_methods']['means'])==0))
    def test_synchronous_within_meeting(self):
        cache=np.zeros((3,3,3,8))
        for index in range(8):cache[:,:,:,index]=1000 if index<4 else -1000
        result=simulate({'measured':cache},[[(0,1,2)]],[1,0,1],rounds=1,draws=8,chunk_size=4)
        self.assertAlmostEqual(result['outcomes']['means'][0][0][0],1/3)

if __name__=='__main__':unittest.main()
