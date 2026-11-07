"""Independent design and nonlinear-averaging identities."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_meeting_format.design import check_design, pair_contacts
from llm_meeting_format.predictors import jensen_gap, sigmoid


class ScientificChecks(unittest.TestCase):
    def test_contact_matched_interventions(self):
        result = check_design()
        self.assertEqual(result["meeting_counts"], {"base": 40, "regroup": 40, "resize": 26})

    def test_auxiliary_effective_weights(self):
        from itertools import combinations
        triads = list(combinations(range(5), 3))
        five = [tuple(range(5))] * 3
        self.assertEqual(pair_contacts(triads), pair_contacts(five))
        self.assertEqual(3 / (6 * 2), 3 / (3 * 4))
        alpha, influence = 0.7, np.array([3.0, -2.0, 0.1, 0.5])
        logits = np.array([alpha + influence[list(peers)].mean() for peers in combinations(range(4), 2)])
        self.assertAlmostEqual(logits.mean(), alpha + influence.mean())
        self.assertGreater(abs(sigmoid(logits).mean() - sigmoid(logits.mean())), 1e-4)

    def test_sigmoid_jensen_bound(self):
        rng = np.random.default_rng(813)
        values = rng.normal(size=(2000, 6)) * rng.uniform(0.01, 12, size=(2000, 1))
        gap, bound = jensen_gap(values)
        self.assertTrue(np.all(np.abs(gap) <= bound + 1e-12))
        cancellation, _ = jensen_gap(np.array([[-2.0, 2.0]]))
        self.assertAlmostEqual(float(cancellation[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
