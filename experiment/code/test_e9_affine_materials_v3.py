"""CPU checks for missing geometry and exact reuse of original source messages."""
from collections import Counter
from fractions import Fraction
import itertools
from pathlib import Path
import unittest
import build_e9_affine_balance_v3 as builder


class AffineMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = builder.renderer(builder.BASE.parent / 'voices_nonadditivity_v2_20261030/code/render_v2.py')

    def item(self):
        return {'item_id': 'fixture', 'topic': 'fixture proposal', 'split': 'fixture',
                'focal_stance': 1, 'focal_statement': 'fixture focal statement',
                'roles': {role: {'stance': -1 if j < 3 else 1, 'argument': 'Verbatim argument ' + role}
                          for j, role in enumerate(builder.ROLES)}}

    def test_full_affine_family_fills_each_ordered_slot_pair(self):
        proof = builder.coverage_proof(self.original)
        self.assertTrue(proof['every_ordered_role_pair_covers_20_ordered_slot_pairs_once'])
        for values in proof['per_pair_histograms'].values():
            self.assertEqual(len(values['combined20']), 20)
            self.assertEqual({r['count'] for r in values['combined20']}, {1})

    def test_actual_messages_not_just_slot_counts_match_for_reuse(self):
        proof = builder.reuse_proof(self.original, [self.item()])
        self.assertEqual(proof['n_item_mask_checks'], 6)
        self.assertTrue(proof['all_equal'])
        for check in proof['checks']:
            self.assertEqual(check['old_multiset_sha256'], check['new_multiset_sha256'])

    def test_new_metadata_changes_only_comment_order_and_record_identity(self):
        item = self.item()
        row = builder.additional_row(self.original, item, 9, 3, 4, 1)
        old = self.original.render(item, 9, 0, 1)
        self.assertEqual(row['permutation'], 19)
        self.assertEqual(row['slot_role_order'], ['Dprime', 'C', 'A', 'D', 'Aprime'])
        self.assertEqual(row['messages'][0], old['messages'][0])
        self.assertEqual(row['candidates'], old['candidates'])
        self.assertEqual(row['positive_letter'], old['positive_letter'])
        for candidate in ('Verbatim argument A', 'Verbatim argument D'):
            self.assertIn(candidate, row['messages'][1]['content'])
        self.assertEqual(row['messages'][1]['content'].count('(has not commented yet)'), 3)
        self.assertEqual(row['messages'][1]['content'].split('\n\nWhat is your position now?')[1],
                         old['messages'][1]['content'].split('\n\nWhat is your position now?')[1])

    def test_stance_blind_linear_adjacency_counterexample_disappears(self):
        roles = builder.ROLES
        old = self.original.permutations()
        new = [builder.affine_order(a, b) for a in (2, 3) for b in range(5)]
        def contrast(orders):
            def interaction(i, j):
                return Fraction(sum(abs(order.index(roles[i]) - order.index(roles[j])) == 1
                                    for order in orders), len(orders))
            oo = sum(interaction(i, j) for i, j in itertools.combinations(range(3), 2)) / 3
            aa = interaction(3, 4)
            oa = sum(interaction(i, j) for i in range(3) for j in (3, 4)) / 6
            return oo + aa - 2 * oa
        self.assertEqual(contrast(old), Fraction(4, 5))
        self.assertEqual(contrast(new), Fraction(-4, 5))
        self.assertEqual(contrast(old + new), 0)

    def test_content_change_invalidates_reuse_hash(self):
        row = self.original.render(self.item(), 1, 0, 0)
        prior = builder.message_hash(row)
        row['messages'][1]['content'] += ' Changed instruction.'
        self.assertNotEqual(prior, builder.message_hash(row))


if __name__ == '__main__':
    unittest.main()
