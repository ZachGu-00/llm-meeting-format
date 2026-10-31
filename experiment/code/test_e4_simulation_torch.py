"""CPU-only checks of paired MC moments, common randomness and update semantics."""
import importlib.util
import itertools
import unittest
import numpy as np
from e4_simulation import simulate as numpy_simulate, predicted_caches
from e4_simulation_torch import simulate


LEFT = [(0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)]
RIGHT = [(0, 1, 3), (0, 2, 4), (1, 2, 5), (3, 4, 5)]
SECTIONS = ('outcomes', 'paired_difference_to_design0', 'paired_residual_U_measured_minus_other_methods')


def scalar_counts(caches, designs, initial, priorities, uniforms, tau=1):
    """Independent trajectory-by-trajectory oracle using explicit common draws."""
    designs = [sorted(tuple(sorted(g)) for g in d) for d in designs]
    universe = sorted(set(itertools.chain.from_iterable(designs)))
    lookup = {g: u for u, g in enumerate(universe)}
    T, R, _ = priorities.shape
    n = next(iter(caches.values())).shape[0]
    initial = np.asarray(initial)
    if initial.ndim == 1:
        initial = np.broadcast_to(initial, (len(designs), n))
    counts = np.zeros((len(caches), len(designs), T, R), dtype=np.int64)
    for mi, logits in enumerate(caches.values()):
        for di, design in enumerate(designs):
            for t in range(T):
                state = initial[di].copy()
                for r in range(R):
                    order = sorted(design, key=lambda g: (priorities[t, r, lookup[g]], g))
                    for group in order:
                        before = state.copy(); decisions = []
                        for pos, j in enumerate(group):
                            k, l = sorted(v for v in group if v != j)
                            pattern = 4 * int(before[j]) + 2 * int(before[k]) + int(before[l])
                            probability = np.exp(-np.logaddexp(0., -logits[j, k, l, pattern]))
                            decisions.append(uniforms[t, r, lookup[group], pos] < probability)
                        state[list(group)] = decisions
                    count = state.sum()
                    counts[mi, di, t, r] = count if tau == 1 else n - count
    return counts


def random_caches(n=6):
    rng = np.random.default_rng(177)
    return {name: rng.normal(0, 1.3, (n, n, n, 8)) for name in ('measured', 'P0', 'PK')}


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch is not installed')
class TorchMonteCarloTests(unittest.TestCase):
    def test_matches_numpy_random_stream_all_paired_moments(self):
        caches = random_caches(); designs = [LEFT, RIGHT, list(reversed(LEFT))]
        initial = [1, 0, 1, 0, 1, 0]
        for tau in (1, -1):
            reference = numpy_simulate(caches, designs, initial, rounds=3, draws=37, chunk_size=11, seed=95, tau=tau)
            actual = simulate(caches, designs, initial, rounds=3, draws=37, chunk_size=11, seed=95, tau=tau)
            for section in SECTIONS:
                for key in ('means', 'Monte_Carlo_standard_errors'):
                    np.testing.assert_allclose(actual[section][key], reference[section][key], atol=1e-14, rtol=1e-13)
            self.assertTrue(np.any(np.asarray(actual['paired_residual_U_measured_minus_other_methods']['Monte_Carlo_standard_errors']) > 0))
            np.testing.assert_array_equal(np.asarray(actual['paired_difference_to_design0']['Monte_Carlo_standard_errors'])[:, 2], 0)

    def test_explicit_tape_scalar_oracle_count_sums_and_mcse_chunk_invariance(self):
        caches = random_caches(); designs = [LEFT, RIGHT]; initial = [[1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0]]
        T, R = 19, 3
        U = len(set(LEFT + RIGHT)); rng = np.random.default_rng(72)
        tape = (rng.random((T, R, U)), rng.random((T, R, U, 3)))
        counts = scalar_counts(caches, designs, initial, *tape)
        delta = counts - counts[:, :1, :, :]
        expected = (counts, delta, delta[:1] - delta[1:])
        results = [simulate(caches, designs, initial, rounds=R, draws=T, chunk_size=b, random_tape=tape) for b in (4, 13)]
        for actual in results:
            for section, integer in zip(SECTIONS, expected):
                observed = actual[section]
                np.testing.assert_array_equal(observed['integer_count_sums'], integer.sum(axis=2))
                np.testing.assert_array_equal(observed['integer_count_square_sums'], (integer * integer).sum(axis=2))
                np.testing.assert_allclose(observed['means'], integer.mean(axis=2) / 6, atol=1e-15)
                np.testing.assert_allclose(observed['Monte_Carlo_standard_errors'], integer.std(axis=2, ddof=1) / (6 * np.sqrt(T)), atol=1e-15)

    def test_simultaneous_members_and_sequential_groups(self):
        n = 4; logits = np.full((n, n, n, 8), np.nan)
        for j, k, l in itertools.permutations(range(n), 3):
            if k < l:
                logits[j, k, l] = [-1000 if pattern == 0 else 1000 for pattern in range(8)]
        designs = [[(0, 1, 2), (1, 2, 3)]]
        priorities = np.array([[[.1, .9]], [[.9, .1]]])
        uniforms = np.full((2, 1, 2, 3), .5)
        result = simulate({'measured': logits}, designs, [1, 0, 0, 0], rounds=1, draws=2,
                          random_tape=(priorities, uniforms))
        # Left-first spreads support to all four; right-first finishes with three.
        self.assertEqual(result['outcomes']['integer_count_sums'], [[[7]]])
        self.assertEqual(result['outcomes']['integer_count_square_sums'], [[[25]]])
        self.assertAlmostEqual(result['outcomes']['means'][0][0][0], .875)
        self.assertAlmostEqual(result['outcomes']['Monte_Carlo_standard_errors'][0][0][0], .125)
        # A cyclic copy within a meeting must rotate, not sequentially erase, the one support bit.
        cyclic = np.full((3, 3, 3, 8), np.nan)
        for j in range(3):
            k, l = sorted(x for x in range(3) if x != j)
            for pattern in range(8):
                bits = {j: pattern >> 2, k: (pattern >> 1) & 1, l: pattern & 1}
                cyclic[j, k, l, pattern] = 1000 if bits[(j + 1) % 3] else -1000
        out = simulate({'measured': cyclic}, [[(0, 1, 2)]], [1, 0, 0], rounds=4, draws=7)
        np.testing.assert_allclose(out['outcomes']['means'], 1 / 3)
        np.testing.assert_array_equal(out['outcomes']['Monte_Carlo_standard_errors'], 0)

    def test_additive_oracle_has_zero_paired_residual_noise(self):
        rng = np.random.default_rng(29); n = 6
        cache = {'n': n, 'alpha': rng.normal(size=(n, 2)), 'pair': rng.normal(size=(n, n, 4)),
                 'triple': np.zeros((n, n, n, 8))}
        predictions = predicted_caches(cache, {k: 0. for k in ('opponent_opponent', 'opponent_ally', 'ally_ally')})
        predictions['measured'] = predictions['P0'].copy()
        actual = simulate(predictions, [LEFT, RIGHT], [1, 1, 0, 0, 0, 0], rounds=3, draws=43, chunk_size=16)
        for key in ('means', 'Monte_Carlo_standard_errors', 'integer_count_sums', 'integer_count_square_sums'):
            np.testing.assert_array_equal(actual['paired_residual_U_measured_minus_other_methods'][key], 0)

    def test_device_rng_reproducible_and_does_not_mutate_global_rng(self):
        import torch
        before = torch.random.get_rng_state().clone()
        kwargs = dict(rounds=2, draws=23, chunk_size=8, seed=41, rng_backend='torch')
        first = simulate(random_caches(), [LEFT, RIGHT], [1, 0, 0, 0, 1, 0], **kwargs)
        second = simulate(random_caches(), [LEFT, RIGHT], [1, 0, 0, 0, 1, 0], **kwargs)
        for section in SECTIONS:
            self.assertEqual(first[section], second[section])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_missing_required_states_are_rejected(self):
        caches = random_caches(); caches['measured'][0, 1, 2, 7] = np.nan
        with self.assertRaisesRegex(ValueError, 'Missing dynamic'):
            simulate(caches, [LEFT], [0] * 6, draws=2)


if __name__ == '__main__':
    unittest.main()
