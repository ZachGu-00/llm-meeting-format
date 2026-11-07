"""Optional Torch implementation of the native E4 Monte Carlo simulator.

Import is Torch-free. ``simulate(..., device='cuda:0')`` runs one group-count
batch (normally 80 STS designs for ONE mapping) on the selected device. The
first eight arguments and existing result keys match e4_simulation.simulate.

Methods/designs share priorities and member uniforms indexed by the sorted
canonical triple. All three member decisions use the pre-meeting state; groups
then update sequentially. Initial bits and transition probabilities refer to
global support. tau only reverses the reported outcome, never the transition.

rng_backend='numpy' preserves the reference seed/chunk random stream. 'torch'
uses a private device generator; its stream differs and is recorded explicitly.
For small exact checks, random_tape=(priorities, uniforms) has shapes
(draws, rounds, n_unique_triples) and (draws, rounds, n_unique_triples, 3).
Uniform member positions follow each canonical triple's ascending agent IDs.
Explicit tapes make chunk-size comparisons use the same trajectories.

No training, model loading, global RNG mutation, or inference environment change.
Monte Carlo error is conditional on fixed caches, initial states and this mapping.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import numpy as np


def _prepare(logit_caches, designs, initial_states):
    names = list(logit_caches)
    if not names or names[0] != 'measured':
        raise ValueError('First cache must be measured for paired prediction residuals')
    caches = np.stack([np.asarray(logit_caches[name], dtype=np.float64) for name in names])
    n = caches.shape[1] if caches.ndim == 5 else 0
    if n < 3 or caches.shape[1:] != (n, n, n, 8):
        raise ValueError('Each cache must have shape (n_agents,n_agents,n_agents,8)')
    canonical = []
    for design in designs:
        groups = []
        for block in design:
            if len(block) != 3 or any(not isinstance(x, (int, np.integer)) for x in block):
                raise ValueError('Each block must contain three integer agent IDs')
            group = tuple(sorted(map(int, block)))
            if len(set(group)) != 3 or group[0] < 0 or group[-1] >= n:
                raise ValueError('Distinct in-range triple members required')
            groups.append(group)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError('Each design needs distinct, nonempty triples')
        canonical.append(sorted(groups))
    if not canonical or len({len(d) for d in canonical}) != 1:
        raise ValueError('Batch requires nonempty designs with equal group counts')
    universe = sorted(set(itertools.chain.from_iterable(canonical)))
    index = {block: i for i, block in enumerate(universe)}
    ids = np.array([[index[block] for block in design] for design in canonical], dtype=np.int64)
    blocks = np.array(universe, dtype=np.int64)
    initial = np.asarray(initial_states)
    if initial.shape == (n,):
        initial = np.broadcast_to(initial, (len(canonical), n))
    if initial.shape != (len(canonical), n) or not np.isin(initial, [0, 1]).all():
        raise ValueError('Initial states must be binary support bits')
    initial = initial.astype(np.uint8, copy=True)
    # Compile each of the eight canonical group states into its three focal
    # probabilities once. Peers remain sorted after choosing the focal member.
    bits = (np.arange(8)[:, None] >> np.array([2, 1, 0])) & 1
    probability = np.empty((len(names), len(blocks), 8, 3), dtype=np.float64)
    for pos in range(3):
        others = [p for p in range(3) if p != pos]
        state_index = 4 * bits[:, pos] + 2 * bits[:, others[0]] + bits[:, others[1]]
        y = caches[:, blocks[:, pos, None], blocks[:, others[0], None],
                   blocks[:, others[1], None], state_index[None, :]]
        if not np.isfinite(y).all():
            raise ValueError('Missing dynamic group-state cache')
        probability[..., pos] = np.exp(-np.logaddexp(0., -y))
    return names, blocks, ids, initial, probability


def simulate(logit_caches, designs, initial_states, rounds=6, draws=100000,
             chunk_size=256, seed=20261030, tau=1, *, device='cpu',
             rng_backend='numpy', random_tape=None):
    """Return outcomes and genuinely paired contrast/residual MC standard errors.

Support counts are accumulated as int64, including their squares. Means and
sample-variance MCSEs are computed in float64 with ddof=1. Paired contrasts use
design 0 within this call; residuals subtract each predicted contrast from the
measured contrast on the SAME trajectory. They do not cover every design pair.
"""
    try:
        import torch
    except ImportError as error:
        raise ImportError('Torch simulation is optional; install torch or use e4_simulation.simulate') from error
    for name, value, minimum in (('draws', draws, 2), ('rounds', rounds, 1), ('chunk_size', chunk_size, 1)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if tau not in (-1, 1):
        raise ValueError('tau must be a global stance sign')
    if rng_backend not in ('numpy', 'torch'):
        raise ValueError('rng_backend must be numpy or torch')
    names, blocks_np, ids_np, initial_np, probability_np = _prepare(logit_caches, designs, initial_states)
    M, D, U, n = len(names), len(ids_np), len(blocks_np), initial_np.shape[1]
    dev = torch.device(device)
    blocks = torch.as_tensor(blocks_np, dtype=torch.long, device=dev)
    ids = torch.as_tensor(ids_np, dtype=torch.long, device=dev)
    initial = torch.as_tensor(initial_np, dtype=torch.uint8, device=dev)
    probability = torch.as_tensor(probability_np, dtype=torch.float64, device=dev)
    totals = torch.zeros((M, D, rounds), dtype=torch.int64, device=dev)
    squares = torch.zeros_like(totals)
    delta_totals, delta_squares = torch.zeros_like(totals), torch.zeros_like(totals)
    residual_totals = torch.zeros((M - 1, D, rounds), dtype=torch.int64, device=dev)
    residual_squares = torch.zeros_like(residual_totals)
    rng = np.random.default_rng(seed) if rng_backend == 'numpy' else torch.Generator(device=dev).manual_seed(seed)
    tape = None
    if random_tape is not None:
        if len(random_tape) != 2:
            raise ValueError('random_tape requires priorities and member uniforms')
        tape = tuple(np.asarray(x, dtype=np.float64) for x in random_tape)
        if tape[0].shape != (draws, rounds, U) or tape[1].shape != (draws, rounds, U, 3):
            raise ValueError('random_tape shape must follow draws/rounds/canonical-triple/member')
        if any(not np.isfinite(x).all() or (x < 0).any() or (x >= 1).any() for x in tape):
            raise ValueError('Explicit random values must be finite in [0,1)')
    mi = torch.arange(M, device=dev)[:, None, None]
    with torch.inference_mode():
        for start in range(0, draws, chunk_size):
            B = min(chunk_size, draws - start)
            state = initial[None, :, None, :].expand(M, D, B, n).clone()
            bi = torch.arange(B, device=dev)[None, :]
            for r in range(rounds):
                if tape is not None:
                    priority = torch.as_tensor(tape[0][start:start + B, r].copy(), dtype=torch.float64, device=dev)
                    uniform = torch.as_tensor(tape[1][start:start + B, r].copy(), dtype=torch.float64, device=dev)
                elif rng_backend == 'numpy':
                    priority = torch.as_tensor(rng.random((B, U)), dtype=torch.float64, device=dev)
                    uniform = torch.as_tensor(rng.random((B, U, 3)), dtype=torch.float64, device=dev)
                else:
                    priority = torch.rand((B, U), generator=rng, dtype=torch.float64, device=dev)
                    uniform = torch.rand((B, U, 3), generator=rng, dtype=torch.float64, device=dev)
                order = torch.argsort(priority[:, ids].permute(1, 0, 2), dim=2, stable=True)
                ordered_groups = ids[:, None, :].expand(D, B, -1).gather(2, order)
                for step in range(ids.shape[1]):
                    group = ordered_groups[:, :, step]
                    members = blocks[group][None, :, :, :].expand(M, -1, -1, -1)
                    before = state.gather(3, members)
                    bits = (4 * before[..., 0] + 2 * before[..., 1] + before[..., 2]).long()
                    probabilities = probability[mi, group[None, :, :], bits, :]
                    decisions = uniform[bi, group, :][None, :, :, :] < probabilities
                    state.scatter_(3, members, decisions.to(torch.uint8))
                outcome = state.sum(dim=3, dtype=torch.int64)
                if tau == -1:
                    outcome = n - outcome
                delta = outcome - outcome[:, :1, :]
                residual = delta[:1, :, :] - delta[1:, :, :]
                for value, sums, sq in ((outcome, totals, squares), (delta, delta_totals, delta_squares),
                                        (residual, residual_totals, residual_squares)):
                    sums[:, :, r] += value.sum(dim=2)
                    sq[:, :, r] += (value * value).sum(dim=2)

    def finish(sums, sq):
        sums, sq = sums.cpu().numpy(), sq.cpu().numpy()
        s, q = sums.astype(np.float64), sq.astype(np.float64)
        variance = np.maximum(0., (q - s * s / draws) / (draws - 1))
        return {'means': (s / (draws * n)).tolist(),
                'Monte_Carlo_standard_errors': (np.sqrt(variance / draws) / n).tolist(),
                'integer_count_sums': sums.tolist(), 'integer_count_square_sums': sq.tolist(),
                'count_normalization_denominator': n}

    return {'methods': names, 'n_designs': D, 'n_agents': n, 'rounds': rounds, 'draws': draws,
        'chunk_size': chunk_size, 'seed': seed, 'tau': tau,
        'common_random_numbers': 'trajectory/round/canonical-triple/member; shared priorities and Bernoulli draws',
        'within_meeting_update': 'simultaneous decisions from pre-meeting states; groups sequential',
        'outcomes': finish(totals, squares),
        'paired_difference_to_design0': finish(delta_totals, delta_squares),
        'paired_residual_U_measured_minus_other_methods': finish(residual_totals, residual_squares),
        'interval_scope': 'Monte Carlo error conditional on caches; topic sampling uncertainty must be computed separately',
        'simulation_backend': {'implementation': 'torch_vectorized_canonical_group_table_v1',
            'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': str(dev),
            'rng': 'explicit_numpy_tape' if tape is not None else 'numpy_default_rng_PCG64' if rng_backend == 'numpy' else 'torch_private_device_generator',
            'seed_and_chunk_define_random_stream': tape is None,
            'probabilities': 'NumPy float64 stable sigmoid compiled from fixed caches',
            'accumulators': 'int64 counts; float64 ddof1 MCSE',
            'priority_ties': 'stable canonical-triple ordering; probability zero for ideal continuous uniforms',
            'canonical_triples': blocks_np.tolist(),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
