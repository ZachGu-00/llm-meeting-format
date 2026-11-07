"""Probe-only scalar-transfer adapter. Native forwards and scientific files stay intact.

This is a different execution adapter, never a replacement calibration/runtime.
The native Engine identity is retained solely to validate its original token IDs.
"""
import math
from pathlib import Path

from loader_v3 import file_hash


class BulkTransferScorer:
    def __init__(self, engine):
        self.engine = engine
        self.identity = {'name': 'bulk_scalar_transfer_probe_v1', 'code_sha256': file_hash(__file__),
            'native_inference_sha256': file_hash(Path(__file__).with_name('inference_v3.py')),
            'forward_policy': 'Unchanged Engine.final_logits and NativeLoader.forward_tail',
            'R0_transfers': 'one batch of four selected scalars plus one batch of native argmax IDs',
            'R1_transfers': 'one gathered vector of all candidate-token logprobs per forward batch',
            'calibration_batch_size_relabelled': False, 'formal_runtime_migration_authorized': False}

    def r0(self, items, options, batch_size):
        e = self.engine; torch = e.torch
        prepared = [e.render(row['messages']) for row in items]; result = []
        for start in range(0, len(items), batch_size):
            ps = prepared[start:start + batch_size]
            z, lp = e.final_logits([p['ids'] for p in ps]); a, b = options['A'], options['B']
            values = torch.stack((z[:, a], z[:, b], lp[:, a], lp[:, b]), dim=1).detach().cpu().tolist()
            greedy_ids = z.argmax(dim=1).detach().cpu().tolist()
            for p, (za, zb, la, lb), greedy in zip(ps, values, greedy_ids):
                mass = math.exp(la) + math.exp(lb)
                if not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-6:
                    raise ValueError('Invalid full-vocabulary option mass')
                result.append({'R0_y_AB': za - zb, 'y_AB': za - zb, 'q': mass, 'logit_A': za, 'logit_B': zb,
                    'logprob_A': la, 'logprob_B': lb, 'greedy_argmax_id': greedy,
                    'greedy_argmax_token': e.tok.decode([greedy], skip_special_tokens=False),
                    'greedy_argmax_selected_letter': e.letter_for_id(greedy, options),
                    'greedy_argmax_any_variant_letter': e.letter_for_id(greedy),
                    'actual_prompt_sha256': p['sha256'], 'actual_prompt_ids_sha256': p['ids_sha256'],
                    'prompt_token_count': len(p['ids']), 'actual_batch_size_R0': len(ps), 'q_below_point5': mass < .5})
        return result

    def r1(self, items, batch_size):
        e = self.engine; torch = e.torch
        flattened = []; results = [{} for _ in items]
        for index, item in enumerate(items):
            p = e.render(item['messages'], readout='R1')
            results[index].update(R1_actual_prompt_sha256=p['sha256'], R1_actual_prompt_ids_sha256=p['ids_sha256'])
            for letter in 'AB':
                separator = ' ' if e.loader.model_kind == 'base' else ''
                joint = e.tok.encode(p['text'] + separator + item['candidates'][letter], add_special_tokens=False)
                if joint[:len(p['ids'])] != p['ids'] or len(joint) <= len(p['ids']):
                    raise ValueError('R1 prompt/continuation boundary changed: ' + item['record_id'] + ' ' + letter)
                if len(joint) > e.loader.max_length:
                    raise ValueError('R1 exceeds max_length')
                flattened.append((index, letter, p['ids'], joint[len(p['ids']):]))
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start:start + batch_size]; keep = max(len(c) for _, _, _, c in batch)
            lp = e.loader.forward_tail([p + c[:-1] for _, _, p, c in batch], keep)
            coordinates = [(j, keep - len(c) + k, token) for j, (_, _, _, c) in enumerate(batch) for k, token in enumerate(c)]
            index = torch.tensor(coordinates, device=lp.device, dtype=torch.long)
            values = lp[index[:, 0], index[:, 1], index[:, 2]].detach().cpu().tolist()
            offset = 0
            for i, letter, p, c in batch:
                token_lp = values[offset:offset + len(c)]; offset += len(c)
                total = sum(token_lp)
                if not all(math.isfinite(x) and x <= 1e-5 for x in token_lp):
                    raise ValueError('Invalid R1 token logprob')
                results[i]['R1_logprob_' + letter] = total
                results[i]['R1_candidate_' + letter] = {'text': items[i]['candidates'][letter],
                    'scored_leading_separator': ' ' if e.loader.model_kind == 'base' else '',
                    'token_ids': c, 'token_logprobs': token_lp, 'sum_logprob': total, 'actual_batch_size': len(batch)}
        for row in results:
            row['R1_y_AB'] = row['R1_logprob_A'] - row['R1_logprob_B']
        return results
