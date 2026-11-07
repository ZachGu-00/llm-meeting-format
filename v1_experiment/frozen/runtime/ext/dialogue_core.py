"""Deterministic, auditable generated-meeting semantics, without a model loader."""
from collections import defaultdict
import hashlib
import itertools
import math
from common import canonical


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed(*parts):
    return int(hashlib.sha256(('voices_v3_dialogue_v1|' + '|'.join(map(str, parts))).encode()).hexdigest()[:15], 16)


def unit(*parts):
    # 53-bit open-interval uniforms; the convention is recorded in the plan.
    return (seed(*parts) % (2**53) + .5) / 2**53


def phrase(stance):
    if stance not in (-1, 1):
        raise ValueError('Stances must be -1 or +1')
    return 'I support this proposal.' if stance == 1 else 'I oppose this proposal.'


def normalized(text):
    return ' '.join(text.split())


def meeting_schedule(config):
    blocks = [tuple(b) for b in config['blocks']]
    n = len(config['agent_ids'])
    if (len(set(config['agent_ids'])) != n or any(len(b) != 3 or tuple(sorted(set(b))) != b for b in blocks)
        or len(set(blocks)) != len(blocks) or set(itertools.chain.from_iterable(blocks)) != set(config['agent_ids'])):
        raise ValueError('Invalid distinct meetings or population')
    if len(config['initial_state']) != n or any(x not in (-1, 1) for x in config['initial_state']):
        raise ValueError('Invalid initial stance vector')
    result = []
    for r in range(config['rounds']):
        order = sorted(blocks, key=lambda b: (seed(config['coupling_key'], 'meeting_order', r, *b), b))
        for s, block in enumerate(order):
            result.append({'index': len(result), 'round': r, 'step': s, 'members': list(block),
                           'priority_seed': seed(config['coupling_key'], 'meeting_order', r, *block)})
    return result


def generation_messages(config, meeting, focal, state, memory):
    lookup = dict(zip(config['agent_ids'], state)); stance = lookup[focal]
    agent = next(a for a in config['agents'] if a['agent_id'] == focal)
    argument = agent['arguments'][str(stance)]['argument']
    prior = memory.get(str(focal))
    history = ('No earlier meeting transcript.' if prior is None else
               '\n'.join(f"Agent {a}: {prior['statements'][str(a)]}" for a in prior['members']))
    user = (f'Proposal: "{config["topic"]}"\n\nYour current position: {phrase(stance)}\n'
            f'Your original argument for this position: "{argument}"\n\n'
            'The immediately preceding meeting you attended:\n' + history + '\n\n'
            'State your current case to the other participants in approximately 60 English words. '
            'Write only your statement. Do not give a numerical probability.')
    return [{'role': 'system', 'content': config['system_message']}, {'role': 'user', 'content': user}]


def generation_seed(config, meeting, focal):
    return seed(config['coupling_key'], 'generation', meeting['round'], *meeting['members'], focal)


def render_readouts(config, meeting, state, generations):
    members = meeting['members']; stances = dict(zip(config['agent_ids'], state))
    generated = {g['focal_agent_id']: g for g in generations}
    if set(generated) != set(members):
        raise ValueError('Each current member must have exactly one saved utterance')
    rows = []
    for focal in members:
        peers = [p for p in members if p != focal]
        cells = [('empty', [], 0)]
        cells += [(f'peer_{p}', [p], position) for p in peers for position in (0, 1)]
        cells += [('joint', peers, position) for position in (0, 1)]
        for kind, included, position in cells:
            if not included:
                assigned = {}
            elif len(included) == 1:
                assigned = {1 if position == 0 else 3: included[0]}
            else:
                assigned = dict(zip((1, 3), included if position == 0 else included[::-1]))
            lines = []
            for slot in range(5):
                peer = assigned.get(slot)
                body = ('(has not commented yet)' if peer is None else
                        phrase(stances[peer]) + ' ' + normalized(generated[peer]['text']))
                lines.append(f'Participant {slot + 1}: {body}')
            for label in (0, 1):
                candidates = dict(zip('AB', [phrase(s) for s in ((1, -1) if label == 0 else (-1, 1))]))
                content = (f'Proposal: "{config["topic"]}"\n\nEarlier in this discussion, you wrote: '
                           f'"{normalized(generated[focal]["text"])}"\n\nComments from the other participants in this round:\n'
                           + '\n'.join(lines) + f'\n\nWhat is your position now?\n(A) {candidates["A"]}'
                           f'\n(B) {candidates["B"]}\nAnswer with a single letter.')
                rid = f'{config["run_id"]}_m{meeting["index"]:03d}_a{focal:02d}_{kind}_p{position}_l{label}'
                messages = [{'role': 'system', 'content': config['system_message']}, {'role': 'user', 'content': content}]
                rows.append({'id': rid, 'record_id': rid, 'run_id': config['run_id'], 'item_id': rid.rsplit('_p', 1)[0],
                    'topic_id': config['topic_id'], 'topic': config['topic'], 'experiment': config['experiment'],
                    'meeting_index': meeting['index'], 'round': meeting['round'], 'focal_agent_id': focal,
                    'focal_stance': stances[focal], 'peer_agent_ids': included,
                    'peer_stances': [stances[p] for p in included], 'condition': kind, 'permutation': position,
                    'label_map': label, 'positive_letter': 'A' if label == 0 else 'B', 'candidates': candidates,
                    'messages': messages, 'prompt_hash': digest(messages), 'slot_agent_ids': [assigned.get(i) for i in range(5)],
                    'positive_direction': 'global_support_stance_+1', 'template_variant': 'E4_F1_generated_dialogue',
                    'validation_status': 'native_generated_text_no_independent_semantic_stance_validation',
                    'peer_prefix_meaning': 'premeeting_agent_state_not_a_judgement_of_generated_prose',
                    'generation_sha256': {str(a): digest(generated[a]) for a in [focal, *included]},
                    'text_display_transform': 'Unicode_whitespace_normalization_only'})
    if len(rows) != 42 or len({r['record_id'] for r in rows}) != 42:
        raise ValueError('Every meeting requires exactly 42 balanced native readouts')
    return rows


def sigmoid(x):
    return math.exp(-max(0., -x)) / (math.exp(-max(0., -x)) + math.exp(-max(0., x)))


def close_meeting(config, meeting, state, memory, generations, rows, kernels):
    expected = render_readouts(config, meeting, state, generations)
    if len(rows) != len(expected):
        raise ValueError('Incomplete meeting response panel')
    cells = defaultdict(dict); q = []
    for source, row in zip(expected, rows):
        if any(row.get(k) != v for k, v in source.items()) or row.get('status') != 'ok':
            raise ValueError('Measured prompt identity differs')
        y = float(row['R0_y_AB']) * (1 if row['positive_letter'] == 'A' else -1)
        mass = float(row['q'])
        if not math.isfinite(y) or not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-6:
            raise ValueError('Invalid native R0 logit or full-vocabulary option mass')
        cells[row['focal_agent_id'], row['condition']][row['permutation'], row['label_map']] = y
        q.append(mass)
    values = {key: sum(sum(v[p, l] for l in (0, 1)) / 2 for p in sorted({k[0] for k in v})) /
              len({k[0] for k in v}) for key, v in cells.items()}
    stances = dict(zip(config['agent_ids'], state)); after = dict(stances); results = []
    for focal in meeting['members']:
        a, b = [p for p in meeting['members'] if p != focal]
        alpha = values[focal, 'empty']; va = values[focal, f'peer_{a}'] - alpha; vb = values[focal, f'peer_{b}'] - alpha
        relation = ('opponent_opponent' if stances[a] != stances[focal] and stances[b] != stances[focal] else
                    'ally_ally' if stances[a] == stances[focal] and stances[b] == stances[focal] else 'opponent_ally')
        kernel = kernels[relation]
        if kernel is None or not math.isfinite(kernel):
            raise ValueError('Undefined independently measured E1 relationship kernel')
        logits = {'measured': values[focal, 'joint'], 'P0': alpha + va + vb,
                  'PK': alpha + va + vb - stances[focal] * kernel * (abs(va) + abs(vb)) / 2}
        probabilities = {k: sigmoid(v) for k, v in logits.items()}
        uniform = unit(config['coupling_key'], 'update', meeting['round'], *meeting['members'], focal)
        after[focal] = 1 if uniform < probabilities['measured'] else -1
        results.append({'focal_agent_id': focal, 'before': stances[focal], 'after': after[focal], 'alpha': alpha,
                        'v': [va, vb], 'relation': relation, 'kernel': kernel, 'logits': logits,
                        'support_probabilities': probabilities, 'sampled_uniform': uniform})
    new_memory = dict(memory)
    transcript = {'meeting_index': meeting['index'], 'members': meeting['members'],
                  'statements': {str(g['focal_agent_id']): g['text'] for g in generations}}
    for focal in meeting['members']:
        new_memory[str(focal)] = transcript
    new_state = [after[a] for a in config['agent_ids']]
    event = {'meeting': meeting, 'before': state, 'after': new_state, 'member_updates': results,
             'generation_sha256': [digest(g) for g in generations], 'memory_before_sha256': digest(memory),
             'memory_after_sha256': digest(new_memory), 'q_min': min(q), 'q_mean': sum(q) / len(q),
             'q_below_point5_fraction': sum(v < .5 for v in q) / len(q)}
    return event, new_state, new_memory
