"""Sum, Mean, Size, Stance, and Static network predictors."""
from itertools import combinations, chain
import json
import numpy as np
from .design import encode, stance_patterns


METHODS = ("observed", "sum", "mean", "size", "stance", "network")


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits)))


def fit_weights(data_root, model, held_out_topic):
    inputs = data_root / "inputs"
    topics = {row["item_id"]: row["topic"] for row in json.loads(
        (inputs / "e1_semantic64_items.json").read_text(encoding="utf-8"))}
    micro = json.loads((inputs / (model + "__e2e_R0_analysis.json")).read_text(encoding="utf-8"))
    selected = [i for i, item in enumerate(micro["item_ids"]) if topics[item] != held_out_topic]
    response = np.asarray(micro["item_count_deltas"])[selected]
    conditions = list(map(tuple, micro["conditions"]))
    opponent = response[:, conditions.index((1, 0))]
    ally = response[:, conditions.index((0, 1))]
    size, stance = {}, {}
    for n in (2, 4):
        cells = [(o, a) for o, a in conditions if o + a == n]
        x = np.concatenate([o * opponent + a * ally for o, a in cells])
        y = np.concatenate([response[:, conditions.index(cell)] for cell in cells])
        size[n] = float(x @ y / (x @ x)) if x @ x > 0 else np.nan
        for o in range(n + 1):
            a = n - o
            if (o, a) not in cells:
                stance[n, o] = (size[n], size[n])
                continue
            active = [i for i, count in enumerate((o, a)) if count]
            x = np.stack([o * opponent, a * ally], axis=1)[:, active]
            y = response[:, conditions.index((o, a))]
            gram = x.T @ x
            ridge = 0.1 * np.trace(gram) / len(active)
            values = np.full(2, size[n])
            if ridge > 0:
                values[active] = np.linalg.solve(
                    gram + ridge * np.eye(len(active)), x.T @ y + ridge * size[n])
            stance[n, o] = tuple(values)
    return size, stance


def meeting_table(cache, five_person_sets, size, stance):
    table = {}
    five = {tuple(key): value for key, value in zip(cache["five_keys"], cache["five_values"])}
    groups = chain(combinations(range(15), 3), [tuple(sorted(s)) for s in five_person_sets])
    for group in groups:
        states = stance_patterns(len(group))
        n = len(group) - 1
        values = np.empty((len(states), len(group), len(METHODS)))
        for position, focal in enumerate(group):
            peer_positions = [i for i in range(len(group)) if i != position]
            peers = [group[i] for i in peer_positions]
            own = states[:, position]
            others = states[:, peer_positions]
            opponents = others != own[:, None]
            n_opponents = opponents.sum(axis=1)
            baseline = cache["alpha"][focal, own]
            influence = np.stack([cache["pair"][focal, peer, 2 * own + others[:, j]] - baseline
                                  for j, peer in enumerate(peers)], axis=1)
            index = encode(np.column_stack((own, others)))
            observed = (cache["triple"][focal, *peers, index] if n == 2 else
                        np.asarray([five[(focal, *peers, int(i))] for i in index]))
            weights = np.asarray([stance[n, int(o)] for o in n_opponents])
            scaled = baseline + size[n] * influence.sum(axis=1)
            mixed = (baseline + weights[:, 0] * (influence * opponents).sum(axis=1)
                     + weights[:, 1] * (influence * ~opponents).sum(axis=1))
            logits = np.stack([observed, baseline + influence.sum(axis=1),
                               baseline + influence.mean(axis=1), scaled, mixed, scaled], axis=1)
            values[:, position, :] = logits
        values[:, :, :-1] = sigmoid(values[:, :, :-1])
        table[group] = values
    return table


def aggregate(table, meetings, members, states, weighting="person"):
    sums = np.zeros((len(states), len(members), len(METHODS)))
    counts = np.zeros(len(members), dtype=int)
    positions = {member: i for i, member in enumerate(members)}
    for meeting in meetings:
        group = tuple(sorted(meeting))
        local = [positions[member] for member in group]
        sums[:, local, :] += table[group][encode(states[:, local])]
        counts[local] += 1
    means = sums / counts[None, :, None]
    means[:, :, -1] = sigmoid(means[:, :, -1])
    return means.mean(axis=1) if weighting == "person" else np.average(means, axis=1, weights=counts)


def jensen_gap(logits):
    values = np.asarray(logits)
    gap = sigmoid(values).mean(axis=-1) - sigmoid(values.mean(axis=-1))
    bound = np.sqrt(3) / 36 * values.var(axis=-1)
    return gap, bound
