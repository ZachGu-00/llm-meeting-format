"""Check cached predictions against the frozen configuration-level export."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from .data import COLUMNS, topic_metadata
from .design import schedules, stance_patterns
from .predictors import aggregate, fit_weights, meeting_table, jensen_gap


def check_caches(root, frame):
    metadata = topic_metadata(root)
    states = stance_patterns(6)
    maximum, rows, jobs = 0.0, 0, 0
    largest_bound_excess = 0.0
    for key, group in frame.groupby(["stage", "model", "topic_id", "readout"], sort=True):
        stage, model, topic, readout = key
        meta = metadata[topic]
        size, stance = fit_weights(root, model, meta["topic"])
        path = root / "cache" / ("__".join(key) + ".npz")
        with np.load(path) as cache:
            table = meeting_table(cache, meta["five_person_sets"], size, stance)
            for _, mapping in group.groupby(["set_id", "ordered_agents"], sort=True):
                mapping = mapping.sort_values("support_bits")
                members = tuple(map(int, mapping.ordered_agents.iloc[0].split()))
                meetings = {name: [tuple(members[i] for i in room) for room in rooms]
                            for name, rooms in schedules().items()}
                values = {name: aggregate(table, rooms, members, states) for name, rooms in meetings.items()}
                incidence = {name: aggregate(table, rooms, members, states, "incidence")[:, 0]
                             for name, rooms in meetings.items()}
                errors = [np.max(np.abs(values[name][:, 0] - mapping[column].to_numpy()))
                          for name, column in (("base", "base_obs"), ("regroup", "regroup_obs"), ("resize", "resize_obs"))]
                for name, prefix in (("regroup", "seat"), ("resize", "size")):
                    difference = values[name] - values["base"]
                    for index, suffix in enumerate(COLUMNS.values()):
                        errors.append(np.max(np.abs(difference[:, index] - mapping[prefix + "_" + suffix].to_numpy())))
                    errors.append(np.max(np.abs(incidence[name] - incidence["base"] - mapping["delta_" + name + "_incidence"].to_numpy())))
                maximum = max(maximum, *errors)
                assert maximum < 1e-12, (key, maximum)
                rows += len(mapping)
            for core in meta["five_person_sets"]:
                members = sorted(core)
                patterns = stance_patterns(5)
                for position, focal in enumerate(members):
                    own = patterns[:, position]
                    baseline = cache["alpha"][focal, own]
                    peer_positions = [i for i in range(5) if i != position]
                    influence = {i: cache["pair"][focal, members[i], 2 * own + patterns[:, i]] - baseline
                                 for i in peer_positions}
                    logits = np.stack([baseline + size[2] * (influence[a] + influence[b])
                                       for a, b in combinations(peer_positions, 2)], axis=1)
                    gap, bound = jensen_gap(logits)
                    excess = np.max(np.abs(gap) - bound)
                    largest_bound_excess = max(largest_bound_excess, float(excess))
                    assert excess < 1e-12
        jobs += 1
        if jobs % 20 == 0:
            print(json.dumps({"cache_jobs_checked": jobs, "configurations_checked": rows}), flush=True)
    assert jobs == 162 and rows == 311040
    return {"cache_jobs_checked": jobs, "cached_configurations_checked": rows,
            "max_cached_prediction_error": float(maximum), "jensen_bound_max_positive_excess": largest_bound_excess}
