"""The base, regroup, and resize schedules used in the paper."""
from collections import Counter
from itertools import combinations
import numpy as np


TRIADS = list(combinations(range(5), 3))
FIVE = tuple(range(5))
PASCH = [(0, 2, 4), (0, 3, 5), (1, 2, 5), (1, 3, 4)]
REGROUPED_PASCH = [(0, 2, 5), (0, 3, 4), (1, 2, 4), (1, 3, 5)]


def schedules():
    return {"base": TRIADS * 2 + PASCH * 5,
            "regroup": TRIADS * 2 + REGROUPED_PASCH * 5,
            "resize": [FIVE] * 6 + PASCH * 5}


def pair_contacts(meetings):
    return Counter(pair for meeting in meetings for pair in combinations(sorted(meeting), 2))


def moved_contacts(first, second):
    return sum(count * len(group) * (len(group) - 1) // 2
               for group, count in (Counter(first) - Counter(second)).items())


def stance_patterns(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(int)


def encode(bits):
    return bits @ (1 << np.arange(bits.shape[1] - 1, -1, -1))


def check_design():
    design = schedules()
    assert pair_contacts(design["base"]) == pair_contacts(design["regroup"]) == pair_contacts(design["resize"])
    for name in ("regroup", "resize"):
        assert moved_contacts(design["base"], design[name]) == 60
        assert moved_contacts(design[name], design["base"]) == 60
    return {"pair_contacts_equal": True, "moved_dyadic_exposures_each_side": 60,
            "meeting_counts": {name: len(groups) for name, groups in design.items()}}
