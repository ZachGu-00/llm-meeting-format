"""Complete empty/single-peer caches; higher-order outcomes never enter selection."""
from collections import defaultdict
import itertools
import math
import numpy as np

def cache(rows,n):
    cells=defaultdict(dict)
    for row in rows:
        if row['cache_kind'] not in ('baseline','pair'):continue
        if row['status']!='ok':raise ValueError('Unclosed low-order observation')
        j=row['focal_agent_id'];peers=tuple(row['peer_agent_ids']);kind=row['cache_kind']
        if not 0<=j<n or len(peers)!=(0 if kind=='baseline' else 1) or any(k==j or not 0<=k<n for k in peers):
            raise ValueError('Invalid low-order focal/peer scope')
        if any(s not in (-1,1) for s in [row['focal_stance'],*row['peer_stances']]):raise ValueError('Invalid stance')
        bits=tuple(int(s==1) for s in [row['focal_stance'],*row['peer_stances']])
        if len(bits)!=len(peers)+1:raise ValueError('Peer state count differs')
        key=(j,peers,bits);render=(row['permutation'],row['label_map'])
        if render in cells[key]:raise ValueError('Duplicate low-order factorial cell')
        if row['positive_letter'] not in ('A','B'):raise ValueError('Invalid option orientation')
        y=float(row['R0_y_AB'])*(1 if row['positive_letter']=='A' else -1)
        if not math.isfinite(y):raise ValueError('Nonfinite native low-order score')
        cells[key][render]=(y,row['positive_letter'])
    expected={(j,(),(s,)) for j in range(n) for s in (0,1)}
    expected|={(j,(k,),(sj,sk)) for j in range(n) for k in range(n) if j!=k for sj,sk in itertools.product((0,1),repeat=2)}
    if set(cells)!=expected:raise ValueError('Incomplete empty/single-peer agent/state coverage')
    alpha=np.full((n,2),np.nan);pairs=np.full((n,n,4),np.nan)
    for (j,peers,bits),values in cells.items():
        positions=[0] if not peers else [0,1]
        if set(values)!={(p,l) for p in positions for l in (0,1)}:raise ValueError('Missing low-order position/label mate')
        if any(values[p,0][1]==values[p,1][1] for p in positions):raise ValueError('Label reversal did not reverse support')
        mean=float(np.mean([np.mean([values[p,l][0] for l in (0,1)]) for p in positions]))
        if not peers:alpha[j,bits[0]]=mean
        else:pairs[j,peers[0],2*bits[0]+bits[1]]=mean
    return {'n':n,'alpha':alpha,'pair':pairs,'triple':np.full((n,n,n,8),np.nan),'five':{},'orientation':'global_support',
            'measured_joint_group_outcomes_used':False}
