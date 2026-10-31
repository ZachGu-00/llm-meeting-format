"""Source-backed STS(15), exact pair projections, and small E4 designs."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/gap-packages/loops/v3.4.5/data/steiner.tbl'
LICENSE_URL='https://raw.githubusercontent.com/gap-packages/loops/v3.4.5/LICENSE'
PASCH_LEFT=((0,2,4),(0,3,5),(1,2,5),(1,3,4))
PASCH_RIGHT=((0,2,5),(0,3,4),(1,2,4),(1,3,5))

def pair_projection(blocks):
    return Counter(tuple(sorted(pair)) for block in blocks for pair in itertools.combinations(block,2))

def validate_sts(blocks,n=15):
    canonical=[tuple(sorted(b)) for b in blocks]
    if len(canonical)!=n*(n-1)//6 or len(set(canonical))!=len(canonical):
        raise ValueError('Wrong block count or repeated block')
    if any(len(set(b))!=3 or min(b)<0 or max(b)>=n for b in canonical):raise ValueError('Invalid triple')
    expected=Counter({pair:1 for pair in itertools.combinations(range(n),2)})
    if pair_projection(canonical)!=expected:raise ValueError('Not exactly one occurrence of every pair')
    return True

def invariant_incidence_graph(blocks):
    """Point colors use isomorphism-invariant cycles of two point-link matchings."""
    import networkx as nx
    third={}
    for block in blocks:
        for a,b in itertools.combinations(block,2):
            third[a,b]=third[b,a]=next(v for v in block if v not in (a,b))
    colors=defaultdict(list)
    for a,b in itertools.combinations(range(15),2):
        remaining=set(range(15))-{a,b,third[a,b]};lengths=[]
        while remaining:
            todo=[min(remaining)];component=set()
            while todo:
                v=todo.pop()
                if v in component:continue
                component.add(v);todo.extend((third[a,v],third[b,v]))
            remaining-=component;lengths.append(len(component))
        signature=tuple(sorted(lengths));colors[a].append(signature);colors[b].append(signature)
    graph=nx.Graph()
    for point in range(15):graph.add_node(point,color='P'+repr(sorted(colors[point])))
    for i,block in enumerate(blocks):
        graph.add_node(15+i,color='B');graph.add_edges_from((v,15+i) for v in block)
    return graph

def verify_nonisomorphic(systems):
    import networkx as nx
    groups=defaultdict(list);exact_checks=0
    for index,blocks in enumerate(systems):
        graph=invariant_incidence_graph(blocks)
        fingerprint=nx.weisfeiler_lehman_graph_hash(graph,node_attr='color',iterations=4)
        for prior_index,prior in groups[fingerprint]:
            exact_checks+=1
            if nx.is_isomorphic(graph,prior,node_match=lambda x,y:x['color']==y['color']):
                raise ValueError(f'Isomorphic systems {prior_index+1} and {index+1}')
        groups[fingerprint].append((index,graph))
    return {'nonisomorphic_count':len(systems),'invariant_hash_classes':len(groups),
            'exact_VF2_ambiguous_pairs_checked':exact_checks,'point_block_colors_preserved':True}

def fetch_and_validate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=urllib.request.urlopen(URL,timeout=45).read()
    (out/'steiner.tbl').write_bytes(raw)
    (out/'LOOPS_LICENSE').write_bytes(urllib.request.urlopen(LICENSE_URL,timeout=45).read())
    cleaned='\n'.join(line.split('#',1)[0] for line in raw.decode().splitlines())
    data=ast.literal_eval(cleaned.split(':=',1)[1].strip().rstrip(';'))
    strings=data[2][data[0].index(16)]
    if len(strings)!=80:raise ValueError('Expected all80 loop-order16 / STS-order15 records')
    systems=[]
    for columns in strings:
        if len(columns)!=3 or any(len(c)!=35 for c in columns):raise ValueError('Invalid source encoding')
        blocks=[tuple(int(c,16) for c in triple) for triple in zip(*columns)]
        validate_sts(blocks);systems.append(blocks)
    iso=verify_nonisomorphic(systems)
    if pair_projection(PASCH_LEFT)!=pair_projection(PASCH_RIGHT):raise AssertionError('Pasch projection mismatch')
    five=tuple(range(5));five_repeat=[five]*3;all_triples=list(itertools.combinations(five,3))
    if pair_projection(five_repeat)!=pair_projection(all_triples):raise AssertionError('E4c projection mismatch')
    result={'source_url':URL,'source_sha256':hashlib.sha256(raw).hexdigest(),
      'source_package':'GAP LOOPS 3.4.5, Nagy and Vojtechovsky; source credits Colbourn and Rosa, Triple Systems',
      'source_license':'GPL-2.0-or-later; original notice and license retained',
      'source_point_encoding':'hexadecimal0..e; order16 is Steiner loop order, STS has15 points',
      'n_systems':80,'n_points':15,'blocks_per_system':35,'all105_pairs_exactly_once':True,
      'nonisomorphism_validation':iso,'systems':[{'id':i+1,'blocks':b} for i,b in enumerate(systems)],
      'pasch':{'left':PASCH_LEFT,'right':PASCH_RIGHT,'pair_projections_identical':True},
      'E4c':{'five_person_meetings':five_repeat,'three_person_meetings':all_triples,'each_pair_count':3}}
    (out/'sts15_all80.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('systems','pasch','E4c')}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default=str(ROOT/'data/e4_graphs'))
    args=parser.parse_args();fetch_and_validate(args.output_dir)
