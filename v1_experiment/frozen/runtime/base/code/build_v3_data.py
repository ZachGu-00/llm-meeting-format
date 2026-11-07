"""Source-only v3 materials. Does not read focal-model outcomes or credentials."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V2 = PROJECT / 'experiments/voices_nonadditivity_v2_20261030'
DATA = ROOT / 'data'
SOURCE = PROJECT / 'work/voices_or_arguments_review_20261029/feasibility/kpa'
sys.path.insert(0, str(V2 / 'code'))
import render_v2 as original

SEED = 'voices-v3-data-20261030-v1'
ROLES = original.ROLES
NEUTRALS = [
    'I would like to hear more before I make a decision.',
    'I can see points on both sides of this discussion.',
    'I am still thinking about the different comments made here.',
    'I have not decided which position I agree with yet.',
    'I would appreciate some time to consider what everyone said.',
    'I am listening to the discussion and keeping an open mind.',
    'I do not have a firm view on this proposal yet.',
    'I want to understand the discussion before stating my position.',
    'I am considering the comments without taking a side yet.',
    'I would like to keep listening before reaching a conclusion.',
    'I am undecided and am reflecting on the points raised.',
    'I have heard the comments but have not chosen a position.',
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

def rank(*parts): return hashlib.sha256((SEED+'|'+ '|'.join(map(str,parts))).encode()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def receipt(path,n=None):
    d={'path':str(Path(path).relative_to(ROOT)).replace('\\','/'),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
    if n is not None:d['n_rows']=n
    return d

def write_rows(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.building')
    seen=set();n=0
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:
            assert r['id'] not in seen, r['id'];seen.add(r['id'])
            assert r['positive_letter'] in ('A','B')
            assert set(r['candidates'])=={'A','B'}
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n');n+=1
    temp.replace(path)
    return receipt(path,n)

def build_core():
    DATA.mkdir(parents=True,exist_ok=True)
    outputs={}
    for filename in ('calibration_prompts.jsonl','pilot7_prompts.jsonl','development7_prompts.jsonl',
                     'development_items.json','pilot_items.json','holdout_items.json'):
        p=DATA/filename;shutil.copyfile(V2/'data'/filename,p)
        assert sha(p)==sha(V2/'data'/filename)
        outputs[filename]=receipt(p)
    parents=load(V2/'data/holdout_items.json'); accepted=load(V2/'holdout_materials/accepted_holdout_items.json')
    old=load(V2/'data/confirmation_entry20/confirmation_entry20_items.json')
    parent_by={x['item_id']:x for x in parents};accepted_ids={x['item_id'] for x in accepted};old_ids={x['item_id'] for x in old}
    assert len(parent_by)==200 and len(accepted_ids)==64 and len(old_ids)==20 and old_ids<=accepted_ids
    assert all(x==parent_by[x['item_id']] for x in accepted+old)
    sets={'holdout200':set(parent_by),'semantic64':accepted_ids,'remaining180':set(parent_by)-old_ids,
          'semantic_remaining44':accepted_ids-old_ids}
    files={name:(DATA/f'e1_{name}_prompts.jsonl').open('wb') for name in sets};counts=collections.Counter()
    try:
        with (V2/'data/holdout32_prompts.jsonl').open('rb') as src:
            for line in src:
                r=json.loads(line)
                for name,ids in sets.items():
                    if r['item_id'] in ids:files[name].write(line);counts[name]+=1
    finally:
        for f in files.values():f.close()
    for name,ids in sets.items():
        assert counts[name]==len(ids)*622
        p=DATA/f'e1_{name}_prompts.jsonl';outputs[p.name]=receipt(p,counts[name])
        dump(DATA/f'e1_{name}_items.json',[x for x in parents if x['item_id'] in ids])
    assert sha(DATA/'e1_holdout200_prompts.jsonl')==sha(V2/'data/holdout32_prompts.jsonl')
    manifest={'seed':SEED,'builder_sha256':sha(__file__),'source_prompt_sha256':sha(V2/'data/holdout32_prompts.jsonl'),
       'accepted_materials_sha256':sha(V2/'holdout_materials/accepted_holdout_items.json'),
       'accepted_overlay_sha256':sha(V2/'holdout_materials/validation_overlay.json'),
       'read_model_outputs':False,'original_lines_and_ids_preserved':True,'outputs':outputs,
       'reused20_ids':sorted(old_ids),'Qwen4_and_LFM_confirmatory_item_ids':sorted(sets['remaining180']),
       'reuse_requires_same_engine_prompt_template_batch_token_lock_and_model_revision':True,
       'prior20_R1_reuse_source':'V2/results/<model>/confirmation20_R1.jsonl',
       'R1_full64_for_new_models':True,'R1_remaining44_only_if_prior20_execution_identity_matches':True}
    dump(DATA/'core_data_manifest.json',manifest)
    print(json.dumps({'stage':'core','outputs':{k:v.get('n_rows') for k,v in outputs.items()}},ensure_ascii=False))

def source_arguments():
    result=[];files=[];seen=set()
    for split,folder in [('train','kpm_data'),('dev','kpm_data'),('test','test_data')]:
        ap=SOURCE/folder/f'arguments_{split}.csv';kp=SOURCE/folder/f'key_points_{split}.csv';lp=SOURCE/folder/f'labels_{split}.csv'
        files.extend([ap,kp,lp]);labels=collections.defaultdict(dict)
        with kp.open(encoding='utf-8-sig') as f:kps={r['key_point_id']:r for r in csv.DictReader(f)}
        with lp.open(encoding='utf-8-sig') as f:
            for idx,r in enumerate(csv.DictReader(f),2):
                assert r['key_point_id'] not in labels[r['arg_id']]
                labels[r['arg_id']][r['key_point_id']]={'value':int(r['label']),'row':idx}
        with ap.open(encoding='utf-8-sig') as f:
            for idx,r in enumerate(csv.DictReader(f),2):
                labs=labels[r['arg_id']];positive=[k for k,v in labs.items() if v['value']==1]
                expected={k for k,v in kps.items() if (v['topic'],v['stance'])==(r['topic'],r['stance'])}
                normalized=' '.join(r['argument'].casefold().split());words=len(r['argument'].split())
                if len(positive)!=1 or set(labs)!=expected or not 8<=words<=30 or normalized in seen:continue
                seen.add(normalized)
                result.append({'argument_id':f'{split}:{r["arg_id"]}','argument':r['argument'],'topic':r['topic'],
                    'stance':int(r['stance']),'kp_id':f'{split}:{positive[0]}','kp_text':kps[positive[0]]['key_point'],
                    'word_count':words,'text_sha256':hashlib.sha256(r['argument'].encode()).hexdigest(),
                    'author_id':None,'author_independence':'unavailable_in_source',
                    'provenance':{'argument_file':str(ap.relative_to(PROJECT)).replace('\\','/'),'argument_csv_row_including_header':idx,
                       'label_file':str(lp.relative_to(PROJECT)).replace('\\','/'),'kp_labels':labs,
                       'positive_kp_count':1,'complete_topic_stance_labels':True,'text_is_original_verbatim':True}})
    return result,{str(p.relative_to(PROJECT)).replace('\\','/'):sha(p) for p in files}

def build_agents():
    source,hashes=source_arguments();dev=load(V2/'data/development_items.json');hold=load(V2/'data/holdout_items.json')
    topics={x['topic'] for x in dev};used={a['argument_id'] for x in dev for a in [x['focal_argument'],*x['roles'].values()]}
    audit=[]
    for t in sorted(topics):
        eligible={str(s):[a for a in source if a['topic']==t and a['stance']==s] for s in (-1,1)}
        fresh={s:[a for a in rows if a['argument_id'] not in used] for s,rows in eligible.items()}
        audit.append({'topic':t,'eligible_per_stance':{s:len(v) for s,v in eligible.items()},
                      'unused_per_stance':{s:len(v) for s,v in fresh.items()},'can_avoid_all_dev_arguments':all(len(v)>=15 for v in fresh.values()),
                      'eligible':all(len(v)>=15 for v in eligible.values()),
                      'minimum_development_arguments_needed':sum(max(0,15-len(v)) for v in fresh.values())})
    selected=[r['topic'] for r in sorted((r for r in audit if r['eligible']),key=lambda r:(r['minimum_development_arguments_needed'],rank('E4_topic',r['topic'])))[:6]]
    assert len(selected)==6 and not set(selected)&{x['topic'] for x in hold}
    out=[]
    for ti,t in enumerate(selected):
        by={s:sorted((a for a in source if a['topic']==t and a['stance']==s),key=lambda a:(a['argument_id'] in used,rank('E4_arg',a['argument_id'])))[:15] for s in(-1,1)}
        agents=[{'agent_id':j,'arguments':{str(s):{**by[s][j],'used_in_v2_development':by[s][j]['argument_id'] in used} for s in(-1,1)}} for j in range(15)]
        assert len({a['argument_id'] for agent in agents for a in agent['arguments'].values()})==30
        out.append({'topic_id':f'V3_E4_TOPIC_{ti+1:02d}','topic':t,'agents':agents})
    dump(DATA/'e4_agents.json',out)
    dump(DATA/'e4_agent_manifest.json',{'seed':SEED,'builder_sha256':sha(__file__),'source_file_sha256':hashes,
       'topic_eligibility':audit,'selected_topics':selected,'selection_rule':'6 eligible topics minimizing number of reused development arguments; fixed SHA breaks ties. Within stance unused arguments first, then fixed SHA.',
       'n_topics':6,'n_agents_per_topic':15,'n_source_arguments':180,
       'development_argument_overlap':sum(a['used_in_v2_development'] for t in out for agent in t['agents'] for a in agent['arguments'].values()),'E1_topic_overlap':0,
       'human_stance_labels':True,'author_identity_not_available':True,'model_outputs_read':False,'output':receipt(DATA/'e4_agents.json')})
    print(json.dumps({'stage':'E4_agents','topics':selected,'arguments':180},ensure_ascii=False))

def envelope(item,experiment,condition,mask,order_index,label_map,lines,metadata=None,heading=None):
    options={letter:original.stance_phrase(s) for letter,s in zip('AB',(1,-1) if label_map==0 else(-1,1))}
    positive=next(l for l in 'AB' if options[l]==original.stance_phrase(-item['focal_stance']))
    content=(f'Proposal: "{item["topic"]}"\n\nEarlier in this discussion, you wrote: "{item["focal_statement"]}"\n\n'+
             (heading or 'Comments from the other participants in this round:')+'\n'+'\n'.join(lines)+
             f'\n\nWhat is your position now?\n(A) {options["A"]}\n(B) {options["B"]}\nAnswer with a single letter.')
    row={'id':f'{item["item_id"]}_{experiment}_{condition}_p{order_index:02d}_l{label_map}',
      'item_id':item['item_id'],'topic':item['topic'],'split':item['split'],'domain':'D1',
      'experiment':experiment,'condition':condition,'subset_mask':mask,'permutation':order_index,'label_map':label_map,
      'positive_letter':positive,'focal_stance':item['focal_stance'],'candidates':options,
      'messages':[{'role':'system','content':original.SYSTEM},{'role':'user','content':content}],
      'template_variant':experiment,'empty_slot_replication':False,'replicate_slot_indices':[order_index],
      'ally_relation':item['ally_relation'],'validation_status':'two_API_judges_all_four_pairs_approved_v2_final_overlay'}
    if metadata:row.update(metadata)
    return row

def role_body(item,role,stance_only=False):
    peer=item['roles'][role]
    return original.stance_phrase(peer['stance'])+('' if stance_only else ' '+peer['argument'])

def e2a(items):
    masks=[m for m in range(32) if m.bit_count()<=2]
    requested={1,3,5,8,9}
    pool_order=sorted(range(12),key=lambda i:rank('neutral_pool_index',i))
    for ix,item in enumerate(items):
        neutral_indices=[pool_order[(ix+4*j)%12] for j in range(3)]
        for mask,n in itertools.product(masks,range(4)):
            bodies={r:role_body(item,r) for bit,r in enumerate(ROLES) if mask&(1<<bit)}
            empty=[r for r in ROLES if r not in bodies]
            neutral_roles=dict(zip(empty[:n],neutral_indices[:n]))
            for p,order in enumerate(original.permutations()):
                lines=[f'Participant {k}: '+(bodies[r] if r in bodies else NEUTRALS[neutral_roles[r]] if r in neutral_roles else '(has not commented yet)') for k,r in enumerate(order,1)]
                for label in (0,1):
                    yield envelope(item,'E2a',f'm{mask:02d}_n{n}',mask,p,label,lines,
                        {'neutral_count':n,'neutral_pool_indices':neutral_indices[:n],
                         'active_roles':list(bodies),'slot_role_order':order,'requested_original_condition':mask in requested,
                         'support_cell_for_C':mask not in requested})

def e2b(items):
    for item in items:
        for mask in range(32):
            active=[r for bit,r in enumerate(ROLES) if mask&(1<<bit)]
            orders=[]
            for order in original.permutations():
                reduced=[r for r in order if r in active]
                if reduced not in orders:orders.append(reduced)
            for p,order in enumerate(orders):
                lines=[f'Participant {k}: {role_body(item,r)}' for k,r in enumerate(order,1)]
                for label in (0,1):
                    yield envelope(item,'E2b',f'm{mask:02d}',mask,p,label,lines,
                       {'active_roles':active,'slot_role_order':order,'n_unique_orders':len(orders),'placeholder_removed':True})

def e2c(items):
    for item in items:
        for mask in range(32):
            active=[r for bit,r in enumerate(ROLES) if mask&(1<<bit)]
            for p,order in enumerate(original.permutations()[:5]):
                lines=[role_body(item,r) if r in active else '(has not commented yet)' for r in order]
                for label in (0,1):
                    yield envelope(item,'E2c',f'm{mask:02d}',mask,p,label,lines,
                       {'active_roles':active,'slot_role_order':order,'speaker_labels_removed':True},
                       'Comments posted in this round (authors not shown)')

def e2d(items):
    for item in items:
        for nopp,nally in itertools.product(range(4),range(3)):
            mask=sum(1<<i for i in range(nopp))+sum(1<<(3+i) for i in range(nally))
            active=[r for bit,r in enumerate(ROLES) if mask&(1<<bit)]
            for p,order in enumerate(original.permutations()):
                lines=[f'Participant {k}: '+(role_body(item,r,True) if r in active else '(has not commented yet)') for k,r in enumerate(order,1)]
                for label in (0,1):
                    yield envelope(item,'E2d',f'o{nopp}_a{nally}',mask,p,label,lines,
                       {'n_opponents':nopp,'n_allies':nally,'active_roles':active,'slot_role_order':order,'stance_only':True})

def extension_peers(items,source):
    out={}
    for item in items:
        used={item['focal_argument_id'],*(a['argument_id'] for a in item['roles'].values())}
        extra={s:sorted((a for a in source if a['topic']==item['topic'] and a['stance']==s and a['argument_id'] not in used),key=lambda a:rank('E2e_extra',item['item_id'],a['argument_id'])) for s in (-1,1)}
        assert len(extra[-item['focal_stance']])>=3 and len(extra[item['focal_stance']])>=1
        peers=[item['roles'][r] for r in ('A','Aprime','C')]+extra[-item['focal_stance']][:3]+[item['roles'][r] for r in ('D','Dprime')]+extra[item['focal_stance']][:1]
        assert len({a['argument_id'] for a in peers})==9
        out[item['item_id']]=peers
    return out

def e2e(items,extra):
    roles=['O1','O2','O3','O4','O5','O6','L1','L2','L3']
    for item in items:
        peers=extra[item['item_id']]
        for nopp,nally in itertools.product(range(7),range(4)):
            active=list(range(nopp))+list(range(6,6+nally));mask=sum(1<<i for i in active)
            for p in range(6):
                order=list(range(p,9))+list(range(p))
                lines=[f'Participant {k}: '+(original.stance_phrase(peers[j]['stance'])+' '+peers[j]['argument'] if j in active else '(has not commented yet)') for k,j in enumerate(order,1)]
                for label in (0,1):
                    yield envelope(item,'E2e',f'o{nopp}_a{nally}',mask,p,label,lines,
                       {'n_opponents':nopp,'n_allies':nally,'n_slots':9,'active_roles':[roles[j] for j in active],
                        'active_argument_ids':[peers[j]['argument_id'] for j in active],
                        'slot_role_order':[roles[j] for j in order],'slot_balance':'six_of_nine_cyclic_orders_not_fully_balanced'})

def build_diagnostics():
    items=sorted(load(V2/'holdout_materials/accepted_holdout_items.json'),key=lambda x:rank('neutral_assignment',x['item_id']))
    source,source_hashes=source_arguments();extra=extension_peers(items,source)
    dump(DATA/'e2_neutral_pool.json',{'sentences':NEUTRALS,'word_counts':[len(s.split()) for s in NEUTRALS],
        'assignment':'fixed hash item order and pool order, indices pool_order[(item_index+4*j)%12]; same identities across counts and subset conditions; exposure count differs by at most1 for every neutral_count',
        'exposure_counts_by_neutral_count':{str(n):dict(collections.Counter(sorted(range(12),key=lambda k:rank('neutral_pool_index',k))[(i+4*j)%12] for i in range(len(items)) for j in range(n))) for n in (1,2,3)}})
    assert all(8<=len(s.split())<=30 for s in NEUTRALS)
    dump(DATA/'e2e_extended_peers.json',extra)
    outputs={}
    for name,rows in [('E2a',e2a(items)),('E2b',e2b(items)),('E2c',e2c(items)),('E2d',e2d(items)),('E2e',e2e(items,extra))]:
        outputs[name]=write_rows(DATA/(name.lower()+'_prompts.jsonl'),rows)
    dump(DATA/'e2_manifest.json',{'seed':SEED,'builder_sha256':sha(__file__),'n_items':64,'item_ids':[x['item_id'] for x in items],
       'source_file_sha256':source_hashes,'outputs':outputs,'model_outputs_read':False,
       'amendments':[
         'E2a adds empty, all5singletons and all10pairs at every neutral count to identify C; original five listed conditions remain flagged.',
         'E2e uses9slots in every cell because the full requested6opponents+3allies cell cannot fit8slots. Root approved before measurement.',
         'E2e six cyclic arrangements follow proposal count but are not fully position balanced across nine slots.',
         'E2b derives at most10 distinct present-only sequences from original dihedral order list; no repeated order is counted twice.',
         'E2c keeps placeholders and removes speaker identifiers only, isolating the author-label manipulation.',
         'E2a/c/d/e physically retain all specified permutation rows including duplicate empty prompts; unlike E1 no implicit replication is needed.'
       ],'E2a_C_estimand':'conditional-on-n-neutral-comments interactions with the matching neutral-only baseline',
       'E2d_C_estimand':'stance-count representative cells; no distinction between same-stance argument identities',
       'E2d_matched_comparison':'For prompt ablation compare with the SAME representative subset masks in E1; pooled E1 count curves also average other role geometries.',
       'E2e_C_estimand':'count-based analogue with9-role lattice not full512subsets; do not use5-role full-lattice analyzer'})
    print(json.dumps({'stage':'E2','outputs':{k:v['n_rows'] for k,v in outputs.items()}}))

def validate_materials():
    checks=0;totals={}
    for name in ('e2a','e2b','e2c','e2d','e2e'):
        by_item=collections.Counter();seen=set();previous=None
        with (DATA/(name+'_prompts.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);assert r['id'] not in seen;seen.add(r['id']);by_item[r['item_id']]+=1
                content=r['messages'][-1]['content'];body=content.split('round:',1)[-1] if name!='e2c' else content.split('(authors not shown)',1)[-1]
                body=body.split('\n\nWhat is your position now?')[0].strip();lines=body.splitlines() if body else []
                if name=='e2a':
                    assert len(lines)==5 and content.count('(has not commented yet)')==5-r['subset_mask'].bit_count()-r['neutral_count']
                    assert sum(content.count(NEUTRALS[i]) for i in r['neutral_pool_indices'])==r['neutral_count']
                elif name=='e2b':assert len(lines)==r['subset_mask'].bit_count() and '(has not commented yet)' not in content
                elif name=='e2c':assert len(lines)==5 and all(not x.startswith('Participant ') for x in lines)
                elif name=='e2d':
                    assert len(lines)==5 and content.count('(has not commented yet)')==5-r['n_opponents']-r['n_allies']
                    assert all(x.split(': ',1)[1] in ('I support this proposal.','I oppose this proposal.','(has not commented yet)') for x in lines)
                else:
                    assert len(lines)==9 and content.count('(has not commented yet)')==9-r['n_opponents']-r['n_allies']
                    assert r['subset_mask'].bit_count()==r['n_opponents']+r['n_allies']
                if r['label_map']==0:previous=r
                else:
                    assert previous['condition']==r['condition'] and previous['permutation']==r['permutation']
                    assert previous['positive_letter']!=r['positive_letter']
                    assert previous['candidates']['A']==r['candidates']['B'] and previous['candidates']['B']==r['candidates']['A']
                checks+=1
        assert len(by_item)==64 and len(set(by_item.values()))==1
        totals[name]={'n_rows':len(seen),'n_items':64,'rows_per_item':next(iter(by_item.values()))}
    agents=load(DATA/'e4_agents.json');manifest=load(DATA/'e4_agent_manifest.json');dev=load(V2/'data/development_items.json');hold=load(V2/'data/holdout_items.json')
    assert len(agents)==6 and {t['topic'] for t in agents}<={i['topic'] for i in dev}
    assert not {t['topic'] for t in agents}&{i['topic'] for i in hold}
    source,_=source_arguments();raw={a['argument_id']:a for a in source};ids=[]
    for t in agents:
        assert len(t['agents'])==15
        for agent in t['agents']:
            for stance,a in agent['arguments'].items():
                assert a['stance']==int(stance) and a['topic']==t['topic']
                assert {k:v for k,v in a.items() if k!='used_in_v2_development'}==raw[a['argument_id']]
                assert a['provenance']['positive_kp_count']==1 and a['provenance']['complete_topic_stance_labels']
                ids.append(a['argument_id']);checks+=1
    assert len(ids)==len(set(ids))==180
    dump(DATA/'material_validation.json',{'status':'pass','checks':checks,'E2':totals,'E4_agents':90,'E4_arguments':180,
        'E4_development_argument_overlap':manifest['development_argument_overlap'],'builder_sha256':sha(__file__),
        'checked_text_against_original_sources':True,'read_model_outputs':False})
    print(json.dumps({'stage':'validation','status':'pass','checks':checks}))

def e4_group_metadata(topic):
    ids=list(range(15));g6=sorted(sorted(ids,key=lambda j:rank('E4_G6',topic['topic_id'],j))[:6])
    shuffled=sorted(ids,key=lambda j:rank('E4_five_partition',topic['topic_id'],j))
    fives=[sorted(shuffled[i:i+5]) for i in range(0,15,5)]
    left=[(0,2,4),(0,3,5),(1,2,5),(1,3,4)]
    right=[(0,2,5),(0,3,4),(1,2,4),(1,3,5)]
    pair_counts=lambda blocks:collections.Counter(pair for block in blocks for pair in itertools.combinations(sorted(block),2))
    pasch_left=[[g6[j] for j in block] for block in left];pasch_right=[[g6[j] for j in block] for block in right]
    assert pair_counts(pasch_left)==pair_counts(pasch_right)
    for group in fives:assert pair_counts([group]*3)==pair_counts(itertools.combinations(group,3))
    return {'topic_id':topic['topic_id'],'topic':topic['topic'],'g6_agent_ids':g6,
      'g6_pasch_left':pasch_left,'g6_pasch_right':pasch_right,'five_person_sets':fives,
      'five_sets_form_disjoint_partition_of15':True,'pairwise_projection_checks_pass':True}

def e4_render(topic,kind,j,peers,states,p,label,group_index=None):
    agents={a['agent_id']:a for a in topic['agents']};peers=sorted(peers);sj=states[j]
    item={'item_id':f'{topic["topic_id"]}_j{j:02d}','topic':topic['topic'],'split':'E4_development_topics',
          'focal_stance':sj,'focal_statement':agents[j]['arguments'][str(sj)]['argument'],'ally_relation':None}
    if kind=='baseline':positions=[];ordered=[]
    elif kind=='pair':positions=[1 if p==0 else 3];ordered=peers
    elif kind=='triple':positions=[1,3];ordered=peers if p==0 else peers[::-1]
    elif kind=='five':positions=[0,1,3,4];ordered=peers if p==0 else peers[::-1]
    else:raise ValueError(kind)
    assigned=dict(zip(positions,ordered));lines=[]
    for slot in range(5):
        if slot in assigned:
            who=assigned[slot];a=agents[who]['arguments'][str(states[who])]
            body=original.stance_phrase(states[who])+' '+a['argument']
        else:body='(has not commented yet)'
        lines.append(f'Participant {slot+1}: {body}')
    state_order=[j,*peers];bits=''.join('1' if states[a]==1 else '0' for a in state_order)
    condition=f'{kind}_g{group_index if group_index is not None else "na"}_j{j:02d}_k'+('-'.join(f'{a:02d}' for a in peers) or 'none')+f'_s{bits}'
    row=envelope(item,'E4',condition,None,p,label,lines)
    row.update(topic_id=topic['topic_id'],cache_kind=kind,focal_agent_id=j,peer_agent_ids=peers,
        focal_stance=sj,peer_stances=[states[a] for a in peers],state_agent_order=state_order,
        state_by_agent={str(a):states[a] for a in state_order},state_bits_focal_then_sorted_peers=bits,
        five_group_index=group_index,positive_letter='A' if label==0 else 'B',
        positive_direction='global_support_stance_+1; negate_for_tau_-1',
        focal_argument_id=agents[j]['arguments'][str(sj)]['argument_id'],
        peer_argument_ids=[agents[a]['arguments'][str(states[a])]['argument_id'] for a in peers],
        slot_agent_ids=[assigned.get(k) for k in range(5)],template_variant='E4_v2_F1',
        validation_status='original_human_stance_labels_single_positive_complete_KP_coverage',
        input_authorship_independence='unavailable_in_source')
    return row

def e4_rows(topic,meta):
    for j in range(15):
        others=[k for k in range(15) if k!=j]
        for peers in itertools.combinations(others,2):
            for values in itertools.product((-1,1),repeat=3):
                states=dict(zip([j,*peers],values))
                for p,label in itertools.product(range(2),range(2)):yield e4_render(topic,'triple',j,peers,states,p,label)
        for k in others:
            for sj,sk,p,label in itertools.product((-1,1),(-1,1),range(2),range(2)):
                yield e4_render(topic,'pair',j,[k],{j:sj,k:sk},p,label)
        for sj,label in itertools.product((-1,1),range(2)):yield e4_render(topic,'baseline',j,[],{j:sj},0,label)
    for gi,group in enumerate(meta['five_person_sets']):
        for values in itertools.product((-1,1),repeat=5):
            states=dict(zip(group,values))
            for j in group:
                peers=[k for k in group if k!=j]
                for p,label in itertools.product(range(2),range(2)):yield e4_render(topic,'five',j,peers,states,p,label,gi)

def validate_e4_topic(path,topic,meta):
    kinds=collections.Counter();keys=set();pairs=collections.defaultdict(set);agents={a['agent_id']:a for a in topic['agents']}
    for line in Path(path).open(encoding='utf-8'):
        r=json.loads(line);kind=r['cache_kind'];j=r['focal_agent_id'];peers=r['peer_agent_ids'];states={int(k):v for k,v in r['state_by_agent'].items()}
        key=(kind,r['five_group_index'],j,tuple(peers),tuple(states[k] for k in [j,*peers]),r['permutation'],r['label_map'])
        assert key not in keys;keys.add(key);kinds[kind]+=1
        assert j not in peers and peers==sorted(set(peers)) and set(states)=={j,*peers}
        assert len(peers)=={'baseline':0,'pair':1,'triple':2,'five':4}[kind]
        assert r['positive_letter']==('A' if r['label_map']==0 else 'B')
        assert r['candidates'][r['positive_letter']]=='I support this proposal.'
        assert r['focal_argument_id']==agents[j]['arguments'][str(states[j])]['argument_id']
        assert len([a for a in r['slot_agent_ids'] if a is not None])==len(peers)
        assert {a for a in r['slot_agent_ids'] if a is not None}==set(peers)
        body=r['messages'][-1]['content']
        assert body.count('(has not commented yet)')==5-len(peers)
        for peer,aid in zip(peers,r['peer_argument_ids']):
            assert aid==agents[peer]['arguments'][str(states[peer])]['argument_id']
            assert agents[peer]['arguments'][str(states[peer])]['argument'] in body
        pairs[key[:-1]].add(r['label_map'])
    assert all(v=={0,1} for v in pairs.values())
    assert dict(kinds)=={'triple':43680,'pair':3360,'baseline':60,'five':1920}
    assert len(keys)==49020 and meta['pairwise_projection_checks_pass']
    return {'status':'pass','rows':len(keys),'counts':dict(kinds),'unique_cache_cells_and_label_pairs_verified':True,
        'focal_and_peer_source_texts_verified':True,'global_support_direction_verified':True,'projection_checks_pass':True}

def build_e4(limit=None):
    topics=load(DATA/'e4_agents.json');out=[]
    for topic in topics[:limit]:
        meta=e4_group_metadata(topic);path=DATA/'e4_cache'/f'{topic["topic_id"]}_prompts.jsonl'
        info=write_rows(path,e4_rows(topic,meta));validation=validate_e4_topic(path,topic,meta)
        out.append({**meta,'output':info,'validation':validation})
        print(json.dumps({'stage':'E4_cache','topic_id':topic['topic_id'],'n_rows':info['n_rows'],'validation':validation['status']}))
    dump(DATA/'e4_cache_manifest.json',{'seed':SEED,'builder_sha256':sha(__file__),'agents_sha256':sha(DATA/'e4_agents.json'),
       'source_agent_manifest_sha256':sha(DATA/'e4_agent_manifest.json'),'n_topics':len(out),'n_rows':sum(x['output']['n_rows'] for x in out),
       'topics':out,'model_outputs_read':False,'positive_direction':'global_support_+1; tau=-1 uses negated score',
       'slot_policy':{'triple':'peers in slots2/4 with both orders','pair':'peer in slot2 or4',
          'baseline':'all5placeholders','five':'four peers in slots1/2/4/5, ascending and descending agent order'},
       'G6_selection':'six lowest fixed hash ranked agents, sorted for canonical Pasch labels',
       'five_set_selection':'fixed hash shuffle of15 agents partitioned into three disjoint sets of5',
       'baseline_count':60,'pair_count':3360,'triple_count':43680,'five_count':1920,
       'note':'E4c has only4peers per focal in a5person group; all32whole_group_states are represented.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['core','agents','diagnostics','validate','e4','all'],default='all');p.add_argument('--topic-limit',type=int);a=p.parse_args()
    if a.stage in ('core','all'):build_core()
    if a.stage in ('agents','all'):build_agents()
    if a.stage in ('diagnostics','all'):build_diagnostics()
    if a.stage in ('validate','all'):validate_materials()
    if a.stage in ('e4','all'):build_e4(a.topic_limit)
if __name__=='__main__':main()
