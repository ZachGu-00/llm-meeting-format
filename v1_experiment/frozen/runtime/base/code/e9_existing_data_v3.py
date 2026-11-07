"""CPU-only E9 label/slot diagnostics from closed E1 runs, with paired topic resampling.

This is only the existing-data label/position component of E9. It neither runs
template/numerical experiments nor changes selection, data or acceptance rules.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import numpy as np
import analyze_v3 as base

ROOT=Path(__file__).resolve().parents[1]
OLD_MODELS={'LiquidAI/LFM2.5-1.2B-Instruct','Qwen/Qwen3.5-4B'}
ROLES=('A','Aprime','C','D','Dprime')
SOURCE_RENDERER_SHA256='8a6fd94bdd66359e8428cfb89f67c25bf9f59ca3659dd43cf4064c1ae6399ff4'


class OriginalPromptContract:
    """Exact original render_v2 messages/options; embedded to avoid a legacy runtime dependency."""
    @staticmethod
    def permutations():
        forward=list(ROLES);reverse=forward[::-1]
        return [x[k:]+x[:k] for x in (forward,reverse) for k in range(5)]

    @staticmethod
    def render(item,mask,permutation,label_map):
        phrase=lambda stance:'I support this proposal.' if stance==1 else 'I oppose this proposal.'
        active={r for b,r in enumerate(ROLES) if mask&(1<<b)};comments=[]
        for k,role in enumerate(OriginalPromptContract.permutations()[permutation],1):
            peer=item['roles'][role]
            body=phrase(peer['stance'])+' '+peer['argument'] if role in active else '(has not commented yet)'
            comments.append(f'Participant {k}: {body}')
        options={letter:phrase(stance) for letter,stance in zip('AB',(1,-1) if label_map==0 else (-1,1))}
        body=(f'Proposal: "{item["topic"]}"\n\nEarlier in this discussion, you wrote: "{item["focal_statement"]}"\n\n'
            'Comments from the other participants in this round:\n'+'\n'.join(comments)+
            f'\n\nWhat is your position now?\n(A) {options["A"]}\n(B) {options["B"]}\nAnswer with a single letter.')
        return {'messages':[{'role':'system','content':'You are a participant in a small-group discussion about a policy proposal.\n'
            "Read the other participants' comments, then state your current position."},{'role':'user','content':body}],
            'candidates':options,'positive_letter':next(l for l in 'AB' if options[l]==phrase(-item['focal_stance']))}


def renderer(root):
    path=root.parents[1]/'experiments/voices_nonadditivity_v2_20261030/code/render_v2.py'
    if path.is_file() and base.sha(path)!=SOURCE_RENDERER_SHA256:raise ValueError('Local source renderer differs from inspected frozen builder')
    return OriginalPromptContract,path if path.is_file() else None


def local(path,root):
    text=str(path).replace('\\','/')
    for marker in ('/workspace/voices_nonadditivity_v3/','/voices_nonadditivity_v3_20261030/'):
        if marker in text:return root/text.split(marker,1)[1]
    p=Path(text);return p if p.is_absolute() else root/p


def expected_input(path,key,root,sources):
    manifest=root/'data/core_data_manifest.json';entry=base.read_json(manifest)['outputs'][key]
    if Path(path).resolve()!=(root/entry['path']).resolve() or base.sha(path)!=entry['sha256']:
        raise ValueError('Input is not the fixed original E1 source: '+str(path))
    sources[str(manifest.resolve())]=base.sha(manifest)


def load_panel(inputs,results,items,render,readout,root,sources):
    """Stream physical rows; broadcast only the verified, semantically identical empty prompt."""
    rp=Path(str(results)+'.run.json');sp=Path(str(results)+'.summary.json')
    run,summary=base.read_json(rp),base.read_json(sp)
    if (run['readout'] not in (readout,'R0_R1') or summary.get('exact_id_and_count_equality') is not True
        or any(summary.get(k)!=v for k,v in run.items()) or run['n_input']!=622*len(items)
        or summary['output_bytes']!=Path(results).stat().st_size):
        raise ValueError('Readout/run/physical panel identity differs from full original E1')
    ids=sorted(items);index={item:i for i,item in enumerate(ids)}
    values=np.full((len(ids),32,10,2),np.nan);observed=np.zeros_like(values,dtype=bool)
    record_ids=set();empty_checks=0;count=0
    for row in base.read_closed(inputs,results):
        item=row['item_id'];mask=row['subset_mask'];p=row['permutation'];label=row['label_map'];rid=row.get('record_id',row['id'])
        if (item not in index or not isinstance(mask,int) or not 0<=mask<32 or not isinstance(p,int) or not 0<=p<10
            or label not in (0,1) or (mask==0 and p!=0) or rid in record_ids):
            raise ValueError('Unexpected item/mask/label/permutation or duplicated physical row')
        record_ids.add(rid);i=index[item]
        if observed[i,mask,p,label]:raise ValueError('Duplicated factorial cell')
        expected=render.render(items[item],mask,p,label)
        if any(row.get(k)!=expected[k] for k in ('messages','candidates','positive_letter')):
            raise ValueError('Original input-builder semantics differ at '+rid)
        if mask==0:
            for permutation in range(10):
                alternate=render.render(items[item],0,permutation,label)
                if any(alternate[k]!=expected[k] for k in ('messages','candidates','positive_letter')):
                    raise ValueError('Cannot broadcast position-dependent empty baseline')
                empty_checks+=1
        value=float(row[readout+'_y_AB'])*(1 if row['positive_letter']=='A' else -1)
        if not np.isfinite(value):raise ValueError('Nonfinite source readout; no row has been dropped')
        values[i,mask,p,label]=value;observed[i,mask,p,label]=True;count+=1
    if count!=622*len(items) or not observed[:,1:,:,:].all() or not observed[:,0,0,:].all():
        raise ValueError('Missing full32, ten-permutation, two-label physical panel')
    values[:,0,:,:]=values[:,0,0:1,:]
    for path in (inputs,results,rp,sp):sources[str(Path(path).resolve())]=base.sha(path)
    return {'item_ids':ids,'topics':[items[x]['topic'] for x in ids],'values':values,'run':run,
        'physical_rows':count,'empty_baseline':{'physical_per_item':2,'logical_per_item':20,
            'source_builder_all_ten_empty_messages_candidates_and_orientation_identical':True,
            'checked_empty_prompt_equivalences':empty_checks,'logical_replications_are_not_new_observations':True},
        'readout':readout}


def components(y):
    """Per-item numerator and denominator of the original ratio-of-sums C."""
    y=np.asarray(y,dtype=float)
    if y.ndim!=2 or y.shape[1]!=32 or not np.isfinite(y).all():raise ValueError('Finite full32 item panel required')
    alpha=y[:,0]
    pair=lambda a,b:y[:,(1<<a)|(1<<b)]-y[:,1<<a]-y[:,1<<b]+alpha
    numerator=np.mean([pair(a,b)+pair(3,4)-.5*(pair(a,3)+pair(b,4)+pair(a,4)+pair(b,3))
                       for a,b in itertools.combinations(range(3),2)],axis=0)
    denominator=np.mean(np.abs(y[:,[1,2,4]]-alpha[:,None]),axis=1)
    return numerator,denominator


def group_definitions(permutations):
    if len(permutations)!=10 or any(sorted(p)!=sorted(ROLES) for p in permutations):raise ValueError('Expected ten source permutations')
    groups=[{'id':'overall','kind':'overall','permutations':list(range(10)),'label_maps':[0,1]}]
    groups.extend({'id':f'label_{l}','kind':'label_map','permutations':list(range(10)),'label_maps':[l]} for l in (0,1))
    groups.extend({'id':f'permutation_{p:02d}','kind':'permutation','permutations':[p],'label_maps':[0,1],
        'roles_in_slots_1_to_5':list(order),'opponent_slots':[i+1 for i,r in enumerate(order) if r in ROLES[:3]]}
        for p,order in enumerate(permutations))
    slots=defaultdict(list)
    for p,order in enumerate(permutations):slots[tuple(i+1 for i,r in enumerate(order) if r in ROLES[:3])].append(p)
    groups.extend({'id':'opponent_slots_'+'_'.join(map(str,s)),'kind':'opponent_slot_set',
        'opponent_slots':list(s),'permutations':ps,'label_maps':[0,1]} for s,ps in sorted(slots.items()))
    return groups


def safe_ratio(num,den):
    return np.divide(num,den,out=np.full(np.broadcast_shapes(np.shape(num),np.shape(den)),np.nan),where=np.asarray(den)>0)


def interval(values):
    finite=np.asarray(values)[np.isfinite(values)]
    return {'percentile_95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
        'defined_draws':len(finite),'undefined_draws':len(values)-len(finite),
        'undefined_denominator_draws_retained_in_count':True}


def scalar(value):return float(value) if np.isfinite(value) else None


def paired_estimates(component_pairs,topics,draws,seed,reference=0):
    """One shared topic-weight draw for every group/readout, including the reference."""
    unique=sorted(set(topics));labels=np.asarray(topics)
    nums=np.asarray([p[0] for p in component_pairs]);dens=np.asarray([p[1] for p in component_pairs])
    if nums.shape!=dens.shape or nums.shape[1]!=len(topics):raise ValueError('Unequal paired item sets')
    tn=np.asarray([[n[labels==t].sum() for t in unique] for n in nums])
    td=np.asarray([[d[labels==t].sum() for t in unique] for d in dens])
    weights=np.random.default_rng(seed).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=draws)
    boot=safe_ratio(weights@tn.T,weights@td.T);points=safe_ratio(nums.sum(axis=1),dens.sum(axis=1));records=[]
    for gi,point in enumerate(points):
        delta=point-points[reference]
        records.append({'C':scalar(point),'C_interval':interval(boot[:,gi]),
            'C_minus_reference':scalar(delta),'paired_difference_interval':interval(boot[:,gi]-boot[:,reference]),
            'numerator_sum':float(nums[gi].sum()),'denominator_sum':float(dens[gi].sum()),
            'point_sign':'undefined' if not np.isfinite(point) else 'positive' if point>0 else 'negative' if point<0 else 'zero',
            'sign_differs_from_reference':bool(point*points[reference]<0) if np.isfinite(point+points[reference]) else None})
    return records,{'draws':draws,'seed':seed,'unit':'topic','topics':unique,'n_topics':len(unique),
        'paired_weights_shared_across_all_groups':True,'estimator':'ratio of summed item numerators and denominators, not mean item ratios',
        'interval_method':'percentile; undefined denominator draws counted explicitly',
        'intervals_are_descriptive_not_new_admission_or_hypothesis_gates':True}


def diagnose(panel,ids,definitions,draws,seed,scope):
    indices=[panel['item_ids'].index(i) for i in sorted(ids)];values=panel['values'][indices]
    topics=[panel['topics'][i] for i in indices];pairs=[]
    for group in definitions:
        y=np.take(np.take(values,group['permutations'],axis=2),group['label_maps'],axis=3).mean(axis=(2,3))
        pairs.append(components(y))
    estimates,bootstrap=paired_estimates(pairs,topics,draws,seed)
    return {'scope':scope,'readout':panel['readout'],'n_items':len(ids),'item_ids':sorted(ids),'bootstrap':bootstrap,
        'groups':[definition|estimate for definition,estimate in zip(definitions,estimates)],
        'actual_availability':{'observed_permutations':10,'observed_opponent_slot_sets':sum(g['kind']=='opponent_slot_set' for g in definitions),
            'all_possible_three_of_five_slot_sets':10,'unobserved_slot_sets_not_imputed':True},
        'negative_near_zero_and_sign_reversals_are_reported_without_exclusion':True,
        'empty_baseline':panel['empty_baseline']}


def paired_readouts(r0,r1,ids,draws,seed):
    common=sorted(set(ids)&set(r1['item_ids']))
    if not common:return {'status':'no_matched_items','n_items':0}
    pairs=[];topics=[]
    for panel in (r0,r1):
        ix=[panel['item_ids'].index(i) for i in common]
        labels=[panel['topics'][i] for i in ix]
        if topics and topics!=labels:raise ValueError('Matched readout topics differ')
        topics=labels;pairs.append(components(panel['values'][ix].mean(axis=(2,3))))
    records,bootstrap=paired_estimates(pairs,topics,draws,seed)
    return {'status':'matched_items_only','n_items':len(common),'item_ids':common,'R0':records[0],'R1':records[1],
        'paired_R1_minus_R0':records[1]['C_minus_reference'],
        'paired_difference_interval':records[1]['paired_difference_interval'],'bootstrap':bootstrap,
        'comparison_to_full180_or200_R0_is_not_a_readout_difference':True,
        'no_generated_probability_or_R2_effect_estimator_used':True}


def build(root,inputs,results,draws=10000,seed=20261031,r1_inputs=None,r1_results=None,keep_items=None):
    root=Path(root).resolve();sources={};render,render_path=renderer(root)
    itemspath=root/'data/e1_holdout200_items.json';allitems={x['item_id']:x for x in base.read_json(itemspath)}
    if len(allitems)!=200:raise ValueError('Fixed E1 parent must contain 200 unique items')
    expected_input(inputs,'e1_holdout200_prompts.jsonl',root,sources)
    r0=load_panel(inputs,results,allitems,render,'R0',root,sources)
    run=r0['run'];parent=run['engine']['parent_loader'];repo=parent['repo_id']
    statepath=root/'results'/repo.replace('/','--')/'e0/e0_driver_state.json';state=base.read_json(statepath)
    lockpath=local(state['selected_calibration'],root);lock=base.read_json(lockpath)
    if (state.get('status')!='complete' or state['qualification']['R0_operational']!='pass'
        or run['calibration_file_sha256']!=base.sha(lockpath) or run['engine']!=lock['engine']
        or run['option_ids']!=lock['option_ids'] or run['batch_size']!=lock['batch_size']):
        raise ValueError('E1 differs from its completed, frozen E0 readout identity')
    corepath=root/'data/core_data_manifest.json';core=base.read_json(corepath);old=set(core['reused20_ids'])
    freshpath=root/'data/e1_remaining180_items.json';fresh={x['item_id'] for x in base.read_json(freshpath)}
    if len(old)!=20 or len(fresh)!=180 or set(allitems)-old!=fresh:raise ValueError('Frozen fresh180 partition changed')
    selected=fresh if repo in OLD_MODELS else set(allitems)
    scope='primary_confirmatory_fresh180' if repo in OLD_MODELS else 'primary_full200'
    definitions=group_definitions(render.permutations())
    result={'schema':'voices_v3_E9_existing_label_slot_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'repo_id':repo,'revision':parent['revision'],'readout':'R0','primary':diagnose(r0,selected,definitions,draws,seed,scope),
        'execution_identity':base.execution_identity([results]),'scope':'Only existing-E1 label/permutation/opponent-slot diagnostics; not complete E9.',
        'new_model_inference':False,'templates_tested_here':False,'batch_sensitivity_tested_here':False,
        'selection_or_quality_thresholds_changed':False,'old20_excluded_from_primary':repo in OLD_MODELS}
    result['source_builder_contract']={'origin':'voices_nonadditivity_v2_20261030/code/render_v2.py original template',
        'inspected_source_sha256':SOURCE_RENDERER_SHA256,'source_file_available_and_hash_verified':render_path is not None,
        'embedded_exact_messages_options_and_orientation_checked_against_every_physical_source_row':True}
    secondary=None
    if keep_items:
        entries=base.read_json(keep_items);entries=entries['item_ids'] if isinstance(entries,dict) else entries
        listed=[x if isinstance(x,str) else x['item_id'] for x in entries];secondary=set(listed)
        if len(listed)!=20 or len(secondary)!=20 or not secondary<=set(allitems):raise ValueError('--keep-items requires 20 unique fixed E1 parent items')
        label='descriptive_prior20_included_not_confirmatory' if repo in OLD_MODELS and secondary&old else 'additional_fixed20_diagnostic'
        result['additional_fixed20']=diagnose(r0,secondary,definitions,draws,seed,label)
        result['additional_fixed20']['does_not_replace_or_modify_primary_sample']=True
        sources[str(Path(keep_items).resolve())]=base.sha(keep_items)
    if r1_results:
        semanticpath=root/'data/e1_semantic64_items.json';semantic=base.read_json(semanticpath)
        items={x['item_id']:x for x in semantic}
        if len(items)!=64 or any(allitems.get(k)!=v for k,v in items.items()):raise ValueError('Semantic64 is not the fixed parent-item subset')
        expected_input(r1_inputs,'e1_semantic64_prompts.jsonl',root,sources)
        r1=load_panel(r1_inputs,r1_results,items,render,'R1',root,sources)
        if any(run[k]!=r1['run'][k] for k in ('engine','batch_size','calibration_file_sha256','option_ids')):
            raise ValueError('Matched R0/R1 have different model/runtime/readout identities')
        result['matched_R0_R1']=paired_readouts(r0,r1,selected,draws,seed)
        expected=44 if repo in OLD_MODELS else 64
        if result['matched_R0_R1']['n_items']!=expected:raise ValueError('Expected fixed fresh44/full64 matched readout panel')
        if secondary is not None:result['additional_fixed20_matched_R0_R1']=paired_readouts(r0,r1,secondary,draws,seed)
        sources[str(semanticpath.resolve())]=base.sha(semanticpath)
    else:result['matched_R0_R1']={'status':'not_requested'}
    for path in (Path(__file__),Path(base.__file__),itemspath,freshpath,corepath,statepath,lockpath):
        sources[str(path.resolve())]=base.sha(path)
    if render_path:sources[str(render_path.resolve())]=base.sha(render_path)
    result['file_sha256']=sources
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--inputs',type=Path);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--r1-inputs',type=Path);p.add_argument('--r1-results',type=Path)
    p.add_argument('--keep-items',type=Path,help='Additional fixed20 descriptive view; never changes the primary180/200 sample')
    p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.draws<1:raise ValueError('At least one bootstrap draw required')
    if args.output.exists():raise FileExistsError('Preserve existing diagnostic report; choose a new output')
    root=args.root.resolve();result=build(root,args.inputs or root/'data/e1_holdout200_prompts.jsonl',args.results,
        args.draws,args.seed,args.r1_inputs or root/'data/e1_semantic64_prompts.jsonl',args.r1_results,args.keep_items)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'output':str(args.output),'repo_id':result['repo_id'],'n_primary':result['primary']['n_items'],
        'matched_readout_n':result['matched_R0_R1'].get('n_items'),'scope':'existing label/slot component only'}))


if __name__=='__main__':main()
