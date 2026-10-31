"""Concise Chinese E4 report from hash-verified aggregates and supplements.

Missing inputs remain pending. A six-model single-round result is not mislabeled
as completed dynamics, free conversation, human validation, or all later stages.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import itertools
import json
import math
from pathlib import Path

from analyze_v3 import read_json,sha
from summarize_e0_panel_v3 import local_path

ROOT=Path(__file__).resolve().parents[1]
PARTS=('E4a','E4b','E4c')
METHODS=('additive','present_mean','side_mean','side_max')


class Evidence:
    def __init__(self,root,remote_root):self.root=Path(root);self.remote_root=remote_root;self.cache={};self.verified={}
    def resolve(self,value):return local_path(value,self.root,self.remote_root)
    def digest(self,path):
        path=Path(path);key=(str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns)
        if key not in self.cache:self.cache[key]=sha(path)
        return self.cache[key]
    def check(self,path,expected):
        path=Path(path)
        if self.digest(path)!=expected:raise ValueError('SHA256 mismatch: '+str(path))
        self.verified[str(path.resolve())]=expected
    def package(self,path,kind):
        path=Path(path);data=read_json(path);missing=[];mismatch=[]
        if not data.get('file_sha256'):raise ValueError('No source hash bindings')
        for source,digest in data['file_sha256'].items():
            mapped=self.resolve(source)
            try:self.check(mapped,digest)
            except FileNotFoundError:missing.append(str(mapped))
            except (ValueError,OSError) as error:mismatch.append(str(error))
        artifact_key='outcomes_and_bootstrap_artifact' if kind=='supplement' else 'bootstrap_artifact'
        artifact=data.get(artifact_key)
        if not artifact:mismatch.append('Missing '+artifact_key)
        else:
            # Sidecar filenames are deliberately relative to the supplied result.
            target=path.parent/artifact['filename']
            if target.resolve().parent!=path.parent.resolve():mismatch.append('Sidecar path is not local to its report')
            else:
                try:self.check(target,artifact['sha256'])
                except FileNotFoundError:missing.append(str(target))
                except (ValueError,OSError) as error:mismatch.append(str(error))
        self.verified[str(path.resolve())]=self.digest(path)
        integrity={'status':'invalid' if mismatch else 'partial_local_sync' if missing else 'verified',
                   'missing_sources':missing,'mismatches':mismatch,'path':str(path),'sha256':self.digest(path)}
        return data,integrity


def finite(value):return isinstance(value,(float,int)) and math.isfinite(value)


def main_metric(value,propagated=True):
    key='E1_kernel_and_E4_topic_propagated' if propagated else 'conditional_on_E1_point_kernel'
    interval=value.get(key,{})
    return {'estimate':value.get('estimate'),'interval95':interval.get('interval95'),
            'undefined_draws':interval.get('undefined_draws'),'interval_scope':key}


def verdict(metric):
    if not metric or not finite(metric.get('estimate')):return 'undefined'
    ci=metric.get('interval95')
    if ci is None or len(ci)!=2 or not all(finite(v) for v in ci):return 'interval_undefined'
    if ci[0]>0:return 'improvement_supported_pointwise'
    if ci[1]<0:return 'degradation_supported_pointwise'
    return 'uncertain_including_zero'


def same_number(left,right):
    if left is None or right is None:return left is right
    return math.isclose(left,right,rel_tol=1e-8,abs_tol=1e-10)


def selection_receipt(path,evidence):
    state={'status':'pending','selected_repos':[],'models':{}}
    try:
        frozen=read_json(path)
        if (frozen.get('schema')!='voices_v3_frozen_E4_model_selection_v1' or frozen.get('status')!='frozen' or
            not frozen.get('all_requested_E1_complete') or not frozen.get('performed')):raise ValueError('E4 selection is not a frozen completed E1 decision')
        ids=frozen['selected_repos']
        if len(ids)!=6 or len(set(ids))!=6:raise ValueError('The frozen E4 panel must contain six unique models')
        panel_path=evidence.resolve(frozen['panel_path']);evidence.check(panel_path,frozen['panel_sha256']);panel=read_json(panel_path)
        if (not panel.get('all_requested_E1_complete') or panel['E4_selection'].get('selected_repos')!=ids or
            panel['E4_selection'].get('status')!='frozen'):raise ValueError('Frozen selection and linked E1 panel disagree')
        by_repo={r['repo_id']:r for r in panel['models']};models={}
        flags=frozen.get('selected_conditional_option_mass_flags',{})
        for repo in ids:
            row=by_repo[repo]
            if not row.get('E4_eligible') or not row.get('E1_complete'):raise ValueError('Selected model lacks a closed eligible E1 result')
            conditional=flags.get(repo,row.get('conditional_option_mass_admission'))
            if repo in flags and 'conditional_option_mass_admission' in row and flags[repo]!=row['conditional_option_mass_admission']:
                raise ValueError('Conditional option-mass flags disagree')
            models[repo]={'conditional_option_mass_admission':conditional,
                         'R0_readout_disagreement_flag':row.get('R0_readout_disagreement_flag'),
                         'E0_tier':row.get('E0_tier'),'identity':row['primary']['execution_identity'],
                         'E1_primary_analysis_sha256':row['primary']['sha256']}
        evidence.verified[str(Path(path).resolve())]=evidence.digest(path)
        state.update(status='verified',selected_repos=ids,models=models,sha256=evidence.digest(path),
                     linked_panel_sha256=frozen['panel_sha256'],linked_panel_sources_rehashed=False)
    except FileNotFoundError as error:state.update(status='pending_or_partial_local_sync',error=str(error))
    except (ValueError,KeyError,TypeError,OSError) as error:state.update(status='invalid',error=str(error))
    return state


def validate_aggregate(data,expected_topics):
    required={'model_identity','single_round','bootstrap','topic_ids','n_topics','file_sha256'}
    if not required<=data.keys():raise ValueError('Not an aggregate_e4_v3 output')
    ids=data['topic_ids']
    if len(ids)!=data['n_topics'] or len(set(ids))!=len(ids) or not set(ids)<=set(expected_topics):raise ValueError('Unexpected/duplicate aggregate topics')
    if data['bootstrap'].get('unit')!='topic':raise ValueError('Aggregate intervals are not topic-cluster intervals')
    if not data.get('single_round_E1_kernel_uncertainty_propagated'):raise ValueError('Primary single-round aggregate lacks the specified E1 propagation')
    for part in PARTS:
        metrics=data['single_round'][part]['metrics']
        for key in ('observed_contrast_RMS','P0_R2_delta','PK_R2_delta','P0_unexplained_fraction_U',
                    'PK_minus_P0_R2','MSE_improvement_P0_minus_PK','fraction_P0_residual_squared_explained_by_PK'):
            if key not in metrics:raise ValueError('Missing primary E4 metric: '+part+'/'+key)
        if set(data['single_round'][part]['topic_sufficient_statistics'])!=set(ids):raise ValueError('Per-topic aggregate support differs')
    return (set(ids)==set(expected_topics) and len(ids)==6 and data.get('planned_six_topics_complete') is True and data['bootstrap'].get('draws')==10000)


def verify_topic_bindings(data,evidence):
    found={}
    for source,digest in data['file_sha256'].items():
        path=evidence.resolve(source)
        if path.suffix.lower()!='.json':continue
        # Per-topic point filenames are fixed by run_e4_analysis.py.
        if not path.name.endswith('_E4_analysis.json'):continue
        topic=read_json(path);tid=topic.get('topic_id')
        if tid not in data['topic_ids']:continue
        if tid in found:raise ValueError('Duplicate per-topic primary source')
        if (topic['execution_identity']['critical_identity']!=data['model_identity'] or topic['E4a']['n_conditions']!=70 or
            topic['E4b']['n_systems']!=80 or topic['E4b']['n_mappings']!=20 or topic['E4c']['n_conditions']!=192 or
            {(r['n_target'],r['tau']) for r in topic['E4b']['configurations']}!={(m,t) for m in (3,5,7) for t in (-1,1)}):
            raise ValueError('Primary topic does not contain the planned E4a/b/c panel')
        cache=evidence.resolve(source).parent/topic['cache_artifact']['filename'];evidence.check(cache,topic['cache_artifact']['sha256'])
        found[tid]=digest
    if set(found)!=set(data['topic_ids']):raise ValueError('Aggregate lacks its bound per-topic primary analyses')
    return found


def index_inputs(paths,kind,evidence,expected_topics,manifest_sha256):
    indexed={};errors=[]
    for path in paths or []:
        try:
            data,integrity=evidence.package(path,kind)
            repo=data['repo_id'] if kind=='supplement' else data['model_identity']['repo']
            if repo in indexed:raise ValueError('Duplicate '+kind+' input for '+repo)
            complete=False;topic_bindings={}
            if integrity['status']=='verified':
                if manifest_sha256 not in data['file_sha256'].values():raise ValueError('Result is not bound to the current E4 topic/agent manifest')
                if kind=='supplement':
                    if data.get('schema')!='voices_v3_E4_single_round_pairwise_rule_supplement_v1':raise ValueError('Unexpected supplement schema')
                    if len(set(data['topic_ids']))!=data['n_topics'] or not set(data['topic_ids'])<=set(expected_topics):raise ValueError('Supplement topic support differs')
                    if not data.get('primary_P0_PK_U_definitions_unchanged') or data.get('E4_fit_parameters')!=0:raise ValueError('Supplement definitions or fitting policy differs')
                    if data['bootstrap'].get('unit')!='topic' or not data['bootstrap'].get('same_topic_draws_for_all_methods'):raise ValueError('Supplement comparisons are not paired topic-cluster estimates')
                    complete=(set(data['topic_ids'])==set(expected_topics) and data['n_topics']==6 and data.get('planned_six_topics_complete') is True and data['bootstrap']['draws']==10000)
                else:
                    complete=validate_aggregate(data,expected_topics);topic_bindings=verify_topic_bindings(data,evidence)
            indexed[repo]={'data':data,'integrity':integrity,'planned_complete':complete,'topic_bindings':topic_bindings}
        except (FileNotFoundError,OSError,ValueError,KeyError,TypeError) as error:
            errors.append({'kind':kind,'path':str(path),'error':str(error)})
    return indexed,errors


def supplement_summary(supplement,primary,selected):
    if supplement['repo_id']!=primary['model_identity']['repo'] or set(supplement['topic_ids'])!=set(primary['topic_ids']):raise ValueError('Supplement and primary model/topic panel differ')
    if supplement['E1_analysis_sha256']!=selected['E1_primary_analysis_sha256']:raise ValueError('Supplement uses another E1 selection source')
    if supplement['E1_analysis_sha256'] not in primary['file_sha256'].values():raise ValueError('Primary and supplement E1 sources differ')
    def point_sources(result):
        return {digest for path,digest in result['file_sha256'].items() if str(path).endswith('_E4_analysis.json')}
    if point_sources(supplement)!=point_sources(primary):raise ValueError('Supplement and primary use different measured topic sources')
    out={}
    for part in PARTS:
        pm=primary['single_round'][part]['metrics'];sm=supplement['single_round'][part]
        if not same_number(pm['P0_R2_delta']['estimate'],sm['methods']['additive']['R2_delta']['estimate']) or not same_number(pm['PK_R2_delta']['estimate'],sm['methods']['PK']['R2_delta']['estimate']):
            raise ValueError('Supplement changed the primary P0/PK result')
        selected_rule=supplement['E1_best_rule']['rule'];comparison=sm['PK_improvement_over_E1_selected_rule']
        improvement=comparison.get('R2_delta_improvement') if comparison else None
        out[part]={'E1_selected_rule':selected_rule,
                   'selected_rule_R2_delta':sm['methods'][selected_rule]['R2_delta'] if selected_rule else None,
                   'PK_minus_selected_rule_R2_delta':improvement,
                   'verdict':verdict(improvement),
                   'PK_minus_each_rule_R2_delta':{name:sm['PK_improvement_over_each_rule'][name]['R2_delta_improvement'] for name in METHODS},
                   'interval_scope':'conditional_on_fixed_E1_kernels_and_E1_rule_selection'}
    return out


def dynamics_summary(aggregate):
    dynamic=aggregate.get('dynamic',{'status':'not_run'});output={'status':dynamic['status'],'planned_complete':False}
    if dynamic['status']!='complete_conditional_topic_aggregation':return output
    configurations={f'm{m}_tau{t}' for m in (3,5,7) for t in (-1,1)};rounds={str(r) for r in range(1,7)}
    b=dynamic.get('by_configuration_and_round',{});a=dynamic.get('E4a_by_round',{})
    b_complete=bool(dynamic.get('E4b_complete_topic_panel') and set(b)==configurations and all(rounds<=set(v) for v in b.values()))
    a_complete=bool(dynamic.get('E4a_complete_topic_panel') and rounds<=set(a))
    topic_set=set(aggregate['topic_ids']);budget_b=b_complete;budget_a=a_complete
    for key in configurations:
        records=dynamic.get('MCSE',{}).get(key,[])
        budget_b=budget_b and len(records)==6 and {r['topic_id'] for r in records}==topic_set
        budget_b=budget_b and all(r.get('total_draws_per_design',0) is not None and r.get('total_draws_per_design',0)>=100000 and r.get('draws',0)>=5000 for r in records)
    arecords=dynamic.get('E4a_MCSE',[])
    budget_a=budget_a and len(arecords)==6 and {r['topic_id'] for r in arecords}==topic_set
    budget_a=budget_a and all(r.get('draws_per_design_per_condition',0)>=100000 for r in arecords)
    output.update(E4a_complete=a_complete,E4b_complete=b_complete,E4a_budget_complete=bool(budget_a),E4b_budget_complete=bool(budget_b),
                  planned_complete=bool(budget_a and budget_b and aggregate['n_topics']==6),
                  E1_kernel_parameter_uncertainty_propagated=dynamic.get('E1_kernel_parameter_uncertainty_propagated',False),
                  interval_scope=dynamic.get('interval_scope'),MC_error_scope=dynamic.get('MC_error_scope'),
                  E4a_MCSE=arecords,E4b_MCSE=dynamic.get('MCSE',{}),final_round={})
    if a_complete:output['final_round']['E4a_PK_minus_P0_R2']=main_metric(a['6']['PK_minus_P0_R2'],False)
    if b_complete:output['final_round']['E4b_PK_minus_P0_R2_by_initial_condition']={key:main_metric(b[key]['6']['PK_minus_P0_R2'],False) for key in sorted(configurations)}
    return output


def rank_correlation(x,y):
    """Small independent check of six-model Spearman points, including ties."""
    if len(x)!=len(y) or not all(finite(v) for v in [*x,*y]):return None
    def ranks(values):
        return [sum(other<value for other in values)+(sum(other==value for other in values)+1)/2 for value in values]
    a,b=ranks(x),ranks(y);center=(len(x)+1)/2
    a=[v-center for v in a];b=[v-center for v in b]
    denominator=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
    return sum(u*v for u,v in zip(a,b))/denominator if denominator else None


def validate_cross_statistic(statistic,x,y,repos):
    rho=rank_correlation(x,y)
    if not same_number(statistic['rho'],rho):raise ValueError('H3 Spearman point differs from its six source values')
    if statistic.get('n_permutations')!=720:raise ValueError('H3 must enumerate all 720 model-label permutations')
    probability=(sum(rank_correlation(x,list(order))>=rho-1e-12 for order in itertools.permutations(y))/720
                 if rho is not None else None)
    if not same_number(statistic['permutation_p_one_sided_positive'],probability):
        raise ValueError('H3 exact one-sided permutation probability differs')
    count=statistic['undefined_draws'];bounds=statistic['interval95']
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=10000:raise ValueError('Invalid H3 undefined-draw count')
    if count and bounds is not None:raise ValueError('H3 must retain undefined draws and withhold the interval')
    if bounds is not None and (len(bounds)!=2 or not all(finite(v) and -1-1e-12<=v<=1+1e-12 for v in bounds) or bounds[0]>bounds[1]):
        raise ValueError('Invalid H3 bootstrap interval')
    omitted=statistic['leave_one_model_out']
    if len(omitted)!=6 or {row['omitted_repo'] for row in omitted}!=set(repos):raise ValueError('H3 leave-one-model panel differs')
    for row in omitted:
        index=repos.index(row['omitted_repo']);keep=[i for i in range(6) if i!=index]
        if not same_number(row['rho'],rank_correlation([x[i] for i in keep],[y[i] for i in keep])):
            raise ValueError('H3 leave-one-model point differs')


def cross_model_summary(path,selection_path,selection,mains,dynamic_inputs,expected_topics,evidence):
    output={'status':'pending_not_supplied','planned_complete':False}
    if path is None:return output
    output['path']=str(path)
    try:
        data,integrity=evidence.package(path,'cross_model');output.update(integrity=integrity,sha256=integrity['sha256'],source_status=data.get('status'))
        if integrity['status']!='verified':
            output['status']=integrity['status'];return output
        if selection['status']!='verified':raise ValueError('H3 needs the verified frozen model selection')
        repos=selection['selected_repos']
        if (data.get('schema')!='voices_v3_E4_cross_model_H3_v1' or data.get('status')!='complete_verified_for_selected_panel'
                or data.get('selected_repos')!=repos or data.get('n_models')!=6):
            raise ValueError('H3 schema/completion/frozen model panel differs')
        if len(data['topic_ids'])!=6 or len(set(data['topic_ids']))!=6 or set(data['topic_ids'])!=set(expected_topics):
            raise ValueError('H3 must use the same complete six E4 topics')
        bindings={str(evidence.resolve(source).resolve()):digest for source,digest in data['file_sha256'].items()}
        selection_digest=evidence.digest(selection_path)
        if data['selection_sha256']!=selection_digest or bindings.get(str(Path(selection_path).resolve()))!=selection_digest:
            raise ValueError('H3 selection SHA/source binding differs')
        bootstrap=data['bootstrap']
        if (bootstrap.get('draws')!=10000 or bootstrap.get('unit')!='topic'
                or any(bootstrap.get(key) is not True for key in ('E1_topic_weights_shared_across_models',
                    'E4_topic_weights_shared_across_models','E1_and_E4_topic_resampling_independent','undefined_draws_not_dropped'))
                or bootstrap.get('existing_model_bootstrap_rows_concatenated') is not False
                or len(bootstrap.get('E4_topic_ids',[]))!=6 or set(bootstrap['E4_topic_ids'])!=set(expected_topics)):
            raise ValueError('H3 lacks the specified joint cross-model topic bootstrap')
        records=data['models']
        if len(records)!=6 or {row['repo_id'] for row in records}!=set(repos):raise ValueError('H3 model rows differ from the frozen six')
        ordered={row['repo_id']:row for row in records};E1_topics=set()
        for repo in repos:
            row=ordered[repo];main=mains.get(repo) or dynamic_inputs.get(repo);selected=selection['models'][repo]
            if not main or main['integrity']['status']!='verified' or not main['planned_complete']:
                raise ValueError('H3 is missing a complete bound primary aggregate for '+repo)
            aggregate=main['data'];aggregate_path=Path(main['integrity']['path']).resolve()
            if (aggregate['model_identity']!=selected['identity'] or row['aggregate_sha256']!=evidence.digest(aggregate_path)
                    or bindings.get(str(aggregate_path))!=row['aggregate_sha256']
                    or row['E1_analysis_sha256']!=selected['E1_primary_analysis_sha256']):
                raise ValueError('H3 aggregate/E1/model identity differs for '+repo)
            candidates={str(evidence.resolve(source).resolve()) for source,digest in aggregate['file_sha256'].items()
                        if digest==row['E1_analysis_sha256']}
            if len(candidates)!=1:raise ValueError('H3 does not identify exactly one original E1 analysis')
            E1_path=Path(next(iter(candidates)))
            if bindings.get(str(E1_path))!=row['E1_analysis_sha256']:raise ValueError('H3 omits the original E1 source binding')
            E1=read_json(E1_path);evidence.check(E1_path,row['E1_analysis_sha256'])
            if (E1['readout']!='R0' or E1['execution_identity']['critical_identity']!=selected['identity']
                    or row['n_E1_items']!=E1['n_items'] or not same_number(row['C'],E1['point']['C'])):
                raise ValueError('H3 C/cohort/execution identity differs from original E1')
            E1_topics.update(E1['topics']);metrics=aggregate['single_round']['E4b']['metrics']
            if (not same_number(row['U'],metrics['P0_unexplained_fraction_U']['estimate'])
                    or not same_number(row['P0_residual_RMS'],metrics['P0_residual_RMS']['estimate'])):
                raise ValueError('H3 U/RMS differs from E4b single-round aggregate')
        if len(bootstrap.get('E1_topics',[]))!=len(E1_topics) or set(bootstrap['E1_topics'])!=E1_topics:
            raise ValueError('H3 resampling topic support differs from its E1 analyses')
        primary=data['primary'];sensitivity=data['sensitivity']['P0_residual_RMS']
        if primary['estimand']!='Spearman(abs(E1_C), E4b_single_round_P0_unexplained_fraction_U)':
            raise ValueError('H3 primary estimand is not the specified E4b U association')
        if sensitivity['estimand']!='Spearman(abs(E1_C), E4b_single_round_P0_residual_RMS)':
            raise ValueError('H3 RMS sensitivity estimand differs')
        x=[abs(ordered[repo]['C']) if ordered[repo]['C'] is not None else None for repo in repos]
        validate_cross_statistic(primary,x,[ordered[repo]['U'] for repo in repos],repos)
        validate_cross_statistic(sensitivity,x,[ordered[repo]['P0_residual_RMS'] for repo in repos],repos)
        output.update(status='complete_verified',planned_complete=True,selected_repos=repos,topic_ids=data['topic_ids'],
            selection_sha256=selection_digest,n_models=6,models=records,primary=primary,sensitivity=data['sensitivity'],
            bootstrap=bootstrap,bootstrap_artifact=data['bootstrap_artifact'])
    except FileNotFoundError as error:output.update(status='pending_or_partial_local_sync',error=str(error))
    except (OSError,ValueError,KeyError,TypeError) as error:output.update(status='invalid',error=str(error))
    return output


def build(selection_path,aggregates=None,supplements=None,dynamics_aggregates=None,root=ROOT,remote_root='/workspace/voices_nonadditivity_v3',cross_model=None):
    root=Path(root).resolve();evidence=Evidence(root,remote_root);selection=selection_receipt(selection_path,evidence)
    manifest_path=root/'data/e4_cache_manifest.json';manifest=read_json(manifest_path);expected=[t['topic_id'] for t in manifest['topics']]
    evidence.verified[str(manifest_path.resolve())]=evidence.digest(manifest_path)
    manifest_sha=evidence.digest(manifest_path)
    mains,errors=index_inputs(aggregates,'aggregate',evidence,expected,manifest_sha)
    extra,problems=index_inputs(supplements,'supplement',evidence,expected,manifest_sha);errors+=problems
    dynamic_inputs,problems=index_inputs(dynamics_aggregates,'dynamic_aggregate',evidence,expected,manifest_sha);errors+=problems
    selected=selection['selected_repos'];rows=[]
    for repo in selected:
        qual=selection['models'][repo];row={'repo_id':repo,**qual,'single_round_status':'pending','supplement_status':'pending','dynamics':{'status':'not_supplied','planned_complete':False},'single_round':{},'stronger_rule_comparison':{}}
        main=mains.get(repo) or dynamic_inputs.get(repo);supplement=extra.get(repo);dynamic=dynamic_inputs.get(repo) or main
        if main:
            row['aggregate_integrity']=main['integrity']
            if main['integrity']['status']=='verified':
                data=main['data']
                if data['model_identity']!=qual['identity']:
                    row['single_round_status']='invalid_selection_model_binding'
                elif qual['E1_primary_analysis_sha256'] not in data['file_sha256'].values():
                    row['single_round_status']='invalid_E1_source_binding'
                else:
                    row['single_round_status']='complete_verified' if main['planned_complete'] else 'partial_topics_or_bootstrap'
                    row['n_topics']=data['n_topics'];row['topic_ids']=data['topic_ids']
                    for part in PARTS:
                        metrics=data['single_round'][part]['metrics']
                        values={key:main_metric(metrics[key]) for key in ('observed_contrast_RMS','P0_R2_delta','PK_R2_delta','P0_unexplained_fraction_U','PK_minus_P0_R2','MSE_improvement_P0_minus_PK','fraction_P0_residual_squared_explained_by_PK')}
                        values['PK_improvement_verdict']=verdict(values['PK_minus_P0_R2']);row['single_round'][part]=values
            else:row['single_round_status']=main['integrity']['status']
        if supplement:
            row['supplement_integrity']=supplement['integrity']
            if supplement['integrity']['status']!='verified':row['supplement_status']=supplement['integrity']['status']
            elif not row['single_round']:row['supplement_status']='waiting_for_verified_primary'
            else:
                try:
                    row['stronger_rule_comparison']=supplement_summary(supplement['data'],main['data'],qual)
                    row['supplement_status']='complete_verified' if supplement['planned_complete'] else 'partial_topics_or_bootstrap'
                except (ValueError,KeyError,TypeError) as error:row.update(supplement_status='invalid_primary_binding',supplement_error=str(error))
        if dynamic and dynamic['integrity']['status']=='verified' and row['single_round']:
            try:
                if dynamic['data']['model_identity']!=main['data']['model_identity'] or dynamic['topic_bindings']!=main['topic_bindings']:
                    raise ValueError('Dynamic aggregate uses another primary model/topic source')
                row['dynamics']=dynamics_summary(dynamic['data']);row['dynamics_integrity']=dynamic['integrity']
            except (ValueError,KeyError,TypeError) as error:row['dynamics']={'status':'invalid_binding_or_schema','planned_complete':False,'error':str(error)}
        elif dynamic:row['dynamics']={'status':dynamic['integrity']['status'],'planned_complete':False}
        rows.append(row)
    unexpected=sorted((set(mains)|set(extra)|set(dynamic_inputs))-set(selected))
    counts={'planned_models':6,'selection_verified':selection['status']=='verified','selected_models':len(selected),
            'single_round_complete_models':sum(r['single_round_status']=='complete_verified' for r in rows),
            'supplement_complete_models':sum(r['supplement_status']=='complete_verified' for r in rows),
            'dynamics_complete_models':sum(r['dynamics'].get('planned_complete',False) for r in rows),
            'conditional_option_mass_models':sum(r.get('conditional_option_mass_admission') is True for r in rows),
            'unknown_option_mass_flag_models':sum(r.get('conditional_option_mass_admission') is None for r in rows)}
    single_complete=(counts['selection_verified'] and all(counts[key]==6 for key in ('selected_models','single_round_complete_models','supplement_complete_models')) and not counts['unknown_option_mass_flag_models'] and not errors and not unexpected)
    H3=cross_model_summary(cross_model,selection_path,selection,mains,dynamic_inputs,expected,evidence)
    cross_complete=H3['status']=='complete_verified'
    if H3.get('error'):errors.append({'kind':'cross_model_H3','path':str(cross_model),'error':H3['error']})
    complete=single_complete and counts['dynamics_complete_models']==6 and cross_complete
    def distribution(subset,comparison):
        values=[]
        for r in subset:
            if comparison=='primary' and r['single_round_status']=='complete_verified':values.append(r['single_round']['E4b']['PK_improvement_verdict'])
            elif comparison=='stronger_rule' and r['supplement_status']=='complete_verified':values.append(r['stronger_rule_comparison']['E4b']['verdict'])
        return dict(Counter(values))
    normal=[r for r in rows if r.get('conditional_option_mass_admission') is False]
    return {'schema':'voices_v3_E4_concise_report_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete_E4_main_and_supplement' if complete else 'partial_or_pending',
        'single_round_stage_complete':single_complete,
        'cross_model_stage_complete':cross_complete,'cross_model_H3':H3,
        'counts':counts,'selection':selection,'models':rows,'input_errors':errors,'inputs_outside_frozen_panel':unexpected,
        'E4b_pointwise_verdict_counts':{'all_selected_models':{'PK_vs_P0':distribution(rows,'primary'),'PK_vs_E1_selected_rule':distribution(rows,'stronger_rule')},
            'normal_option_mass_subset':{'n_models':len(normal),'PK_vs_P0':distribution(normal,'primary'),'PK_vs_E1_selected_rule':distribution(normal,'stronger_rule')}},
        'evidence_levels':{'inputs':'Real ArgKP argument texts; agents, prescribed stances and meeting designs are experimentally constructed.',
            'single_round':'Measured native LLM output-layer logits under the fixed prompts, then deterministic cache-based outcome aggregation.',
            'dynamics':'Monte Carlo state updates driven by the measured local-response caches; not observed free conversation or human discussion.'},
        'scope_status':{'E3':'skipped_by_user','E5':'deferred_with_E3','E4d':'optional_not_assessed','E6':'not_run','E7':'not_run','E8':'not_run','E9':'full_stage_not_run'},
        'interpretation_limits':['Only six independent E4 topics; mappings, states and system pairs are not independent topic replications.',
            'The six models were purposively chosen using E1 absolute C and architecture coverage, with usable mandatory models; they are not a representative sample of LLMs.',
            'STS contrasts are within common mapping/initial configurations, not differences after averaging all mappings.',
            'E1 relationship kernels average ten cyclic/reversed five-slot positions, whereas E4 triple meetings use fixed slots 2/4 with both orders. This adds a position-transfer assumption: position sensitivity can contribute prediction error, while within-E4 design comparisons share the same slot policy.',
            'Primary single-round intervals propagate E1-kernel and E4-topic uncertainty; supplementary-rule and dynamic intervals condition on point E1 estimates.',
            'Displayed95% intervals are pointwise; this report does not claim simultaneous coverage or a new multiplicity-adjusted hypothesis test.',
            'Beating additive P0 alone does not establish insufficiency of all group-aware algorithms using pairwise response data.',
            'Conditional-q models are marked; normally admitted subset verdict counts are descriptive, not an unselected model-population estimate.'],
        'verification_scope':'Direct aggregate/supplement/H3 source hashes, linked topic/cache artifacts, sidecars, frozen selection-panel receipt and model/E1 bindings; H3 point correlations and exact permutations independently checked, without new raw inference or bootstrap replay.',
        'verified_source_sha256':evidence.verified,'reporter_sha256':sha(__file__)}


def cell(metric,show_interval=True):
    if not metric or not finite(metric.get('estimate')):return '未定义'
    value=f"{metric['estimate']:.4g}"
    if show_interval:
        ci=metric.get('interval95');value+=(' ['+', '.join(f'{x:.4g}' for x in ci)+']') if ci is not None and len(ci)==2 and all(finite(x) for x in ci) else ' [区间未定义]'
    return value


def markdown(report):
    c=report['counts'];lines=['# E4 结果简报','',
        f"单轮主分析已完整校验 {c['single_round_complete_models']}/6 个模型，较强规则对照 {c['supplement_complete_models']}/6，多轮模拟 {c['dynamics_complete_models']}/6。"
        +(' E4 主实验及补充分析完成。' if report['status']=='complete_E4_main_and_supplement' else ' 其余仍为未完成、未提供或待校验。'),'',
        '完成度仅指预定实验与文件校验，不等于出现正面结果；PK 无明确改善或退化均照实报告。模型按 E1 的 |C|、架构覆盖及可用的预指定模型选择，不能视作所有 LLM 的代表样本。','',
        '输入使用真实 ArgKP 论点文本。单轮结果来自固定提示下的原生 LLM 输出层读数及其汇总；多轮结果是这些实测缓存驱动的蒙特卡洛更新，不是自由生成对话或人类讨论实验。','',
        '| 模型 / 实验 | 差异 RMS | P0 / PK R²Δ | PK−P0 ΔR²（95%） | E1 优选规则 | PK−该规则 ΔR²（95%） |',
        '|---|---:|---:|---:|---|---:|']
    for row in report['models']:
        name=row['repo_id'].split('/')[-1]
        if row.get('conditional_option_mass_admission') is True:name+=' †'
        elif row.get('conditional_option_mass_admission') is None:name+=' [资格标记待核对]'
        if not row['single_round']:
            lines.append('| '+name+' | '+row['single_round_status']+' | — | — | — | — |');continue
        for part in PARTS:
            main=row['single_round'][part];supp=row['stronger_rule_comparison'].get(part,{})
            label=name+' / '+part
            if row['single_round_status']!='complete_verified':label+=f" [{row.get('n_topics',0)}/6议题]"
            lines.append('| '+' | '.join([label,cell(main['observed_contrast_RMS'],False),
                cell(main['P0_R2_delta'],False)+' / '+cell(main['PK_R2_delta'],False),cell(main['PK_minus_P0_R2']),
                str(supp.get('E1_selected_rule') or row['supplement_status']),cell(supp.get('PK_minus_selected_rule_R2_delta'))])+' |')
    distribution=report['E4b_pointwise_verdict_counts']['all_selected_models']
    for title,key in [('相对加性 P0','PK_vs_P0'),('相对 E1 预选规则','PK_vs_E1_selected_rule')]:
        d=distribution[key]
        lines+=['',f"E4b {title}：点状区间支持改善 {d.get('improvement_supported_pointwise',0)} 个模型，支持退化 {d.get('degradation_supported_pointwise',0)} 个，跨零 {d.get('uncertain_including_zero',0)} 个；未完成或区间未定义的结果不计入改善。"]
    normal=report['E4b_pointwise_verdict_counts']['normal_option_mass_subset']
    lines+=['',f"正常 q 门子集为 {normal['n_models']} 个模型；其中 PK 相对 P0、相对 E1 预选规则的区间支持改善数分别为 "
            f"{normal['PK_vs_P0'].get('improvement_supported_pointwise',0)}、{normal['PK_vs_E1_selected_rule'].get('improvement_supported_pointwise',0)}。此子集统计同样只是描述。"]
    H3=report['cross_model_H3']
    if H3['status']=='complete_verified':
        primary=H3['primary'];sensitivity=H3['sensitivity']['P0_residual_RMS']
        values=[row['rho'] for row in primary['leave_one_model_out'] if finite(row['rho'])]
        span=('['+f'{min(values):.4g}, {max(values):.4g}'+']') if values else '未定义'
        probability=primary['permutation_p_one_sided_positive']
        pvalue=f'{probability:.4g}' if finite(probability) else '未定义'
        lines+=['',f"H3 跨模型检验已完成：E4b 的 Spearman(|C|, U) = "
            +cell({'estimate':primary['rho'],'interval95':primary['interval95']})
            +f"；720 个完整排列的正向单侧 p = {pvalue}；逐一去掉模型后的 ρ 范围 {span}（{6-len(values)} 个未定义）。",
            'P0 残差 RMS 敏感性：ρ = '+cell({'estimate':sensitivity['rho'],'interval95':sensitivity['interval95']})
            +f"。主统计 / RMS 的未定义 bootstrap 次数为 {primary['undefined_draws']} / {sensitivity['undefined_draws']}（各 10000 次），未删除未定义抽样。",
            '该检验针对目的选取的六个模型，E1 与 E4 议题分别重采样且各自跨模型共享权重；不是模型总体推断或选模校正后的推断。RMS 仅为敏感性分析，H1 的 Holm-2 校正保持不变。']
    else:
        lines+=['',f"H3 跨模型 Spearman(|C|, U) 尚未完成有效校验（{H3['status']}），因此整份 E4 报告仍为部分完成；单轮结果的独立完成状态保留。"]
    lines+=['','差异 RMS 衡量同一初始配置下两种分组结果的差异；E4b 始终在共同映射内比较 80 个设计。E4c 的 PK 为 pair-truncated 版本。',
            'E1 核对十种循环/反转位置取平均，E4 三人会固定使用第 2/4 槽并平均两种顺序，因此还假定影响核可以跨位置迁移。位置敏感性可能贡献预测误差；E4 内不同设计共享同一槽位规则，不构成设计之间的位置差异。',
            '主分析的区间同时传播 E1 核估计与 E4 议题不确定性；规则对照及动态区间条件于固定 E1 估计。仅六个独立议题，区间为点状 95%，不代表多重校正后的统一通过。','',
            '† 表示通过 conditional-q 操作门的模型，仍保留原始读数限制。正常 q 门子集的并列结果及各模型完整状态见 JSON。','',
            '| 模型 | 多轮状态 | G6 / STS 预算完成 | 第 6 轮 G6：PK−P0 ΔR² | 第 6 轮 STS：改善 / 退化 / 跨零 | 最大结果 MCSE |','|---|---|---|---:|---|---:|']
    for row in report['models']:
        d=row['dynamics'];final=d.get('final_round',{});b=final.get('E4b_PK_minus_P0_R2_by_initial_condition',{})
        decisions=Counter(verdict(metric) for metric in b.values())
        se_records=d.get('E4a_MCSE',[])+[r for records in d.get('E4b_MCSE',{}).values() for r in records]
        ses=[r.get('outcome_MCSE_maximum') for r in se_records if finite(r.get('outcome_MCSE_maximum'))]
        counts=' / '.join(str(decisions.get(key,0)) for key in ('improvement_supported_pointwise','degradation_supported_pointwise','uncertain_including_zero')) if b else '未提供'
        lines.append('| '+' | '.join([row['repo_id'].split('/')[-1],d['status'],str(d.get('E4a_budget_complete','—'))+' / '+str(d.get('E4b_budget_complete','—')),
            cell(final.get('E4a_PK_minus_P0_R2')),counts,f'{max(ses):.4g}' if ses else '未提供'])+' |')
    lines+=['','动态预算按 STS 每设计/初始条件总计 10 万轨迹、20 个映射各 5000；G6 每设计/条件 10 万。MCSE 在 JSON 单独报告，不等同于议题区间，也未冒称传播动态 E1 核不确定性。',
            'E3 按要求跳过；依赖它的 E5 暂缓。E6–E8 未运行，完整 E9 未运行；本报告不声称已做自由对话或内部机制验证。',
            'PK 优于加性 P0 本身不能证明所有使用两两读数的分组算法都不足；较强规则对照需一并判断。']
    if report['input_errors'] or report['selection']['status']!='verified':lines+=['','存在输入缺失或完整性问题，详见 JSON；相关结果未计作完成。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',required=True)
    p.add_argument('--aggregate',action='append');p.add_argument('--supplement',action='append');p.add_argument('--dynamics-aggregate',action='append')
    p.add_argument('--cross-model',type=Path)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--remote-root',default='/workspace/voices_nonadditivity_v3')
    p.add_argument('--output-prefix',type=Path,required=True);args=p.parse_args()
    paths=[Path(str(args.output_prefix)+s) for s in ('.json','.md')]
    if any(p.exists() for p in paths):raise FileExistsError('Refusing to overwrite a report snapshot')
    report=build(args.selection,args.aggregate,args.supplement,args.dynamics_aggregate,args.root,args.remote_root,args.cross_model)
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    for path,text in zip(paths,[json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',markdown(report)]):
        with path.open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'status':report['status'],'counts':report['counts'],'output':str(paths[0])}))


if __name__=='__main__':main()
