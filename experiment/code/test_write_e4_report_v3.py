"""Receipt/partial-report tests using clearly synthetic, CPU-only fixtures."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_v3 import sha
from write_e4_report_v3 import build,markdown,dynamics_summary


REMOTE='/workspace/voices_nonadditivity_v3'
TOPICS=[f'T{i}' for i in range(6)]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    return path


def interval(value):
    return {'estimate':value,'interval95':[value-.05,value+.05],'undefined_draws':0}


def main_value(value):
    result=interval(value);bounds={k:v for k,v in result.items() if k!='estimate'}
    return {'estimate':value,'conditional_on_E1_point_kernel':copy.deepcopy(bounds),
            'E1_kernel_and_E4_topic_propagated':copy.deepcopy(bounds)}


def fixture(root,with_dynamics=True):
    """No model scores here: only output-schema receipts for report validation."""
    def recorded(path):return REMOTE+'/'+path.relative_to(root).as_posix()
    manifest=write(root/'data/e4_cache_manifest.json',{'topics':[{'topic_id':tid} for tid in TOPICS]})
    selection={'schema':'voices_v3_frozen_E4_model_selection_v1','status':'frozen','performed':True,
        'all_requested_E1_complete':True,'selected_repos':[f'fixture/model{i}' for i in range(6)],
        'selected_conditional_option_mass_flags':{f'fixture/model{i}':i==0 for i in range(6)}}
    panel={'all_requested_E1_complete':True,'models':[],'E4_selection':copy.deepcopy(selection)}
    aggregates=[];supplements=[]
    for i,repo in enumerate(selection['selected_repos']):
        directory=root/f'results/model{i}';identity={'repo':repo,'revision':'synthetic','precision':'fp32',
            'remedy':'original','model_kind':'fixture','option_ids':{'A':1,'B':2}}
        e1=write(directory/'e1_primary.json',{'fixture':True,'repo':repo,'readout':'R0','n_items':200,
            'topics':['E1_fixture']*200,'point':{'C':(-1 if i%2 else 1)*(.1+.05*i)},
            'execution_identity':{'critical_identity':identity}})
        hashes={recorded(manifest):sha(manifest),recorded(e1):sha(e1)}
        panel['models'].append({'repo_id':repo,'E4_eligible':True,'E1_complete':True,
            'conditional_option_mass_admission':i==0,'R0_readout_disagreement_flag':False,'E0_tier':'R0_primary',
            'primary':{'execution_identity':identity,'sha256':sha(e1)}})
        for tid in TOPICS:
            cache=directory/(tid+'_caches.npz')
            with cache.open('wb') as stream:np.savez(stream,fixture=np.array([i]))
            point=write(directory/(tid+'_E4_analysis.json'),{'topic_id':tid,
                'execution_identity':{'critical_identity':identity},
                'cache_artifact':{'filename':cache.name,'sha256':sha(cache)},
                'E4a':{'n_conditions':70},'E4b':{'n_systems':80,'n_mappings':20,
                    'configurations':[{'n_target':m,'tau':t} for m in (3,5,7) for t in (-1,1)]},
                'E4c':{'n_conditions':192}})
            hashes[recorded(point)]=sha(point);hashes[recorded(cache)]=sha(cache)
        P0=.2+.01*i;PK=.5+.01*i;U=1-P0;observed=.1+.005*i
        values={'observed_contrast_RMS':observed,'P0_R2_delta':P0,'PK_R2_delta':PK,
            'P0_unexplained_fraction_U':U,'P0_residual_RMS':math.sqrt(U)*observed,
            'PK_minus_P0_R2':.3,'MSE_improvement_P0_minus_PK':.3*observed**2,
            'fraction_P0_residual_squared_explained_by_PK':.3/U}
        aggregate={'model_identity':identity,'topic_ids':TOPICS,'n_topics':6,'planned_six_topics_complete':True,
            'bootstrap':{'draws':10000,'unit':'topic'},'single_round_E1_kernel_uncertainty_propagated':True,
            'single_round':{part:{'metrics':{k:main_value(v) for k,v in values.items()},
                'topic_sufficient_statistics':{tid:{} for tid in TOPICS}} for part in ('E4a','E4b','E4c')},
            'file_sha256':hashes,'dynamic':{'status':'not_run'}}
        if with_dynamics:
            configurations=[f'm{m}_tau{t}' for m in (3,5,7) for t in (-1,1)]
            rounds={str(r):{'PK_minus_P0_R2':main_value(.3)} for r in range(1,7)}
            aggregate['dynamic']={'status':'complete_conditional_topic_aggregation','E4a_complete_topic_panel':True,
                'E4b_complete_topic_panel':True,'E1_kernel_parameter_uncertainty_propagated':False,
                'E4a_by_round':copy.deepcopy(rounds),
                'by_configuration_and_round':{k:copy.deepcopy(rounds) for k in configurations},
                'MCSE':{k:[{'topic_id':tid,'draws':5000,'total_draws_per_design':100000,
                    'outcome_MCSE_maximum':.003} for tid in TOPICS] for k in configurations},
                'E4a_MCSE':[{'topic_id':tid,'draws_per_design_per_condition':100000,
                    'outcome_MCSE_maximum':.001} for tid in TOPICS]}
        sidecar=directory/'aggregate_bootstrap.npz'
        with sidecar.open('wb') as stream:np.savez(stream,fixture=np.array([i]))
        aggregate['bootstrap_artifact']={'filename':sidecar.name,'sha256':sha(sidecar)}
        aggregates.append(write(directory/'aggregate.json',aggregate))
        methods={'additive':P0,'present_mean':.7+.01*i,'side_mean':.6+.01*i,'side_max':.4+.01*i,'PK':PK}
        supplement={'schema':'voices_v3_E4_single_round_pairwise_rule_supplement_v1','repo_id':repo,
            'topic_ids':TOPICS,'n_topics':6,'planned_six_topics_complete':True,
            'primary_P0_PK_U_definitions_unchanged':True,'E4_fit_parameters':0,
            'E1_analysis_sha256':sha(e1),'E1_best_rule':{'rule':'present_mean'},
            'bootstrap':{'draws':10000,'unit':'topic','same_topic_draws_for_all_methods':True},
            'single_round':{part:{'methods':{method:{'R2_delta':interval(value)} for method,value in methods.items()},
                'PK_improvement_over_each_rule':{method:{'R2_delta_improvement':interval(PK-value)} for method,value in methods.items() if method!='PK'},
                'PK_improvement_over_E1_selected_rule':{'R2_delta_improvement':interval(-.2)}} for part in ('E4a','E4b','E4c')},
            'file_sha256':hashes}
        sidecar=directory/'supplement_outcomes_and_bootstrap.npz'
        with sidecar.open('wb') as stream:np.savez(stream,fixture=np.array([i]))
        supplement['outcomes_and_bootstrap_artifact']={'filename':sidecar.name,'sha256':sha(sidecar)}
        supplements.append(write(directory/'supplement.json',supplement))
    panel_path=write(root/'panel.json',panel)
    selection.update(panel_path=recorded(panel_path),panel_sha256=sha(panel_path))
    return write(root/'selection.json',selection),aggregates,supplements


def synthetic_H3(root,selection_path,aggregates):
    """Synthetic cross-model receipts only; no model execution or bootstrap claim."""
    def recorded(path):return REMOTE+'/'+path.relative_to(root).as_posix()
    selection=json.loads(selection_path.read_text(encoding='utf-8'));repos=selection['selected_repos']
    bindings={recorded(selection_path):sha(selection_path)};models=[]
    for repo,path in zip(repos,aggregates):
        data=json.loads(path.read_text(encoding='utf-8'));bindings.update(data['file_sha256']);bindings[recorded(path)]=sha(path)
        e1_path=path.parent/'e1_primary.json';e1=json.loads(e1_path.read_text(encoding='utf-8'))
        metrics=data['single_round']['E4b']['metrics']
        models.append({'repo_id':repo,'E1_analysis_sha256':sha(e1_path),'aggregate_sha256':sha(path),
            'C':e1['point']['C'],'U':metrics['P0_unexplained_fraction_U']['estimate'],
            'P0_residual_RMS':metrics['P0_residual_RMS']['estimate'],'n_E1_items':200})
    def rho(x,y):
        def ranks(values):
            ordered=sorted(values)
            return np.array([np.mean([j+1 for j,v in enumerate(ordered) if v==value]) for value in values])
        a,b=ranks(x),ranks(y)
        return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
    x=[abs(row['C']) for row in models]
    def statistic(key,estimand):
        y=[row[key] for row in models];r=rho(x,y)
        return {'estimand':estimand,'rho':r,'interval95':None if r is None else [r,r],
            'undefined_draws':10000 if r is None else 0,'n_permutations':720,
            'permutation_p_one_sided_positive':None if r is None else sum(rho(x,list(order))>=r-1e-12 for order in itertools.permutations(y))/720,
            'leave_one_model_out':[{'omitted_repo':repo,'rho':rho([v for j,v in enumerate(x) if j!=i],[v for j,v in enumerate(y) if j!=i])} for i,repo in enumerate(repos)]}
    primary=statistic('U','Spearman(abs(E1_C), E4b_single_round_P0_unexplained_fraction_U)')
    sensitivity=statistic('P0_residual_RMS','Spearman(abs(E1_C), E4b_single_round_P0_residual_RMS)')
    sidecar=root/'H3_bootstrap.npz'
    with sidecar.open('wb') as stream:
        np.savez(stream,rho_draws=np.full(10000,np.nan if primary['rho'] is None else primary['rho']))
    value={'schema':'voices_v3_E4_cross_model_H3_v1','status':'complete_verified_for_selected_panel',
        'selection_sha256':sha(selection_path),'selected_repos':repos,'n_models':6,'topic_ids':TOPICS,'models':models,
        'primary':primary,'sensitivity':{'P0_residual_RMS':sensitivity},'bootstrap':{'draws':10000,'unit':'topic',
            'E1_topics':['E1_fixture'],'E4_topic_ids':TOPICS,'E1_topic_weights_shared_across_models':True,
            'E4_topic_weights_shared_across_models':True,'E1_and_E4_topic_resampling_independent':True,
            'existing_model_bootstrap_rows_concatenated':False,'undefined_draws_not_dropped':True},
        'bootstrap_artifact':{'filename':sidecar.name,'sha256':sha(sidecar)},'file_sha256':bindings}
    return write(root/'H3.json',value)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.selection,self.aggregates,self.supplements=fixture(self.root)
        self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)

    def report(self,aggregates=None,supplements=None,cross_model=True):
        return build(self.selection,self.aggregates if aggregates is None else aggregates,
            self.supplements if supplements is None else supplements,root=self.root,
            cross_model=self.cross_model if cross_model is True else cross_model)

    def mutate(self,path,change):
        data=json.loads(path.read_text(encoding='utf-8'));change(data);write(path,data)

    def test_closed_six_remote_path_bindings_and_negative_stronger_baseline(self):
        report=self.report()
        self.assertEqual(report['status'],'complete_E4_main_and_supplement')
        self.assertTrue(report['single_round_stage_complete'])
        self.assertTrue(report['cross_model_stage_complete'])
        self.assertEqual(report['cross_model_H3']['status'],'complete_verified')
        self.assertEqual(report['counts']['conditional_option_mass_models'],1)
        verdicts=report['E4b_pointwise_verdict_counts']
        self.assertEqual(verdicts['all_selected_models']['PK_vs_P0']['improvement_supported_pointwise'],6)
        self.assertEqual(verdicts['all_selected_models']['PK_vs_E1_selected_rule']['degradation_supported_pointwise'],6)
        self.assertEqual(verdicts['normal_option_mass_subset']['n_models'],5)
        text=markdown(report)
        self.assertIn('支持退化 6 个',text);self.assertIn('不是自由生成对话',text)
        self.assertIn('-0.2 [-0.25, -0.15]',text);self.assertIn('pair-truncated',text)
        self.assertIn('Spearman(|C|, U)',text);self.assertIn('720 个完整排列',text)
        self.assertIn('Holm-2',text)
        self.assertIn(str(self.cross_model.resolve()),report['verified_source_sha256'])
        self.assertIn(str((self.root/'H3_bootstrap.npz').resolve()),report['verified_source_sha256'])
        json.dumps(report,allow_nan=False)

    def test_missing_results_are_pending_not_zero_or_failed_science(self):
        report=self.report([],[])
        self.assertEqual(report['status'],'partial_or_pending')
        self.assertEqual(report['counts']['single_round_complete_models'],0)
        self.assertTrue(all(r['single_round_status']=='pending' for r in report['models']))
        self.assertEqual(report['E4b_pointwise_verdict_counts']['all_selected_models']['PK_vs_P0'],{})

    def test_cache_tamper_and_missing_sidecar_cannot_complete(self):
        (self.root/'results/model0/T0_caches.npz').write_bytes(b'tampered synthetic receipt')
        (self.root/'results/model1/aggregate_bootstrap.npz').unlink()
        report=self.report()
        self.assertEqual(report['models'][0]['single_round_status'],'invalid')
        self.assertEqual(report['models'][1]['single_round_status'],'partial_local_sync')
        self.assertEqual(report['counts']['single_round_complete_models'],4)
        self.assertEqual(report['status'],'partial_or_pending')

    def test_incomplete_dynamic_budget_keeps_single_round_complete(self):
        self.mutate(self.aggregates[0],lambda r:r['dynamic']['MCSE']['m3_tau1'][0].update(draws=4999,total_draws_per_design=99980))
        self.mutate(self.aggregates[1],lambda r:r.update(dynamic={'status':'not_run'}))
        report=self.report()
        self.assertTrue(report['single_round_stage_complete'])
        self.assertEqual(report['counts']['dynamics_complete_models'],4)
        self.assertEqual(report['status'],'partial_or_pending')
        self.assertFalse(report['models'][0]['dynamics']['E4b_budget_complete'])

    def test_numeric_match_does_not_replace_shared_topic_source_binding(self):
        original=self.root/'results/model0/T0_E4_analysis.json'
        other=self.root/'other/T0_E4_analysis.json';value=json.loads(original.read_text());value['different_fixture']=True;write(other,value)
        def change(data):
            key=next(k for k in data['file_sha256'] if k.endswith('/T0_E4_analysis.json'))
            del data['file_sha256'][key];data['file_sha256'][str(other)]=sha(other)
        self.mutate(self.supplements[0],change)
        report=self.report()
        self.assertEqual(report['models'][0]['supplement_status'],'invalid_primary_binding')
        self.assertEqual(report['counts']['supplement_complete_models'],5)

    def test_different_E1_source_or_frozen_selection_cannot_complete(self):
        self.mutate(self.supplements[0],lambda r:r.update(E1_analysis_sha256='another source'))
        report=self.report()
        self.assertEqual(report['models'][0]['supplement_status'],'invalid_primary_binding')
        self.mutate(self.selection,lambda r:r.update(panel_sha256='wrong'))
        report=self.report()
        self.assertEqual(report['selection']['status'],'invalid');self.assertFalse(report['single_round_stage_complete'])

    def test_two_topics_are_reported_as_partial_and_do_not_enter_verdict_counts(self):
        keep=TOPICS[:2]
        def shrink(data):
            data.update(topic_ids=keep,n_topics=2,planned_six_topics_complete=False)
            for section in data['single_round'].values():section['topic_sufficient_statistics']={tid:{} for tid in keep}
        self.mutate(self.aggregates[0],shrink)
        report=self.report()
        self.assertEqual(report['models'][0]['single_round_status'],'partial_topics_or_bootstrap')
        self.assertEqual(report['counts']['single_round_complete_models'],5)
        self.assertIn('[2/6议题]',markdown(report))

    def test_undefined_and_crossing_intervals_are_not_positive_findings(self):
        def change(data):
            m=data['single_round']['E4b']['metrics']['PK_minus_P0_R2']
            m['E1_kernel_and_E4_topic_propagated']['interval95']=[-.1,.4]
        self.mutate(self.aggregates[0],change)
        def undefined(data):
            m=data['single_round']['E4b']['metrics']['PK_minus_P0_R2']
            m['E1_kernel_and_E4_topic_propagated'].update(interval95=None,undefined_draws=1)
        self.mutate(self.aggregates[1],undefined)
        report=self.report();counts=report['E4b_pointwise_verdict_counts']['all_selected_models']['PK_vs_P0']
        self.assertEqual(counts['improvement_supported_pointwise'],4)
        self.assertEqual(counts['uncertain_including_zero'],1);self.assertEqual(counts['interval_undefined'],1)

    def test_missing_H3_keeps_single_round_complete_but_report_partial(self):
        report=self.report(cross_model=None)
        self.assertTrue(report['single_round_stage_complete'])
        self.assertFalse(report['cross_model_stage_complete'])
        self.assertEqual(report['status'],'partial_or_pending')
        self.assertIn('H3 跨模型',markdown(report))

    def test_H3_wrong_model_panel_or_topic_is_invalid(self):
        self.mutate(self.cross_model,lambda r:r['models'][0].update(repo_id='fixture/not_selected'))
        report=self.report()
        self.assertEqual(report['cross_model_H3']['status'],'invalid')
        self.assertTrue(report['single_round_stage_complete'])
        self.assertEqual(report['status'],'partial_or_pending')
        self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)
        self.mutate(self.cross_model,lambda r:r['topic_ids'].__setitem__(0,'unplanned_topic'))
        self.assertEqual(self.report()['cross_model_H3']['status'],'invalid')

    def test_H3_wrong_selection_source_SHA_and_sidecar_cannot_complete(self):
        self.mutate(self.cross_model,lambda r:r.update(selection_sha256='wrong'))
        self.assertEqual(self.report()['cross_model_H3']['status'],'invalid')
        self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)
        (self.root/'H3_bootstrap.npz').write_bytes(b'changed sidecar')
        report=self.report()
        self.assertFalse(report['cross_model_stage_complete'])
        self.assertTrue(report['single_round_stage_complete'])

    def test_H3_wrong_bound_C_U_or_aggregate_SHA_is_invalid(self):
        for key,value in [('C',999),('U',999),('P0_residual_RMS',999),('aggregate_sha256','wrong'),('E1_analysis_sha256','wrong')]:
            with self.subTest(key=key):
                self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)
                self.mutate(self.cross_model,lambda r:r['models'][0].update({key:value}))
                report=self.report()
                self.assertEqual(report['cross_model_H3']['status'],'invalid')
                self.assertEqual(report['status'],'partial_or_pending')

    def test_H3_wrong_rho_or_permutation_probability_is_invalid(self):
        for key,value in [('rho',.25),('permutation_p_one_sided_positive',.25)]:
            with self.subTest(key=key):
                self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)
                self.mutate(self.cross_model,lambda r:r['primary'].update({key:value}))
                self.assertEqual(self.report()['cross_model_H3']['status'],'invalid')

    def test_undefined_rho_is_completed_measurement_without_positive_threshold(self):
        for path in self.aggregates:
            self.mutate(path,lambda r:r['single_round']['E4b']['metrics']['P0_unexplained_fraction_U'].update(estimate=.8))
        self.cross_model=synthetic_H3(self.root,self.selection,self.aggregates)
        report=self.report()
        self.assertTrue(report['cross_model_stage_complete'])
        self.assertEqual(report['status'],'complete_E4_main_and_supplement')
        self.assertIsNone(report['cross_model_H3']['primary']['rho'])
        self.assertIsNone(report['cross_model_H3']['primary']['interval95'])
        self.assertEqual(report['cross_model_H3']['primary']['undefined_draws'],10000)
        self.assertIn('检验已完成',markdown(report));self.assertIn('未定义',markdown(report))


if __name__=='__main__':unittest.main()
