"""Regenerate the writing tables from the curated numerical evidence."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def readcsv(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def readjson(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def write(name,rows):
    if not rows:raise ValueError('Empty writing table '+name)
    p=ROOT/'tables'/name;p.parent.mkdir(exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    group=[r for r in readcsv('analysis/paired_designs.csv') if r['stage']=='G2' and r['first']=='matched_size' and r['second']=='matched_seat']
    write('01_same_people_group_effect.csv',group)
    scores=[r for r in readcsv('analysis/method_scores.csv') if r['stage']=='G2' and r['family']=='E4c']
    write('02_meeting_size_predictors.csv',scores)
    gains=[r for r in readcsv('analysis/paired_predictors.csv') if r['stage']=='G2' and r['family']=='E4c']
    write('03_prediction_improvements.csv',gains)
    neutral={r['model']:r for r in readcsv('micro/neutral_summary.csv')}
    micro=[]
    for p in sorted((ROOT/'inputs').glob('*__e2e_R0_analysis.json')):
        d=json.loads(p.read_text());model=p.name.split('__')[0];ci=d['bootstrap']['intervals']['power_t_no_allies']['interval95'];n=neutral[model]
        micro.append({'model':model,'n_items':d['n_items'],'n_topics':d['n_topics'],'power_t_1to6':d['point']['power_t_no_allies'],
            'power_t_ci_lo':ci[0] if ci else None,'power_t_ci_hi':ci[1] if ci else None,
            'neutral_delta_logit':n['paired_logit_difference'],'neutral_ci_lo':n['ci_lo'],'neutral_ci_hi':n['ci_hi'],
            'classification':'continuous_effects_exploratory','source':'inputs/'+p.name})
    write('04_micro_count_and_neutral_effects.csv',micro)
    old=readjson('inputs/G4__final__report.json');original=[]
    for r in old['primary_comparisons']:
        original.append({'dataset':'G4_original','model':r['repo_id'],'family':r['family'],'n_topics':6,
                         'n_runs_family':48 if r['family']=='G6' else 240,'P0_minus_PK_logit_mse':r['estimate'],
                         'ci_lo':r['interval95'][0],'ci_hi':r['interval95'][1],'Holm_p_six_tests':r['Holm_p']})
    write('05_original_dialogue_prediction.csv',original)
    new=readjson('dialogue/replication_3090/replication_report.json');rep=[]
    compare={r['family']:r for r in new['old_new_comparison']}
    for r in new['families']:
        a=r['primary_conditional_loss_improvement'];c=compare[r['family']]
        rep.append({'dataset':'LFM_3090_20261106','model':'LiquidAI/LFM2.5-1.2B-Instruct','family':r['family'],'n_topics':6,
                    'n_runs_family':48 if r['family']=='G6' else 240,'P0_minus_PK_logit_mse':a['estimate'],
                    'ci_lo':a['interval95'][0],'ci_hi':a['interval95'][1],'Holm_p_two_tests':r['replication_two_family_Holm_p'],
                    'new_minus_original_gain':c['new_minus_old_conditional_gain'],'paired_ci_lo':c['interval95'][0],'paired_ci_hi':c['interval95'][1]})
    write('06_LFM_3090_replication.csv',rep)
    gate=readjson('historical/analysis_results/Gate2/report.json');tests=[]
    for r in gate['primary_family']:
        estimates=r.get('estimates',{r['name']:r.get('estimate',str(r.get('agreements',''))+'/'+str(r.get('models','')))})
        for part,estimate in estimates.items():
            ci=r.get('CI95',{})
            if isinstance(ci,dict):ci=ci.get(part)
            tests.append({'test':r['name'],'component':part,'estimate':estimate,
                          'ci_lo':ci[0] if ci else None,'ci_hi':ci[1] if ci else None,'Holm_p':r['Holm_p'],
                          'passed_R0_family':r['passes_R0_family_alpha05'],
                          'R1_matched_direction':r['matched_G3_R1_point_direction_agrees'],
                          'source':'historical/analysis_results/Gate2/report.json'})
    write('07_frozen_confirmatory_tests.csv',tests)
    write('08_LFM_original_replication_pairs.csv',readcsv('dialogue/replication_3090/old_vs_3090.csv'))
    versions=[]
    for model in sorted(r['model'] for r in micro):
        path='provenance/model_snapshots/'+model+'.json';snapshot=readjson(path)
        versions.append({'model':model,'repo_id':snapshot['repo_id'],
                         'archived_download_revision':snapshot['revision'],
                         'license':snapshot.get('license',''),'snapshot_status':snapshot['status'],
                         'source_url':snapshot['source'],'metadata_path':path})
    write('09_model_versions.csv',versions)
    print('WRITING_TABLES_READY',flush=True)
if __name__=='__main__':main()

