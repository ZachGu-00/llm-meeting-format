"""Independent, read-only E0 selection and closed E1 readout-quality audit.

Writes only the requested audit outputs. No admission rules, effect estimates,
row filtering, model loading, GPU calls, queue changes or protocol changes.
Raw/input JSONL are streamed together; only bounded q summaries are retained.
"""
import argparse
from array import array
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODES=('original','t2','prefill')
RECOVERED='voices_v3_E0_taskpilot_driver_v1'
CONDITIONAL='voices_v3_E0_conditional_driver_v1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def local(value,root):
    text=str(value).replace('\\','/')
    for marker in ('/workspace/voices_nonadditivity_v3/','/voices_nonadditivity_v3_20261030/'):
        if marker in text:return root/text.split(marker,1)[1]
    p=Path(text)
    return p if p.is_absolute() else root/p


class QStats:
    """Exact moments/tail counts; bounded exact quantiles or explicit histogram intervals."""
    def __init__(self,exact_limit=0,bins=1000):
        self.n=0;self.invalid=0;self.total=0.;self.minimum=None;self.maximum=None
        self.below={str(x):0 for x in (.1,.5,.7,.9)}
        self.hist=np.zeros(bins,dtype=np.int64);self.values=array('d') if exact_limit else None
        self.limit=exact_limit
    def add(self,q):
        if isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or not 0<=q<=1:
            self.invalid+=1;return
        self.n+=1;self.total+=q;self.minimum=q if self.minimum is None else min(self.minimum,q)
        self.maximum=q if self.maximum is None else max(self.maximum,q)
        for threshold in self.below:self.below[threshold]+=int(q<float(threshold))
        self.hist[min(len(self.hist)-1,int(q*len(self.hist)))]+=1
        if self.values is not None:
            if self.n<=self.limit:self.values.append(q)
            else:self.values=None
    def report(self):
        result={'valid_q_rows':self.n,'invalid_or_missing_q_rows':self.invalid,
            'min':self.minimum,'max':self.maximum,'mean':self.total/self.n if self.n else None,
            'below_counts':self.below,'below_fractions':{k:v/self.n if self.n else None for k,v in self.below.items()}}
        if not self.n:return result
        probabilities=(.01,.05,.25,.5,.75,.95,.99)
        if self.values is not None:
            result.update(quantile_method='exact_linear_numpy',quantiles=dict(zip(map(str,probabilities),map(float,np.quantile(self.values,probabilities)))))
        else:
            cumulative=np.cumsum(self.hist);width=1/len(self.hist)
            result.update(quantile_method='bounded_histogram_inverse_CDF_intervals',histogram_bin_width=width,
                quantile_intervals={str(p):[int(np.searchsorted(cumulative,max(1,math.ceil(p*self.n))))/len(self.hist),
                    min(1.,(int(np.searchsorted(cumulative,max(1,math.ceil(p*self.n))))+1)/len(self.hist))] for p in probabilities})
        return result


def iter_rows(path,digest):
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip():yield json.loads(line)


def calibration(path,root):
    path=local(path,root);lock=load(path)
    if lock.get('lock_content_sha256')!=hashlib.sha256(encoded({k:v for k,v in lock.items() if k!='lock_content_sha256'})).hexdigest():
        raise ValueError('Calibration lock content hash differs: '+str(path))
    source=root/'data/calibration_prompts.jsonl'
    if sha(source)!=lock['input_file_sha256']:raise ValueError('Calibration input differs')
    return lock,{'file':str(path.relative_to(root)),'file_sha256':sha(path),
        'lock_content_sha256':lock['lock_content_sha256'],'input_sha256':lock['input_file_sha256'],
        'median_q':lock['median_q'],'argmax_format_proxy':lock['greedy_any_variant_fraction'],
        'option_ids':lock['option_ids'],'diagnostic_scope':'20 unrelated token-calibration prompts; not actual-task qualification'}


def audit_closed_raw(raw,input_path,lockpath,root,exact_limit=250000,old_ids=None):
    raw=Path(raw);rp=Path(str(raw)+'.run.json');sp=Path(str(raw)+'.summary.json')
    missing=[str(p) for p in (raw,rp,sp,Path(input_path),Path(lockpath)) if not p.is_file()]
    if missing:return {'status':'not_closed_or_not_synced','missing':missing}
    run,summary=load(rp),load(sp);lock,cal=calibration(lockpath,root);runhash=sha(rp)
    if (summary.get('status')!='complete' or not summary.get('exact_id_and_count_equality') or
        summary.get('n_output')!=run.get('n_input') or summary.get('run_file_sha256')!=runhash or
        any(summary.get(k)!=v for k,v in run.items()) or run.get('readout') not in ('R0','R0_R1') or
        run.get('calibration_file_sha256')!=cal['file_sha256'] or run.get('engine')!=lock['engine'] or
        run.get('option_ids')!=lock['option_ids'] or run.get('batch_size')!=lock['batch_size']):
        raise ValueError('Run/summary/current calibration identities disagree: '+str(raw))
    overall=QStats(exact_limit);groups=defaultdict(QStats)
    rh,ih,ids_hash,rows_hash=(hashlib.sha256() for _ in range(4));ids_hash.update(b'[');rows_hash.update(b'[')
    format_ids=set(run.get('format_record_ids',[]));r2_ids=set(run.get('r2_record_ids',[]))
    diag={name:{'expected':len(ids),'seen':0,'missing_generated':0,'successes':0,'ties':0} for name,ids in [('format',format_ids),('R2',r2_ids)]}
    count=0;argmax_nonletter=0;item_ids=set();item_limit_exceeded=False
    for source,row in itertools.zip_longest(iter_rows(input_path,ih),iter_rows(raw,rh)):
        if source is None or row is None:raise ValueError('Input/output row counts differ')
        source['record_id']=source.get('record_id',source.get('id'));rid=source['record_id']
        if row.get('run_file_sha256')!=runhash or row.get('status')!='ok' or any(row.get(k)!=v for k,v in source.items()):
            raise ValueError('Raw source/run identity differs at row '+str(count))
        if count:ids_hash.update(b',');rows_hash.update(b',')
        ids_hash.update(encoded(rid));rows_hash.update(encoded(source));count+=1
        item=row.get('item_id')
        if item not in item_ids:
            if len(item_ids)<10000:item_ids.add(item)
            else:item_limit_exceeded=True
        q=row.get('q');overall.add(q)
        mask=row.get('subset_mask');size=mask.bit_count() if isinstance(mask,int) and mask>=0 else 'unknown'
        keys={'subset_size':str(size),'label_map':str(row.get('label_map')),'positive_letter':str(row.get('positive_letter')),
              'permutation':str(row.get('permutation')),'subset_size_x_label':str(size)+'/'+str(row.get('label_map'))}
        if old_ids is not None:keys['freshness']='prior20' if row.get('item_id') in old_ids else 'fresh180'
        for dimension,value in keys.items():
            if len(groups)>256 and (dimension,value) not in groups:value='OTHER'
            groups[dimension,value].add(q)
        argmax_nonletter+=row.get('greedy_argmax_any_variant_letter') not in ('A','B')
        for name,selected in [('format',format_ids),('R2',r2_ids)]:
            if rid not in selected:continue
            d=diag[name];d['seen']+=1
            text=row.get('generated_first_token_text')
            flag='diagnostic_format_selected' if name=='format' else 'diagnostic_r2_selected'
            if row.get(flag) is not True or not isinstance(text,str) or 'generated_first_token_id' not in row:
                d['missing_generated']+=1;continue
            letter=text.strip() if text.strip() in ('A','B') else None
            if name=='format':d['successes']+=letter is not None
            else:
                value=row.get('R0_y_AB',row.get('y_AB'))
                expected='A' if value>0 else 'B' if value<0 else None
                d['ties']+=expected is None;d['successes']+=expected is not None and letter==expected
    ids_hash.update(b']');rows_hash.update(b']')
    if (count!=run['n_input'] or rh.hexdigest()!=summary['output_sha256'] or raw.stat().st_size!=summary['output_bytes'] or
        ih.hexdigest()!=run['input_file_sha256'] or ids_hash.hexdigest()!=run['input_record_ids_sha256'] or
        rows_hash.hexdigest()!=run['input_rows_sha256']):raise ValueError('Streaming count/source/output hashes disagree')
    for d in diag.values():
        d['fraction']=d['successes']/d['expected'] if d['expected'] and d['seen']==d['expected'] and not d['missing_generated'] else None
    grouped=defaultdict(dict)
    for (dimension,value),stats in groups.items():grouped[dimension][value]=stats.report()
    return {'status':'verified_closed','raw_file':str(raw.relative_to(root)),'n_rows':count,
        'n_items':None if item_limit_exceeded else len(item_ids),'unique_item_memory_limit_exceeded':item_limit_exceeded,
        'raw_sha256':rh.hexdigest(),'input_sha256':ih.hexdigest(),'run_sha256':runhash,'summary_sha256':sha(sp),
        'calibration':cal,'remedy':run['engine']['remedy'],'batch_size':run['batch_size'],
        'engine_sha256':hashlib.sha256(encoded(run['engine'])).hexdigest(),
        'q':overall.report(),'q_by':dict(grouped),'generated_diagnostics':diag,
        'full_vocab_argmax_not_A_or_B_fraction':argmax_nonletter/count if count else None,
        'low_q_interpretation':'R0 A/B log-odds still measure relative preference, but low full-vocabulary A+B mass weakens alignment with an unconstrained first-token answer. Report these rows; this audit does not remove them or change model eligibility.'}


def safe_raw(*args,**kwargs):
    try:return audit_closed_raw(*args,**kwargs)
    except (OSError,ValueError,KeyError,TypeError) as error:return {'status':'audit_error','error':str(error)}


def verify_analysis_sources(analysis,stream,root):
    """Use the just-streamed hashes for large files; hash other source artifacts once."""
    sources=analysis.get('file_sha256',{})
    if not sources:raise ValueError('Selected analysis has no source hashes')
    verified=[]
    for value,expected in sources.items():
        path=local(value,root);substitution=None
        if path==root/stream['raw_file']:actual=stream['raw_sha256']
        elif path.name in ('pilot7_prompts.jsonl','development7_prompts.jsonl'):actual=stream['input_sha256']
        else:actual=sha(path)
        if actual!=expected and path==root/'execution_protocol.txt':
            archived=root/'provenance/execution_protocol_E0_original.txt'
            if archived.is_file() and sha(archived)==expected:
                actual=expected;substitution=str(archived.relative_to(root))
        if actual!=expected:raise ValueError('Selected analysis source hash differs: '+str(path))
        verified.append({'file':str(path.relative_to(root)),'sha256':expected,'archived_snapshot':substitution})
    return verified


def analyze_e0(folder,root,exact_limit):
    statepath=folder/'e0_driver_state.json'
    if not statepath.is_file():return {'status':'state_not_synced'}
    state=load(statepath);history=[];cursor=state
    for _ in range(32):
        if not cursor.get('previous_state_snapshot'):break
        p=local(cursor['previous_state_snapshot'],root)
        if sha(p)!=cursor['previous_state_sha256']:raise ValueError('Prior E0 state snapshot hash differs')
        cursor=load(p);history.append({'file':str(p.relative_to(root)),'sha256':sha(p),'schema':cursor.get('schema'),
            'selected_remedy':cursor.get('selected_remedy'),'final_qualification':cursor.get('final_qualification'),
            'calibration_only_skips':[a.get('remedy') for a in cursor.get('attempts',[]) if a.get('decision')=='advance_due_to_calibration_format_or_q']})
    skips=state.get('original_calibration_only_skips') or [a.get('remedy') for a in state.get('attempts',[]) if a.get('decision')=='advance_due_to_calibration_format_or_q']
    amended=state.get('schema') in (RECOVERED,CONDITIONAL)
    conditional=state.get('schema')==CONDITIONAL
    if amended and sha(local(state['amendment_file'],root))!=state['amendment_sha256']:raise ValueError('E0 routing amendment changed')
    selected=state.get('selected_remedy');attempts=[]
    for remedy in MODES:
        rdir=folder/remedy;lockpath=rdir/'token_calibration.json'
        if not lockpath.is_file():continue
        _,cal=calibration(lockpath,root);row={'remedy':remedy,'selected_now':remedy==selected,'calibration':cal}
        decision=next((a.get('decision') for a in state.get('attempts',[]) if a.get('remedy')==remedy),None)
        row['current_driver_decision']=decision
        for stage,inp in [('pilot','pilot7_prompts.jsonl'),('development','development7_prompts.jsonl')]:
            ap=rdir/(stage+'_E0_analysis.json');raw=rdir/(stage+'_R0_R1.jsonl')
            state_attempt=next((a for a in state.get('attempts',[]) if a.get('remedy')==remedy),{})
            declared=state_attempt.get(stage,{}).get('analysis')
            if declared:ap=local(declared,root)
            elif conditional and remedy==selected:ap=rdir/(stage+'_E0_conditional_analysis.json')
            if not ap.exists():row[stage]={'status':'not_measured_or_not_synced'};continue
            a=load(ap);entry={'analysis_file':str(ap.relative_to(root)),'analysis_sha256':sha(ap),
                'n_items':a.get('n_items'),'bootstrap_draws':a.get('bootstrap_draws'),
                'operational_gates':a.get('operational_gates'),'qualification':a.get('qualification'),
                'operational_policy':a.get('operational_policy'),
                'conditional_policy_provenance':a.get('conditional_policy_provenance'),
                'historical_analysis_receipt_status':a.get('execution_integrity',{}).get('R0',{}).get('status')}
            sp=Path(str(raw)+'.summary.json');rp=Path(str(raw)+'.run.json')
            if sp.exists() and rp.exists():
                summary=load(sp);receipt=a.get('execution_integrity',{}).get('R0',{})
                entry['closed_result_receipt']={'raw_file':str(raw.relative_to(root)),
                    'declared_raw_sha256':summary.get('output_sha256'),'run_sha256':sha(rp),'summary_sha256':sha(sp),
                    'sidecars_match_analysis':receipt.get('run_sha256')==sha(rp) and receipt.get('summary_sha256')==sha(sp)}
            if remedy==selected:
                entry['stream_audit']=safe_raw(raw,root/'data'/inp,lockpath,root,exact_limit)
                receipt=a.get('execution_integrity',{}).get('R0',{})
                stream=entry['stream_audit']
                if stream.get('status')=='verified_closed' and (receipt.get('run_sha256')!=stream['run_sha256'] or receipt.get('summary_sha256')!=stream['summary_sha256']):
                    stream.update(status='audit_error',error='Selected analysis does not bind the streamed sidecars')
                if stream.get('status')=='verified_closed':
                    checks={}
                    reported=a.get('operational_gates',{})
                    for gate,diagnostic in [('first_token_format','format'),('R2_concordance','R2')]:
                        expected=reported.get(gate,{}).get('value');actual=stream['generated_diagnostics'][diagnostic]['fraction']
                        checks[gate]=expected is not None and actual is not None and math.isclose(expected,actual,abs_tol=1e-12)
                    q=stream['q'];expected=reported.get('option_mass_median',{}).get('value')
                    if 'quantiles' in q:checks['option_mass_median']=expected is not None and math.isclose(expected,q['quantiles']['0.5'],abs_tol=1e-12)
                    else:
                        interval=q.get('quantile_intervals',{}).get('0.5')
                        checks['option_mass_median']=None if interval is None or expected is None else interval[0]-.001<=expected<=interval[1]+.001
                    checks['n_items']=stream['n_items']==a.get('n_items')
                    stream['reported_analysis_values_match_raw']=checks
                    if any(v is False for v in checks.values()):stream.update(status='audit_error',error='Selected analysis quality/count differs from raw diagnostics')
                if stream.get('status')=='verified_closed':
                    stream['analysis_sources_verified']=verify_analysis_sources(a,stream,root)
            else:entry['raw_restreamed']=False
            row[stage]=entry
        attempts.append(row)
    current=next((r for r in attempts if r['selected_now']),None)
    selected_suffix='development_E0_conditional_analysis.json' if conditional else 'development_E0_analysis.json'
    if current and (local(state.get('selected_analysis',''),root)!=folder/selected/selected_suffix or
        local(state.get('selected_calibration',''),root)!=folder/selected/'token_calibration.json'):
        raise ValueError('Selected remedy and current analysis/calibration paths disagree')
    verified=bool(current) and all(current[s].get('stream_audit',{}).get('status')=='verified_closed' for s in ('pilot','development'))
    full=bool(current) and current['development'].get('n_items')==100 and current['development'].get('bootstrap_draws')==10000
    return {'status':'selected_measurements_verified' if state.get('status')=='complete' and verified and full else 'selection_or_measurements_incomplete',
        'state_file':str(statepath.relative_to(root)),'state_sha256':sha(statepath),'driver_status':state.get('status'),
        'selection_provenance':'conditional_mass_reassessed' if conditional else 'taskpilot_recovered' if amended else 'legacy_calibration_skip_selection_pending_recovery_or_sync' if skips else 'original_selection_not_affected_by_amendment',
        'calibration_used_for_admission':state.get('calibration_used_for_admission',not amended),
        'original_calibration_only_skips':skips,'selected_remedy':selected,'selected_calibration':state.get('selected_calibration'),
        'selected_analysis':state.get('selected_analysis'),'reported_final_qualification':state.get('final_qualification'),
        'amendment_sha256':state.get('amendment_sha256'),'prior_state_chain':history,'remedies':attempts,
        'initial_queue_status_used_to_select_model':False}


def build(root,models=None,exact_limit=250000):
    root=Path(root).resolve();panel=load(root/'setup/download_queue.json');reports=[]
    old_file=root/'data/core_data_manifest.json';old_ids=set(load(old_file).get('reused20_ids',[])) if old_file.exists() else set()
    for item in panel:
        repo=item['repo_id']
        if models and repo not in models:continue
        folder=root/'results'/repo.replace('/','--');report={'repo_id':repo}
        print('AUDIT '+repo,flush=True)
        try:report['E0']=analyze_e0(folder/'e0',root,exact_limit)
        except OSError as error:report['E0']={'status':'incomplete_local_sync','error':str(error)}
        except (ValueError,KeyError,TypeError) as error:report['E0']={'status':'audit_error','error':str(error)}
        e1=[];lockvalue=report['E0'].get('selected_calibration')
        if lockvalue:
            for sp in sorted((folder/'e1').glob('*R0.jsonl.summary.json')):
                raw=Path(str(sp)[:-len('.summary.json')])
                if raw.name!='holdout200_R0.jsonl':continue
                e1.append(safe_raw(raw,root/'data/e1_holdout200_prompts.jsonl',local(lockvalue,root),root,exact_limit,
                    old_ids if repo in ('LiquidAI/LFM2.5-1.2B-Instruct','Qwen/Qwen3.5-4B') else None))
        report['E1_R0']={'status':'audited_closed_files' if e1 else 'closed_R0_not_synced_or_not_available','files':e1}
        reports.append(report)
    return {'schema':'voices_v3_readout_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Read-only diagnostic audit; current E0 state takes precedence over original queue summaries. No new gates, model selection, effects or row filtering.',
        'exact_quantile_max_rows':exact_limit,'group_quantile_resolution':.001,
        'historical_E0_scope':'All local calibration locks and historical analysis metrics are shown; raw/input streams are rechecked for the CURRENT selected pilot and development only.',
        'model_runtime_or_GPU_executed':False,'source_code_sha256':sha(__file__),'models':reports}


def markdown(report):
    lines=['# Readout audit / 读数质量审计','','仅诊断，不改变模型准入、实验协议或保留的数据；E0 依据当前选择及恢复快照，不依据初始队列旧状态。','',
        '| 模型 | 当前 E0 选择 / 来源 | 选择材料核对 | E1 闭合 R0 行数 | q 中位数 | q<0.5 | q<0.7 |','|---|---|---|---:|---:|---:|---:|']
    for m in report['models']:
        e=m['E0'];files=m['E1_R0']['files'];r=files[0] if files else {};q=r.get('q',{})
        median=q.get('quantiles',{}).get('0.5',q.get('quantile_intervals',{}).get('0.5'))
        values=[m['repo_id'],str(e.get('selected_remedy','—'))+' / '+e.get('selection_provenance','—'),e['status'],
            r.get('n_rows','—'),median if median is not None else '—',q.get('below_fractions',{}).get('0.5','—'),q.get('below_fractions',{}).get('0.7','—')]
        lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in values)+' |')
    lines+=['','q 是完整词表中 A+B 的概率质量。q 较低时，A/B 相对 log-odds 仍可计算，但与不受约束的首 token 回答之间的对应更弱；这些行保留并单独报告。',
        '', 'JSON 包含校准题与真实任务的区别、当前 pilot/full100 质量、补救与旧状态链、锁及原始文件哈希，以及 E1 按人数、标签、正向字母、槽位和人数×标签分组的 q 分布。',
        '', '总体分位数在存储上限内精确计算；超过上限及分组分位数报告宽度 0.001 的直方图区间。没有把未完成或未同步文件当成模型质量失败。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--models',nargs='+');p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--exact-quantile-max-rows',type=int,default=250000)
    a=p.parse_args()
    if a.exact_quantile_max_rows<0:raise ValueError('Quantile memory limit must be nonnegative')
    result=build(a.root,a.models,a.exact_quantile_max_rows);a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for extension,text in [('.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),('.md',markdown(result))]:
        path=Path(str(a.output_prefix)+extension);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(path)
    print(json.dumps({'models':len(result['models']),'report_prefix':str(a.output_prefix),'diagnostic_only':True}))


if __name__=='__main__':main()
