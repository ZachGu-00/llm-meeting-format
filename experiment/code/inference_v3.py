"""v3 BF16 final-head A/B logits and full-candidate teacher forcing.

No generated numerical probabilities. Calibration uses 20 unrelated messages;
greedy diagnostics independently call native generate for exactly one token.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from loader_v3 import NativeLoader,canonical_hash,read_jsonl

ROOT=Path(__file__).resolve().parents[1]
SINGLE='Answer with a single letter.'
FULL='Answer with the full text of your choice.'
SCHEMA='voices_nonadditivity_v3_readouts_v1'
T2='Give only the requested answer, without any explanation or preamble.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Artifact already exists with different content: '+str(path))
        return
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def normalized_items(path):
    items=read_jsonl(path);seen=set()
    for source in items:
        record_id=source.get('record_id',source.get('id'))
        if not isinstance(record_id,str) or not record_id or record_id in seen:
            raise ValueError('Missing or duplicate id')
        if 'id' in source and source['id']!=record_id:raise ValueError('id/record_id disagree')
        source['record_id']=record_id;seen.add(record_id)
        if not isinstance(source.get('messages'),list) or not source['messages']:
            raise ValueError('messages required')
        if source.get('positive_letter') not in {'A','B'}:raise ValueError('positive_letter must be A/B')
        if 'prompt_hash' in source and source['prompt_hash']!=canonical_hash(source['messages']):
            raise ValueError('prompt_hash differs from exact messages')
        if not isinstance(source.get('candidates'),dict) or set(source['candidates'])!={'A','B'}:
            raise ValueError('Full candidates mapping A/B required')
        if any(not isinstance(x,str) or not x for x in source['candidates'].values()):
            raise ValueError('Full candidate texts must be nonempty strings')
    return items


def r1_messages(messages):
    result=[dict(x) for x in messages]
    if result[-1]['role']!='user' or not result[-1]['content'].rstrip().endswith(SINGLE):
        raise ValueError('R1 requires the exact final single-letter instruction')
    text=result[-1]['content'];end=len(text.rstrip());start=end-len(SINGLE)
    result[-1]['content']=text[:start]+FULL+text[end:]
    return result


def normalize_native_ids(value):
    if isinstance(value,dict) or hasattr(value,'keys'):
        value=value['input_ids']
    if hasattr(value,'tolist'):value=value.tolist()
    if len(value)==1 and isinstance(value[0],list):value=value[0]
    if not isinstance(value,list) or not value or any(not isinstance(x,int) for x in value):
        raise ValueError('Unexpected tokenizer apply_chat_template return type')
    return value


def render_prompt(tokenizer,messages,model_kind,remedy,readout,max_length):
    """Render exact experimental context; '(' guides format, never masks logits."""
    if remedy not in {'original','t2','prefill'} or readout not in {'R0','R1'}:
        raise ValueError('Unknown remedy or readout')
    transformed=r1_messages(messages) if readout=='R1' else [dict(m) for m in messages]
    if remedy!='original':
        answer=FULL if readout=='R1' else SINGLE
        last=transformed[-1]
        if last['role']!='user' or not last['content'].rstrip().endswith(answer):
            raise ValueError('Remedy requires a final exact answer instruction')
        end=len(last['content'].rstrip());start=end-len(answer)
        last['content']=last['content'][:start]+T2+'\n'+answer+last['content'][end:]
    if model_kind=='chat':
        text=tokenizer.apply_chat_template(transformed,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        native=normalize_native_ids(tokenizer.apply_chat_template(transformed,tokenize=True,
            return_dict=False,add_generation_prompt=True,enable_thinking=False))
        if tokenizer.encode(text,add_special_tokens=False)!=native:
            raise ValueError('Official template text/token rendering disagrees')
        suffix=text[-200:]
        if (suffix.rfind('<think>')>suffix.rfind('</think>') or
            suffix.rstrip().endswith(('<|channel>thought','[THINK]'))):
            raise ValueError('Official generation prefix still opens thinking')
    elif model_kind=='base':
        if any(m['role'] not in {'system','user'} for m in transformed):
            raise ValueError('Plain base template supports only system/user input')
        text=(tokenizer.bos_token or '')+'\n\n'.join(m['content'] for m in transformed)+'\n\nAnswer:'
    else:
        raise ValueError('Unknown model kind')
    # R1 is a full-sentence format probe. Its candidates do not begin with an
    # option letter, so the letter-specific '(' prefill is intentionally absent.
    prefix='(' if remedy=='prefill' and readout=='R0' else ''
    text+=prefix
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>max_length:
        raise ValueError('Empty prompt or max_length exceeded; no truncation')
    return {'text':text,'ids':ids,'sha256':hashlib.sha256(text.encode()).hexdigest(),
        'ids_sha256':canonical_hash(ids),'messages_sha256':canonical_hash(transformed),
        'assistant_prefill':prefix,'remedy':remedy,'readout':readout}


def selected_ids(items,n):
    return {r['record_id'] for r in sorted(items,key=lambda r:(hashlib.sha256(
        r['record_id'].encode()).hexdigest(),r['record_id']))[:min(n,len(items))]}


class Engine:
    def __init__(self,model_dir,repo_id,revision,max_length=8192,remedy='original',
                 model_kind='auto',attn_implementation='eager'):
        self.remedy=remedy
        self.loader=NativeLoader(model_dir,repo_id,revision,'bfloat16','cuda:0',
                                 attn_implementation,max_length,model_kind=model_kind)
        self.torch=self.loader.torch;self.tok=self.loader.tokenizer;self.model=self.loader.model
        self.identity={'schema':SCHEMA,'code_sha256':sha(__file__),
            'native_loader_sha256':sha(Path(__file__).with_name('loader_v3.py')),
            'parent_loader':self.loader.identity,'precision':'native_bfloat16',
            'enable_thinking':False,'template_policy':'official_native_chat_or_explicit_plain_base',
            'model_kind':self.loader.model_kind,'remedy':remedy,
            'remedy_instruction':T2 if remedy!='original' else None,
            'assistant_prefill_R0':'(' if remedy=='prefill' else '',
            'assistant_prefill_R1':'',
            'R1_candidate_separator':' ' if self.loader.model_kind=='base' else '',
            'vocabulary_constraint':'none_full_vocabulary_logits',
            'R0':'actual_final_output_logits_zA_minus_zB; q=full_vocabulary_pA_plus_pB',
            'R1':'sum_full_continuation_logprob_A_minus_B_no_eos_no_length_normalization',
            'R1_instruction_replacement':{'from':SINGLE,'to':FULL},
            'softcap':'preserve_actual_model_output','cross_prompt_cache_reuse':False,
            'greedy_diagnostic':'independent_native_generate_max_new_tokens_1_no_sampling'}
        self.variants={}
        for letter in 'AB':
            variants=[];seen=set()
            for text in [letter,' '+letter]:
                ids=self.tok.encode(text,add_special_tokens=False)
                if len(ids)==1 and ids[0] not in seen:
                    variants.append({'text':text,'token_id':ids[0]});seen.add(ids[0])
            if not variants:raise ValueError('No one-token variant for '+letter)
            self.variants[letter]=variants
        if {x['token_id'] for x in self.variants['A']}&{x['token_id'] for x in self.variants['B']}:
            raise ValueError('A/B variant token sets overlap')

    def render(self,messages,readout='R0'):
        return render_prompt(self.tok,messages,self.loader.model_kind,self.remedy,
                             readout,self.loader.max_length)

    def tensors(self,sequences):
        torch=self.torch;width=max(map(len,sequences))
        ids=torch.full((len(sequences),width),self.tok.pad_token_id,dtype=torch.long,device='cuda:0')
        mask=torch.zeros_like(ids)
        for i,s in enumerate(sequences):ids[i,-len(s):]=torch.tensor(s,device='cuda:0');mask[i,-len(s):]=1
        position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
        return {'input_ids':ids,'attention_mask':mask,'position_ids':position}

    def final_logits(self,sequences):
        torch=self.torch;kw=self.tensors(sequences);kw['use_cache']=False
        if self.loader.keep_logits:kw['logits_to_keep']=1
        with torch.inference_mode():z=self.model(**kw).logits[:,-1,:].float()
        if not bool(torch.isfinite(z).all()):raise ValueError('Nonfinite final logits')
        return z,torch.log_softmax(z,dim=-1)

    def letter_for_id(self,token_id,option_ids=None):
        if option_ids is not None:return next((k for k,v in option_ids.items() if token_id==v),None)
        return next((k for k,vs in self.variants.items() if token_id in {v['token_id'] for v in vs}),None)

    def calibrate(self,items,batch_size):
        if len(items)!=20:raise ValueError('Exactly 20 unrelated token-calibration prompts required')
        prepared=[self.render(r['messages']) for r in items];observations=[]
        for start in range(0,20,batch_size):
            ps=prepared[start:start+batch_size];z,lp=self.final_logits([p['ids'] for p in ps])
            for j,p in enumerate(ps):
                observations.append({'record_id':items[start+j]['record_id'],
                    'actual_prompt_sha256':p['sha256'],'prompt_ids_sha256':p['ids_sha256'],
                    'variant_logprobs':{k:[float(lp[j,v['token_id']]) for v in self.variants[k]] for k in 'AB'},
                    'variant_logits':{k:[float(z[j,v['token_id']]) for v in self.variants[k]] for k in 'AB'},
                    'greedy_argmax_id':int(z[j].argmax()),'batch_size':len(ps)})
        selected={};details={}
        for letter in 'AB':
            means=[statistics.mean(math.exp(r['variant_logprobs'][letter][j]) for r in observations)
                for j in range(len(self.variants[letter]))]
            best=max(range(len(means)),key=lambda j:means[j])
            selected[letter]=self.variants[letter][best]['token_id']
            details[letter]={'variants':self.variants[letter],'mean_full_vocab_probabilities':means,
                'selected_index':best,'selected_text':self.variants[letter][best]['text'],'selected_token_id':selected[letter]}
        qs=[sum(math.exp(r['variant_logprobs'][k][details[k]['selected_index']]) for k in 'AB') for r in observations]
        audit=[]
        audit_ids=selected_ids(items,8)
        for item,p,prior in zip(items,prepared,observations):
            if item['record_id'] not in audit_ids:continue
            z,lp=self.final_logits([p['ids']]);a,b=selected['A'],selected['B']
            previous_y=prior['variant_logits']['A'][details['A']['selected_index']]-prior['variant_logits']['B'][details['B']['selected_index']]
            single_y=float(z[0,a])-float(z[0,b]);single_q=math.exp(float(lp[0,a]))+math.exp(float(lp[0,b]))
            previous_q=sum(math.exp(prior['variant_logprobs'][k][details[k]['selected_index']]) for k in 'AB')
            audit.append({'record_id':item['record_id'],'calibration_batch_size':prior['batch_size'],
                'batch1_y_AB':single_y,'calibration_batch_y_AB':previous_y,'abs_y_difference':abs(single_y-previous_y),
                'batch1_q':single_q,'calibration_batch_q':previous_q,'abs_q_difference':abs(single_q-previous_q),
                'batch1_argmax_id':int(z[0].argmax()),'calibration_batch_argmax_id':prior['greedy_argmax_id']})
        return {'schema':'voices_v3_token_calibration_v1','engine':self.identity,
            'option_ids':selected,'selection':details,'tie_rule':'first_variant_literal_precedes_space',
            'n':20,'batch_size':batch_size,'input_rows_sha256':canonical_hash(items),
            'observations':observations,'median_q':statistics.median(qs),'q_per_prompt':qs,
            'greedy_any_variant_fraction':statistics.mean(self.letter_for_id(r['greedy_argmax_id']) is not None for r in observations),
            'batch_audit':{'n':len(audit),'selection':'first8_by_record_id_SHA256','rows':audit,
                'max_abs_y_difference':max(x['abs_y_difference'] for x in audit),
                'max_abs_q_difference':max(x['abs_q_difference'] for x in audit),
                'argmax_agreement_fraction':statistics.mean(x['batch1_argmax_id']==x['calibration_batch_argmax_id'] for x in audit),
                'fixed_automatic_pass_threshold':None,'interpretation':'BF16 batching diagnostic only; no historical FP32 threshold applied'},
            'used_development_or_confirmation_for_selection':False}

    def validate_calibration(self,lock):
        if lock['schema']!='voices_v3_token_calibration_v1' or lock['engine']!=self.identity or lock['n']!=20:
            raise ValueError('Calibration engine/scope differs')
        if lock.get('lock_content_sha256')!=canonical_hash({k:v for k,v in lock.items() if k!='lock_content_sha256'}):
            raise ValueError('Calibration lock content changed')
        for k in 'AB':
            if lock['selection'][k]['variants']!=self.variants[k] or lock['option_ids'][k] not in [v['token_id'] for v in self.variants[k]]:
                raise ValueError('Calibration option token mapping differs')
        return lock['option_ids']

    def r0(self,items,options,batch_size):
        prepared=[self.render(r['messages']) for r in items];result=[]
        for start in range(0,len(items),batch_size):
            ps=prepared[start:start+batch_size];z,lp=self.final_logits([p['ids'] for p in ps])
            for j,p in enumerate(ps):
                a,b=options['A'],options['B'];za,zb=float(z[j,a]),float(z[j,b]);la,lb=float(lp[j,a]),float(lp[j,b])
                q=math.exp(la)+math.exp(lb);greedy=int(z[j].argmax())
                if not math.isfinite(q) or not 0<=q<=1+1e-6:raise ValueError('Invalid full-vocabulary option mass')
                result.append({'R0_y_AB':za-zb,'y_AB':za-zb,'q':q,'logit_A':za,'logit_B':zb,
                    'logprob_A':la,'logprob_B':lb,'greedy_argmax_id':greedy,
                    'greedy_argmax_token':self.tok.decode([greedy],skip_special_tokens=False),
                    'greedy_argmax_selected_letter':self.letter_for_id(greedy,options),
                    'greedy_argmax_any_variant_letter':self.letter_for_id(greedy),
                    'actual_prompt_sha256':p['sha256'],'actual_prompt_ids_sha256':p['ids_sha256'],
                    'prompt_token_count':len(p['ids']),'actual_batch_size_R0':len(ps),
                    'q_below_point5':q<.5})
        return result

    def r1(self,items,batch_size):
        flattened=[];results=[{} for _ in items]
        for i,item in enumerate(items):
            p=self.render(item['messages'],readout='R1')
            results[i].update(R1_actual_prompt_sha256=p['sha256'],R1_actual_prompt_ids_sha256=p['ids_sha256'])
            for letter in 'AB':
                text=item['candidates'][letter]
                separator=' ' if self.loader.model_kind=='base' else ''
                joint=self.tok.encode(p['text']+separator+text,add_special_tokens=False)
                if joint[:len(p['ids'])]!=p['ids'] or len(joint)<=len(p['ids']):
                    raise ValueError('R1 prompt/continuation boundary changed: '+item['record_id']+' '+letter)
                if len(joint)>self.loader.max_length:raise ValueError('R1 exceeds max_length')
                continuation=joint[len(p['ids']):]
                flattened.append((i,letter,p['ids'],continuation))
        for start in range(0,len(flattened),batch_size):
            batch=flattened[start:start+batch_size];keep=max(len(c) for _,_,_,c in batch)
            lp=self.loader.forward_tail([p+c[:-1] for _,_,p,c in batch],keep)
            for j,(i,letter,p,c) in enumerate(batch):
                token_lp=[float(lp[j,keep-len(c)+k,t]) for k,t in enumerate(c)]
                total=sum(token_lp)
                if not all(math.isfinite(x) and x<=1e-5 for x in token_lp):raise ValueError('Invalid R1 token logprob')
                results[i]['R1_logprob_'+letter]=total
                results[i]['R1_candidate_'+letter]={'text':items[i]['candidates'][letter],
                    'scored_leading_separator':' ' if self.loader.model_kind=='base' else '',
                    'token_ids':c,'token_logprobs':token_lp,'sum_logprob':total,'actual_batch_size':len(batch)}
        for row in results:row['R1_y_AB']=row['R1_logprob_A']-row['R1_logprob_B']
        return results

    def greedy(self,items,options,batch_size):
        result=[];torch=self.torch
        overrides={'do_sample':False,'num_beams':1,'num_return_sequences':1,'max_new_tokens':1,
            'min_new_tokens':0,'min_length':0,'use_cache':True,'return_dict_in_generate':True,
            'output_scores':False,'pad_token_id':self.tok.pad_token_id,'repetition_penalty':1.0,
            'no_repeat_ngram_size':0,'forced_bos_token_id':None,'forced_eos_token_id':None,
            'bad_words_ids':None,'suppress_tokens':None,'begin_suppress_tokens':None}
        for start in range(0,len(items),batch_size):
            batch=items[start:start+batch_size];prepared=[self.render(r['messages']) for r in batch]
            enc=self.tensors([p['ids'] for p in prepared]);width=enc['input_ids'].shape[1]
            with torch.inference_mode():out=self.model.generate(**enc,**overrides).sequences
            if out.shape!=(len(batch),width+1) or not torch.equal(out[:,:width],enc['input_ids']):
                raise ValueError('Greedy diagnostic did not return exactly one native token per prompt')
            for j in range(len(batch)):
                token_id=int(out[j,-1]);result.append({'generated_first_token_id':token_id,
                    'generated_first_token_text':self.tok.decode([token_id],skip_special_tokens=False),
                    'generated_first_token_letter':self.letter_for_id(token_id),
                    'generated_first_token_selected_letter':self.letter_for_id(token_id,options),
                    'actual_batch_size_generate':len(batch),'generation_overrides':overrides,
                    'diagnostic_is_actual_generate_call':True})
        return result


def run_score(engine,items,args,lock):
    option_ids=engine.validate_calibration(lock)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    format_ids=selected_ids(items,args.format_n) if args.readouts!='R1' else set()
    r2_ids=selected_ids(items,args.r2_n) if args.readouts!='R1' else set()
    identity={'schema':SCHEMA,'engine':engine.identity,'mode':'score','readout':args.readouts,
        'input_file_sha256':sha(args.input),'input_rows_sha256':canonical_hash(items),'n_input':len(items),
        'input_record_ids_sha256':canonical_hash([r['record_id'] for r in items]),
        'calibration_file_sha256':sha(args.calibration),'option_ids':option_ids,'batch_size':args.batch_size,
        'format_record_ids':sorted(format_ids),'r2_record_ids':sorted(r2_ids)}
    runpath=Path(str(output)+'.run.json');summarypath=Path(str(output)+'.summary.json')
    if output.exists() and not runpath.exists():raise ValueError('Existing scores lack bound run identity')
    write_new(runpath,identity)
    previous=read_jsonl(output) if output.exists() else []
    if len(previous)>len(items) or [r['record_id'] for r in previous]!=[r['record_id'] for r in items[:len(previous)]]:
        raise ValueError('Resume rows must match the fixed exact input prefix')
    runhash=sha(runpath)
    for item,row in zip(items,previous):
        if row.get('run_file_sha256')!=runhash or any(row.get(k)!=v for k,v in item.items()):
            raise ValueError('Resume row identity or source changed')
    if summarypath.exists():
        summary=json.loads(summarypath.read_text())
        if summary.get('status')!='complete' or len(previous)!=len(items) or summary['output_sha256']!=sha(output):
            raise ValueError('Closed output no longer matches summary')
        return summary
    start_time=time.perf_counter()
    with output.open('a',encoding='utf-8',newline='\n') as handle:
        for start in range(len(previous),len(items),args.batch_size):
            batch=items[start:start+args.batch_size];results=[{} for _ in batch]
            if args.readouts in {'R0','R0_R1'}:
                for dest,row in zip(results,engine.r0(batch,option_ids,args.batch_size)):dest.update(row)
                diag=[item for item in batch if item['record_id'] in format_ids|r2_ids]
                generated=engine.greedy(diag,option_ids,args.batch_size) if diag else []
                mapping={item['record_id']:row for item,row in zip(diag,generated)}
                for item,dest in zip(batch,results):
                    dest.update(mapping.get(item['record_id'],{}))
                    dest.update(diagnostic_format_selected=item['record_id'] in format_ids,
                        diagnostic_r2_selected=item['record_id'] in r2_ids)
            if args.readouts in {'R1','R0_R1'}:
                for dest,row in zip(results,engine.r1(batch,args.batch_size)):dest.update(row)
            for item,measured in zip(batch,results):
                row={**item,**measured,'readout':args.readouts,'run_file_sha256':runhash,
                    'status':'ok','option_ids':option_ids,'batch_size':args.batch_size}
                handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
                previous.append(row)
            handle.flush();os.fsync(handle.fileno())
            if start//args.progress_every != len(previous)//args.progress_every or len(previous)==len(items):
                print(json.dumps({'completed':len(previous),'total':len(items),'readout':args.readouts}),flush=True)
    if [r['record_id'] for r in previous]!=[r['record_id'] for r in items]:raise ValueError('Completion ID mismatch')
    summary={**identity,'status':'complete','n_output':len(previous),'exact_id_and_count_equality':True,
        'output_sha256':sha(output),'output_bytes':output.stat().st_size,'run_file_sha256':runhash,
        'seconds_this_invocation':time.perf_counter()-start_time}
    write_new(summarypath,summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['calibrate','score'],required=True)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--model-dir',required=True);parser.add_argument('--repo-id',required=True)
    parser.add_argument('--revision',required=True);parser.add_argument('--calibration')
    parser.add_argument('--readouts',choices=['R0','R1','R0_R1'],default='R0')
    parser.add_argument('--batch-size',type=int,default=1);parser.add_argument('--max-length',type=int,default=8192)
    parser.add_argument('--format-n',type=int,default=50);parser.add_argument('--r2-n',type=int,default=200)
    parser.add_argument('--remedy',choices=['original','t2','prefill'],default='original')
    parser.add_argument('--model-kind',choices=['auto','chat','base'],default='auto')
    parser.add_argument('--attn-implementation',choices=['eager','sdpa','flash_attention_2'],default='eager')
    parser.add_argument('--progress-every',type=int,default=100)
    args=parser.parse_args()
    if args.batch_size<1 or args.format_n<0 or args.r2_n<0 or args.progress_every<1:raise ValueError('Invalid batch/diagnostic count')
    items=normalized_items(args.input)
    if not items:raise ValueError('Empty input')
    if args.mode=='calibrate' and Path(args.output).exists():raise ValueError('Calibration already frozen; do not overwrite')
    if args.mode=='score' and not args.calibration:raise ValueError('--calibration required')
    engine=Engine(args.model_dir,args.repo_id,args.revision,args.max_length,args.remedy,
                  args.model_kind,args.attn_implementation)
    if args.mode=='calibrate':
        lock=engine.calibrate(items,args.batch_size);lock['input_file_sha256']=sha(args.input)
        lock['lock_content_sha256']=canonical_hash(lock);write_new(args.output,lock)
        print(json.dumps({'status':'calibrated','repo_id':args.repo_id,'median_q':lock['median_q'],
            'greedy_any_variant_fraction':lock['greedy_any_variant_fraction'],'option_ids':lock['option_ids']}));return
    lock=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    print(json.dumps(run_score(engine,items,args,lock),ensure_ascii=False))


if __name__=='__main__':main()
