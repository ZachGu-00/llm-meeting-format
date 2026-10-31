"""CPU tests of prompt interventions and exact continuation alignment."""
import json
import functools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from inference_v3 import Engine, FULL, SINGLE, T2, render_prompt
from loader_v3 import snapshot_identity, file_hash, native_fp32_modules, canonical_hash, kernel_dispatch_identity


class TinyTokenizer:
    bos_token = '<bos>'
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert kwargs['enable_thinking'] is False
        text=''.join('<'+m['role']+'>'+m['content'] for m in messages)+'<assistant><think></think>\n'
        return self.encode(text) if tokenize else text


MESSAGES=[{'role':'system','content':'Read the comments.'},
          {'role':'user','content':'Your choice?\n(A) Yes\n(B) No\n'+SINGLE}]


class PromptAndScoringTests(unittest.TestCase):
    def test_kernel_dispatch_reads_actual_closure_not_wraps_metadata(self):
        def reference(*args, **kwargs): pass
        def accelerated(*args, **kwargs): pass
        accelerated.__module__ = 'fla.ops.gated_delta_rule.chunk'
        accelerated.__qualname__ = 'chunk_gated_delta_rule'
        def wrapper(implementation, is_new_implementation):
            @functools.wraps(reference)
            def wrapped(*args, **kwargs):
                if is_new_implementation:
                    return implementation(*args, **kwargs)
                return implementation(*args, **kwargs)
            return wrapped
        fn = wrapper(accelerated, True)
        self.assertEqual(fn.__module__, reference.__module__)
        result = kernel_dispatch_identity(fn)
        self.assertEqual(result['implementation_module'], 'fla.ops.gated_delta_rule.chunk')
        self.assertEqual(result['implementation_qualname'], 'chunk_gated_delta_rule')
        self.assertEqual(result['selection'], 'installed_package')
        self.assertEqual(result, kernel_dispatch_identity(wrapper(accelerated, True)))
        self.assertNotIn('0x', json.dumps(result))
        self.assertEqual(kernel_dispatch_identity(wrapper(reference, False))['selection'], 'torch_reference')

    def test_native_model_fp32_set_is_deterministic_json_metadata(self):
        model=SimpleNamespace(_keep_in_fp32_modules={'dt_bias','A_log'})
        value=native_fp32_modules(model)
        self.assertEqual(value,['A_log','dt_bias'])
        self.assertEqual(json.loads(json.dumps(value)),value)
        self.assertEqual(canonical_hash({'modules':value}),canonical_hash({'modules':['A_log','dt_bias']}))
        self.assertIsNone(native_fp32_modules(SimpleNamespace()))

    def test_remedy_changes_format_context_without_forcing_option(self):
        tok=TinyTokenizer()
        original=render_prompt(tok,MESSAGES,'chat','original','R0',8192)
        self.assertEqual(original['text'],tok.apply_chat_template(MESSAGES,enable_thinking=False))
        t2=render_prompt(tok,MESSAGES,'chat','t2','R0',8192)
        prefix=render_prompt(tok,MESSAGES,'chat','prefill','R0',8192)
        self.assertIn(T2+'\n'+SINGLE,t2['text'])
        self.assertEqual(prefix['text'],t2['text']+'(')
        self.assertEqual(prefix['ids'],tok.encode(prefix['text']))
        self.assertEqual(prefix['assistant_prefill'],'(')
        self.assertEqual(MESSAGES[-1]['content'].splitlines()[-1],SINGLE)

    def test_full_sentence_probe_has_no_letter_specific_prefix(self):
        r=render_prompt(TinyTokenizer(),MESSAGES,'chat','prefill','R1',8192)
        self.assertIn(T2+'\n'+FULL,r['text'])
        self.assertNotIn(SINGLE,r['text'])
        self.assertEqual(r['assistant_prefill'],'')
        self.assertTrue(r['text'].endswith('</think>\n'))

    def test_base_prompt_and_no_silent_truncation(self):
        r=render_prompt(TinyTokenizer(),MESSAGES,'base','original','R0',8192)
        self.assertTrue(r['text'].startswith('<bos>'))
        self.assertTrue(r['text'].endswith('Answer:'))
        with self.assertRaisesRegex(ValueError,'max_length'):
            render_prompt(TinyTokenizer(),MESSAGES,'base','original','R0',5)

    def test_open_thinking_prefix_is_detected_closed_empty_block_is_valid(self):
        class ThinkingTokenizer(TinyTokenizer):
            def apply_chat_template(self,messages,tokenize=False,**kwargs):
                text='<assistant><think>\n'
                return self.encode(text) if tokenize else text
        with self.assertRaisesRegex(ValueError,'thinking'):
            render_prompt(ThinkingTokenizer(),MESSAGES,'chat','original','R0',8192)
        self.assertIn('</think>',render_prompt(TinyTokenizer(),MESSAGES,'chat','original','R0',8192)['text'])

    def test_teacher_forcing_all_tokens_aligned_across_batch_sizes(self):
        engine=Engine.__new__(Engine);engine.tok=TinyTokenizer();engine.remedy='prefill'
        def tail(sequences,keep):
            out=np.empty((len(sequences),keep,256))
            for i,s in enumerate(sequences):
                for k in range(keep):
                    out[i,k,:]=-(len(s)-keep+k+1)/100-np.arange(256)/1000
            return out
        engine.loader=SimpleNamespace(model_kind='chat',max_length=8192,forward_tail=tail)
        items=[{'record_id':'one','messages':MESSAGES,'candidates':{'A':'Yes','B':'No'}},
               {'record_id':'two','messages':MESSAGES,'candidates':{'A':'Absolutely','B':'Never'}}]
        single=engine.r1(items,1);batched=engine.r1(items,4)
        for item,a,b in zip(items,single,batched):
            p=engine.render(item['messages'],'R1')
            for letter in 'AB':
                ids=engine.tok.encode(item['candidates'][letter])
                expected=sum(-(len(p['ids'])+k)/100-token/1000 for k,token in enumerate(ids))
                self.assertAlmostEqual(a['R1_logprob_'+letter],expected)
                self.assertAlmostEqual(a['R1_logprob_'+letter],b['R1_logprob_'+letter])
                self.assertEqual(len(a['R1_candidate_'+letter]['token_logprobs']),len(ids))

    def test_manifest_rejects_metadata_drift_and_extra_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config.json').write_text('{}')
            (root/'model.safetensors').write_bytes(b'test-placeholder')
            (root/'RESEARCH_REVISION.txt').write_text('abc')
            files={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in root.iterdir()}
            (root/'snapshot_manifest.json').write_text(json.dumps({'repo_id':'test/model','revision':'abc','files':files}))
            snapshot_identity(root,'test/model','abc')
            (root/'config.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'metadata changed'):
                snapshot_identity(root,'test/model','abc')
            (root/'config.json').write_text('{}')
            (root/'extra.safetensors').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'unmanifested'):
                snapshot_identity(root,'test/model','abc')


if __name__=='__main__':unittest.main()
