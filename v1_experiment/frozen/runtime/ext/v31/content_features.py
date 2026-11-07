"""Fixed CPU sentence embeddings and same-keypoint features, no outcome access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPO='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
FILES=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
       'vocab.txt','modules.json','1_Pooling/config.json','sentence_bert_config.json']


def build(base,extension):
    base,ext=Path(base),Path(extension);root=ext/'v31';sys.path.insert(0,str(ext))
    from common import load,sha,verify_bindings,write_new
    from huggingface_hub import hf_hub_download
    out=root/'data/M2';target=out/'manifest.json'
    if target.exists():
        result=load(target);verify_bindings(result['source_sha256']);return result
    cache=root/'feature_model';token=Path('/workspace/.hf_home/token').read_text().strip()
    for name in FILES:
        try:hf_hub_download(REPO,name,revision=REVISION,local_dir=cache,token=token)
        except Exception as e:raise RuntimeError('Feature model download failed: '+type(e).__name__) from None
    pooling=load(cache/'1_Pooling/config.json')
    if (not pooling['pooling_mode_mean_tokens'] or pooling['pooling_mode_cls_token'] or
        pooling['pooling_mode_max_tokens'] or pooling['pooling_mode_mean_sqrt_len_tokens']):
        raise ValueError('Official sentence pooling differs')
    modules=load(cache/'modules.json')
    if [m['type'].split('.')[-1] for m in modules]!=['Transformer','Pooling','Normalize']:
        raise ValueError('Unexpected official sentence embedding pipeline')
    model_manifest={'repo':REPO,'revision':REVISION,'files':{n:{'sha256':sha(cache/n),'bytes':(cache/n).stat().st_size} for n in FILES},
                    'device':'cpu','dtype':'float32','pooling':'attention-mask mean then L2 normalize','max_length':256,
                    'weights_restoration':'Exact pinned Hugging Face revision; safetensors SHA256 recorded, weights excluded from Git'}
    write_new(out/'model_manifest.json',model_manifest)
    old=base/'data/e4_agents.json';new=root/'data/e4_agents_new.json';topics=load(old)+load(new)
    arguments={a['argument_id']:a for topic in topics for agent in topic['agents'] for a in agent['arguments'].values()}
    ids=sorted(arguments); texts=[arguments[a]['argument'] for a in ids]
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(cache,local_files_only=True)
    model=AutoModel.from_pretrained(cache,local_files_only=True,attn_implementation='eager').float().cpu().eval()
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(texts),64):
            encoded=tok(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt')
            output=model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
            pooled=(output*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
    vectors=np.concatenate(vectors);similarity=vectors@vectors.T
    if vectors.shape!=(len(ids),384) or not np.isfinite(vectors).all() or not np.allclose(np.diag(similarity),1,atol=1e-5):
        raise ValueError('Invalid sentence embedding output')
    pairs={}
    for topic in topics:
        topic_ids=[a['argument_id'] for agent in topic['agents'] for a in agent['arguments'].values()]
        indices=[ids.index(a) for a in topic_ids]
        kp=[arguments[a]['kp_id'] for a in topic_ids]
        pairs[topic['topic_id']]={'argument_ids':topic_ids,'global_indices':indices,
                                 'same_keypoint':[[a==b for b in kp] for a in kp],
                                 'cosine_similarity':similarity[np.ix_(indices,indices)].tolist()}
    output=out/'embeddings.npz'
    with output.open('xb') as f:np.savez_compressed(f,embeddings=vectors,argument_ids=np.array(ids))
    write_new(out/'pair_features.json',pairs)
    sources={str(p):sha(p) for p in [old,new,Path(__file__),out/'model_manifest.json',output,out/'pair_features.json',root/'data/M1_manifest.json']}
    sources.update({str(cache/n):sha(cache/n) for n in FILES if n!='model.safetensors'})
    result={'schema':'voices_v31_M2_content_features_v1','topics':len(topics),'arguments':len(ids),
            'model':REPO,'revision':REVISION,'model_outputs_read':False,'GPU_used':False,'source_sha256':sources,
            'embedding_scope':'Original six and fourteen new topics; all within-topic argument pairs, independent of focal LLM outcomes'}
    write_new(target,result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--extension',required=True)
    a=p.parse_args();build(a.base,a.extension)
