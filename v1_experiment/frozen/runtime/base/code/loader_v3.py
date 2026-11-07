"""Native offline BF16 loader for v3. Downloads are bound by a local manifest."""
from __future__ import annotations
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import sys


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def native_fp32_modules(model):
    value=getattr(model,'_keep_in_fp32_modules',None)
    return None if value is None else sorted(map(str,[value] if isinstance(value,str) else value))


def kernel_dispatch_identity(function):
    """Resolve the native HF optional-package closure, not functools.wraps' copied name."""
    current, seen = function, set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        captured = inspect.getclosurevars(current).nonlocals if inspect.isfunction(current) else {}
        implementation = captured.get('implementation')
        if callable(implementation) and 'is_new_implementation' in captured:
            return {'selection': 'installed_package' if captured['is_new_implementation'] else 'torch_reference',
                'implementation_module': getattr(implementation, '__module__', None),
                'implementation_qualname': getattr(implementation, '__qualname__',
                    getattr(implementation, '__name__', None)),
                'identity_source': 'native_optional_package_closure'}
        current = getattr(current, '__wrapped__', None)
    return {'selection': 'direct_callable_or_unrecognized_wrapper',
        'implementation_module': getattr(function, '__module__', None),
        'implementation_qualname': getattr(function, '__qualname__', getattr(function, '__name__', None)),
        'identity_source': 'callable_metadata_only'}


def native_kernel_identity(model):
    distributions = {}
    for name in ('causal-conv1d', 'flash-linear-attention', 'fla-core', 'triton'):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    dispatch = {}
    names = ('causal_conv1d_fn', 'causal_conv1d_update', 'torch_chunk_gated_delta_rule',
             'torch_recurrent_gated_delta_rule', 'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule')
    for module_name in sorted({type(layer).__module__ for layer in model.modules()}):
        module = sys.modules.get(module_name)
        for name in names:
            function = getattr(module, name, None)
            if callable(function):
                dispatch[module_name + '.' + name] = kernel_dispatch_identity(function)
    return {'optional_kernel_distributions': distributions,
        'native_kernel_dispatch': dispatch,
        'native_kernel_dispatch_scope': 'selected_callable_for_non_export_native_forward; not a profiler trace',
        'hub_kernels_requested': bool(getattr(model, '_use_kernels', False))}


def snapshot_identity(model_dir, repo_id, revision):
    root = Path(model_dir).resolve()
    marker = root / 'RESEARCH_REVISION.txt'
    manifest_path = root / 'snapshot_manifest.json'
    if not marker.is_file() or marker.read_text().strip() != revision:
        raise ValueError('Local revision marker missing or mismatched')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('repo_id') != repo_id or manifest.get('revision') != revision:
        raise ValueError('Snapshot manifest identifies a different model/revision')
    weights, metadata = {}, {}
    for name, entry in manifest['files'].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size != entry['bytes']:
            raise ValueError('Invalid or incomplete snapshot file: ' + name)
        if not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64:
            raise ValueError('Snapshot file has no SHA256: ' + name)
        if p.suffix == '.safetensors':
            weights[name] = entry
        else:
            digest = file_hash(p)
            if digest != entry['sha256']:
                raise ValueError('Snapshot metadata changed: ' + name)
            metadata[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.safetensors')}
    if actual != set(weights) or not weights or 'config.json' not in metadata:
        raise ValueError('Missing or unmanifested snapshot weights/config')
    return {'snapshot_manifest_sha256': file_hash(manifest_path),
        'weight_manifest_sha256': canonical_hash(weights),
        'weight_verification': 'sizes_checked_at_load_sha256_recorded_by_downloader',
        'metadata_sha256': metadata}


class NativeLoader:
    def __init__(self, model_dir, repo_id, revision, dtype='bfloat16', device='cuda:0',
                 attn_implementation='eager', max_length=8192, model_kind='auto', tokenizer_only=False):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        import transformers
        self.model_dir = str(Path(model_dir).resolve())
        self.max_length, self.device = max_length, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer needs a PAD or EOS token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            template = self.tokenizer.get_chat_template()
        except ValueError:
            template = None
        self.model_kind = ('chat' if template else 'base') if model_kind == 'auto' else model_kind
        if self.model_kind == 'chat' and not template:
            raise ValueError('Chat checkpoint has no official chat template')
        self.identity = {'repo_id': repo_id, 'revision': revision, 'dtype': dtype,
            'transformers_version': transformers.__version__, 'model_kind': self.model_kind,
            'attn_implementation': attn_implementation, 'code_sha256': file_hash(__file__),
            'template_sha256': hashlib.sha256(template.encode()).hexdigest() if template else None,
            'max_length': max_length, 'random_seed': 20261030,
            **snapshot_identity(self.model_dir, repo_id, revision)}
        self.model = None
        if tokenizer_only:
            return
        if not os.environ.get('STY'):
            raise RuntimeError('Model inference must run inside GNU screen (STY)')
        import torch
        self.torch = torch
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261030)
        config = transformers.AutoConfig.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False)
        if getattr(config, 'quantization_config', None):
            raise ValueError('v3 requires original BF16, not a quantized checkpoint')
        architecture = (getattr(config, 'architectures', None) or [None])[0]
        model_class = getattr(transformers, architecture, None) if architecture else None
        if model_class is None or not architecture.endswith(('ForCausalLM', 'ForConditionalGeneration')):
            raise ValueError('No supported native generative architecture: ' + str(architecture))
        self.model, loading = model_class.from_pretrained(self.model_dir,
            local_files_only=True, trust_remote_code=False, dtype=getattr(torch, dtype),
            device_map={'': device}, attn_implementation=attn_implementation,
            output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise RuntimeError('Incomplete checkpoint loading: ' + str(loading))
        self.model.eval()
        signature = inspect.signature(self.model.forward)
        self.keep_logits = 'logits_to_keep' in signature.parameters
        parameter_dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        fp32_names = [name for name,p in self.model.named_parameters() if p.dtype == torch.float32]
        if ('torch.' + dtype not in parameter_dtypes or
            not set(parameter_dtypes) <= {'torch.' + dtype, 'torch.float32'}):
            raise RuntimeError('Unexpected parameter dtype mix: ' + str(parameter_dtypes))
        self.identity.update({'architecture': architecture, 'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
            'parameter_dtypes': parameter_dtypes,
            'native_fp32_parameter_names': fp32_names,
            'native_keep_in_fp32_modules': native_fp32_modules(self.model),
            'parameter_count_loaded': sum(p.numel() for p in self.model.parameters()),
            'logits_to_keep_supported': self.keep_logits,
            'unexpected_checkpoint_keys': sorted(map(str, loading.get('unexpected_keys', []))),
            **native_kernel_identity(self.model)})

    def forward_tail(self, sequences, keep):
        torch = self.torch
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(sequences):
            ids[i, -len(s):] = torch.tensor(s, device=self.device)
            mask[i, -len(s):] = 1
        position = mask.cumsum(-1) - 1
        position.masked_fill_(mask == 0, 0)
        kwargs = {'input_ids': ids, 'attention_mask': mask, 'position_ids': position, 'use_cache': False}
        if self.keep_logits:
            kwargs['logits_to_keep'] = keep
        with torch.inference_mode():
            logits = self.model(**kwargs).logits[:, -keep:, :].float()
            return torch.log_softmax(logits, dim=-1)
