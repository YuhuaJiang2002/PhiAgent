"""Process-local pinned Sol + FirstBlockCache integration, including spawned ranks."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from .contracts import digest, read_json

REVISION='6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb'
ENVIRONMENT={
    'H3_SOL_ATTN':'1','H3_FIRSTBLOCKCACHE':'1','H3_EASYCACHE':'0',
    'H3_POLICY_NAME':'fullopt_exact','H3_SOL_TAU':'1.0','H3_SOL_THRESH_TYPE':'exact',
    'H3_SOL_DENSE_STEPS':'10','H3_SOL_DENSE_LAYERS':'2','H3_SOL_SINK_MODE':'prefix',
    'H3_SOL_DENSITY_MODE':'all','H3_SOL_CORRECTNESS_GATE':'1','H3_CACHE_THRESHOLD':'0.08',
    'H3_EXPECTED_SOL_BACKEND':'triton','SOL_ATTN_STRICT':'1',
    'H3_SOL_GATE_MAX_ABS':'.15','H3_SOL_GATE_MEAN_ABS':'.002','H3_SOL_GATE_REL_L2':'.005',
}
_installed=False


def install():
    global _installed
    if _installed:return
    source=Path(os.environ['PHI_EGO_SOL_ROOT'])
    manifest=read_json(source/'phiagent_source_manifest.json')
    if manifest.get('revision')!=REVISION or not manifest.get('sha256'):
        raise RuntimeError('Use the pinned external Sol source preparer')
    for name,expected in manifest['sha256'].items():
        if digest(source/name)!=expected:raise RuntimeError(f'Modified Sol source {name}')
    sys.path[:0]=[str(source),str(source/'techniques/sparse_backends')]
    os.environ.update(ENVIRONMENT)
    from models.minimax_h3.A100.registration import _verify_upstream_model
    _verify_upstream_model()
    from models.minimax_h3.A100 import adapter
    from sol_attn import interface
    def portable_backend(arch,*,cute_available=None):
        if arch[0]<8:raise RuntimeError('Unsupported Sol GPU architecture')
        return 'triton'
    interface._backend_for_arch=portable_backend
    from .sol_gate import checked_gate
    from .sol_stats import error_stats
    adapter._error_stats=error_stats
    adapter._run_correctness_gate=checked_gate
    from models.minimax_h3.A100.model import MiniMaxH3DiTModel
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
    ModelRegistry.register_model('MiniMaxH3DiTModel',MiniMaxH3DiTModel)
    ModelRegistry.register_model('MiniMaxH3Transformer3DModel',MiniMaxH3DiTModel)
    from sglang.multimodal_gen.runtime.managers.memory_managers import component_manager as cm
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import VanillaD2HStrategy
    original=cm.build_component_residency_strategy
    class AuditedTextOffload(VanillaD2HStrategy):
        def finish_use(self,module,use,state):
            import torch
            before=torch.cuda.memory_allocated()
            super().finish_use(module,use,state)
            torch.cuda.synchronize();torch.cuda.empty_cache()
            parameters=list(module.parameters())
            if not all(p.device.type=='cpu' for p in parameters):
                raise RuntimeError('Text encoder offload incomplete')
            context=adapter._STATE.context
            adapter.emit_event('text_encoder_offloaded',component=use.component_name,
                request_epoch=Path(os.environ['H3_REQUEST_EPOCH_FILE']).read_text().strip(),
                allocated_before=before,allocated_after=torch.cuda.memory_allocated(),
                parameter_bytes=sum(p.numel()*p.element_size() for p in parameters))
    def residency(component_name,module,server_args):
        if component_name=='text_encoder':
            if cm.is_fsdp_managed_module(module):
                raise RuntimeError('Unvalidated native FSDP text offload is forbidden')
            return AuditedTextOffload()
        return original(component_name,module,server_args)
    cm.build_component_residency_strategy=residency
    _installed=True


def audit(folder,epoch,ranks):
    from collections import Counter
    import json
    events=[]
    for path in Path(folder).glob('sol_events_rank*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('request_epoch')==epoch:events.append(row)
    results={}
    for rank in range(ranks):
        rows=[row for row in events if row['rank']==rank]
        sparse=[row for row in rows if row['event']=='first_sparse_forward']
        gates=[row for row in rows if row['event']=='real_qkv_correctness_gate']
        counts=Counter(row['action'] for row in rows if row['event']=='firstblockcache_decision')
        results[str(rank)]={'sparse_backends':sorted({row['backend'] for row in sparse}),
                            'gates_passed':bool(gates) and all(row['passed'] for row in gates),
                            'cache_compute':counts['compute'],'cache_reuse':counts['reuse']}
    engaged=all(row['sparse_backends']==['triton'] and row['gates_passed'] and row['cache_compute']>0 for row in results.values())
    return {'verified_sol_and_cache_execution':engaged,'per_rank':results,
            'reuse_observed_on_all_ranks':all(row['cache_reuse']>0 for row in results.values()),
            'scope':'Execution evidence only, not a matched speed or sparse-video quality benchmark'}
