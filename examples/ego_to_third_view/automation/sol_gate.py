"""Bounded-memory Sol gate with FP32 adjudication of BF16-reference outliers.

All original numerical limits are retained. A failing BF16 SDPA comparison is
not silently accepted: every offending query/head is checked against explicit
FP32 QK/softmax/V arithmetic, rounded once to the output dtype.
"""
import os
import torch
from .sol_stats import error_stats

@torch.no_grad()
def checked_gate(kernel,q,k,v,*,scale,thresh_type):
    from models.minimax_h3.A100 import adapter
    context=adapter._STATE.context
    epoch=context.request_epoch if context is not None else 'unknown'
    prefix_tokens=int(context.prefix_tokens) if context is not None else 0
    shape=(epoch,prefix_tokens,int(q.shape[1]),int(q.shape[2]),int(q.shape[3]))
    if shape in adapter._STATE.gated_shapes:
        return
    actual=kernel(q,k,v,scale=scale,tau=-1000.,thresh_type=thresh_type,kv_splits=1)
    # Validate the output that the adapter actually consumes: its full prefix
    # queries are overwritten by _dense_queries after every Sol invocation.
    # Do not reject a discarded raw-kernel prefix, or omit prefix validation.
    if prefix_tokens:
        actual[:,:prefix_tokens]=adapter._dense_queries(q,k,v,start=0,tokens=prefix_tokens,scale=scale)
    reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.,is_causal=False,scale=scale).transpose(1,2)
    stats=error_stats(actual,reference)
    limits={'max_abs':float(os.getenv('H3_SOL_GATE_MAX_ABS','.15')),'mean_abs':float(os.getenv('H3_SOL_GATE_MEAN_ABS','.002')),'rel_l2':float(os.getenv('H3_SOL_GATE_REL_L2','.005'))}
    original=dict(stats)
    recheck=None
    if stats['max_abs']>limits['max_abs'] and stats['mean_abs']<=limits['mean_abs'] and stats['rel_l2']<=limits['rel_l2']:
        rows=[]
        for offset in range(0,q.shape[1],1024):
            delta=(actual[:,offset:offset+1024].float()-reference[:,offset:offset+1024].float()).abs().amax(dim=-1)
            for b,t,h in torch.nonzero(delta>limits['max_abs']).cpu().tolist():
                rows.append((b,t+offset,h))
        if len(rows)>256:
            raise RuntimeError(f'Too many BF16-reference outlier rows for bounded FP32 adjudication: {len(rows)}')
        maximum=0.
        details=[]
        tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            for b,h in sorted(set((b,h) for b,t,h in rows)):
                indices=[t for bb,t,hh in rows if (bb,hh)==(b,h)]
                queries=q[b,indices,h].float()
                keys=k[b,:,h].float()
                vals=v[b,:,h].float()
                logits=torch.matmul(queries,keys.T)*scale
                weights=torch.softmax(logits,dim=-1)
                fp32=torch.matmul(weights,vals)
                rounded=fp32.to(actual.dtype).float()
                got=actual[b,indices,h].float()
                old=reference[b,indices,h].float()
                err=(got-rounded).abs().max().item()
                maximum=max(maximum,err)
                details.append({'batch':b,'head':h,'query_indices':indices,'sol_vs_fp32_rounded_max_abs':err,'sdpa_vs_fp32_rounded_max_abs':(old-rounded).abs().max().item(),'sol_vs_fp32_unrounded_max_abs':(got-fp32).abs().max().item()})
        finally:
            torch.backends.cuda.matmul.allow_tf32=tf32
        recheck={'query_head_rows':len(rows),'max_abs':maximum,'details':details,'reference':'Explicit FP32 QK-softmax-V, TF32 disabled, final BF16 rounding only'}
        adapter.emit_event('fp32_reference_adjudication',shape=list(q.shape),original_stats=original,limits=limits,**recheck)
        # Non-outlier rows already meet the original maximum limit. The limit
        # is a conservative upper bound for their maximum, not a new tolerance.
        stats['max_abs']=max(limits['max_abs'],maximum)
    passed=all(stats[name]<=limit for name,limit in limits.items())
    adapter.emit_event('real_qkv_correctness_gate',request_epoch=epoch,passed=passed,shape=list(q.shape),stats=stats,limits=limits,original_sdpa_stats=original,fp32_reference_adjudication=recheck,scope='Full consumed output: exact prefix queries plus Sol target queries; tau=-1000 dense-equivalent check',exact_prefix_queries=prefix_tokens)
    if not passed:
        from pathlib import Path
        if os.getenv('H3_SAVE_FAILED_QKV')=='1' and os.getenv('H3_REQUEST_EPOCH_FILE'):
            capture=Path(os.environ['H3_REQUEST_EPOCH_FILE']).parent/f'failed_gate_qkv_rank{os.getenv("RANK","unknown")}.pt'
            torch.save({'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),'scale':scale,'prefix_tokens':prefix_tokens,'stats':stats},capture)
        raise RuntimeError(f'Sol correctness gate failed, including FP32 outlier adjudication: {stats}')
    adapter._STATE.gated_shapes.add(shape)
