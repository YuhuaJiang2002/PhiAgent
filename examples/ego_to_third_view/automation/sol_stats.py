"""Same Sol correctness metrics with bounded temporary float32 storage."""
import torch

@torch.no_grad()
def error_stats(actual, expected, *, chunk_elements=1048576):
    if actual.shape!=expected.shape:
        raise ValueError('Correctness comparison shape mismatch')
    a=actual.reshape(-1)
    b=expected.reshape(-1)
    maximum=torch.zeros((),device=a.device,dtype=torch.float32)
    total_abs=torch.zeros((),device=a.device,dtype=torch.float64)
    total_squared=torch.zeros_like(total_abs)
    reference_squared=torch.zeros_like(total_abs)
    for offset in range(0,a.numel(),chunk_elements):
        aa=a[offset:offset+chunk_elements].float()
        bb=b[offset:offset+chunk_elements].float()
        difference=aa-bb
        maximum=torch.maximum(maximum,difference.abs().max())
        total_abs+=difference.abs().sum(dtype=torch.float64)
        total_squared+=difference.square().sum(dtype=torch.float64)
        reference_squared+=bb.square().sum(dtype=torch.float64)
    return {'max_abs':maximum.item(),'mean_abs':(total_abs/a.numel()).item(),'rel_l2':torch.sqrt(total_squared/reference_squared.clamp_min(1e-24)).item()}
