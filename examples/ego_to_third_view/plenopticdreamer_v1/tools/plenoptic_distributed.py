"""One NCCL communicator per process; CPU Gloo across DP replicas."""
import hashlib
from datetime import timedelta
import torch
import torch.distributed as dist


def make_groups(world, cp, rank, dp_backend='gloo'):
    group = None
    if cp > 1:
        for first in range(0, world, cp):
            candidate = dist.new_group(list(range(first, first + cp)), backend='nccl',
                                       timeout=timedelta(minutes=5))
            if first == rank // cp * cp:
                group = candidate
    if dp_backend not in ('gloo','nccl'):
        raise ValueError('DP backend must be gloo or nccl')
    leaders = dist.new_group(list(range(0, world, cp)), backend=dp_backend,
                             timeout=timedelta(minutes=5)) if world > cp else None
    return group, leaders


def verify_initial_parameters(parameters, world):
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    values = [None] * world
    dist.all_gather_object(values, digest.hexdigest())
    if len(set(values)) != 1:
        raise RuntimeError('Initial trainable parameters differ across ranks')


def synchronize_gradients(parameters, world, cp=1, group=None, leaders=None,
                          bucket_bytes=25 * 1024 * 1024, dp_backend='gloo'):
    """Average the local spatial losses over CP and DP, in a fixed order.

    Each CP group sums on its leader. Only leaders copy the sum to CPU and
    reduce across nodes. The global mean is broadcast within each CP group.
    No NCCL communicator spans nodes or overlaps the CP communicator.
    """
    if world == 1:
        return
    torch.cuda.synchronize()
    missing = torch.tensor([int(any(p.grad is None for p in parameters))])
    dist.all_reduce(missing, op=dist.ReduceOp.MAX)
    if missing.item():
        raise RuntimeError('Missing trainable gradient on at least one rank')
    rank = dist.get_rank()
    source = rank // cp * cp
    bucket, size = [], 0

    def flush():
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        if cp > 1:
            dist.reduce(flat, dst=source, group=group)
        if dp_backend=='nccl':
            # No rank may enqueue its CP broadcast while a leader is using
            # the DP communicator. Finish both GPU phases explicitly.
            torch.cuda.synchronize()
            if rank==source:
                if world>cp:
                    dist.all_reduce(flat,group=leaders)
                flat.div_(world)
                torch.cuda.synchronize()
            dist.barrier()  # CPU Gloo; nonleaders have no pending GPU work.
        elif rank == source:
            cpu = flat.cpu()
            if world > cp:
                dist.all_reduce(cpu, group=leaders)
            flat.copy_(cpu.div_(world))
        if cp > 1:
            dist.broadcast(flat, src=source, group=group)
        offset = 0
        for parameter in bucket:
            parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()

    for parameter in parameters:
        nbytes = parameter.numel() * parameter.element_size()
        if bucket and size + nbytes > bucket_bytes:
            flush()
            bucket, size = [], 0
        bucket.append(parameter)
        size += nbytes
    if bucket:
        flush()
    torch.cuda.synchronize()
