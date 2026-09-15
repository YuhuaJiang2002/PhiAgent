"""Select h20-1 validation GPUs without changing any existing GPU process."""
import socket
import subprocess

MINIMUM_FREE_MIB = 36864
LEADER_FREE_MIB = 61440
# Keep the numerical protocol fixed even when more GPUs become available.
SUPPORTED_COUNTS = (4,)


def gpu_rows():
    if socket.gethostname().split('.')[0] != 'h20-1':
        raise RuntimeError('Validation GPU selection is permitted only on h20-1')
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return [dict(zip(('index','used','free','utilization'), map(int, line.split(','))))
            for line in raw.splitlines() if line.strip()]

def memory_requirements(video_batch_size=1):
    if type(video_batch_size) is not int or video_batch_size not in (1, 2):
        raise ValueError('Video batch size must be 1 or 2')
    # Batch 2 offers little speedup here and reserves much more allocator memory.
    # Keep it opt-in and require almost idle cards for the largest supported k=4.
    return (LEADER_FREE_MIB, MINIMUM_FREE_MIB) if video_batch_size == 1 else (96256, 92160)


def choose_devices(rows, requested_gpu_indices=None, video_batch_size=1):
    by_index = {row['index']: row for row in rows}
    leader_min, follower_min = memory_requirements(video_batch_size)

    if requested_gpu_indices is not None:
        if (len(requested_gpu_indices) != 4 or len(set(requested_gpu_indices)) != 4
                or any(type(index) is not int or index < 0 for index in requested_gpu_indices)):
            raise ValueError('Request exactly four distinct non-negative GPU indices')
        missing = [index for index in requested_gpu_indices if index not in by_index]
        low_memory = [index for index in requested_gpu_indices
                      if index in by_index and by_index[index]['free'] < follower_min]
        if missing:
            raise ValueError(f'Requested GPU indices do not exist: {missing}')
        if low_memory:
            raise RuntimeError(
                f'Requested GPUs have less than {follower_min/1024:g} GiB free: {low_memory}'
            )
        eligible = [by_index[index] for index in requested_gpu_indices]
    else:
        eligible = [row for row in rows if row['free'] >= follower_min]
        if len(eligible) < 4:
            raise RuntimeError(f'Fixed validation needs four GPUs with {follower_min/1024:g} GiB free, '
                               f'including one leader with {leader_min/1024:g} GiB free')

    leader = max(eligible, key=lambda row: (row['free'], -row['index']))
    if leader['free'] < leader_min:
        raise RuntimeError(f'The validation encoder leader needs {leader_min/1024:g} GiB free')
    remaining = sorted(
        (row for row in eligible if row['index'] != leader['index']),
        key=lambda row: (row['utilization'] > 5, -row['free'], row['index']),
    )
    return [leader['index']] + [row['index'] for row in remaining[:3]]


def select_devices(requested_gpu_indices=None, video_batch_size=1):
    return choose_devices(gpu_rows(), requested_gpu_indices=requested_gpu_indices,
                          video_batch_size=video_batch_size)


def availability(requested_gpu_indices=None, video_batch_size=1):
    rows = gpu_rows()
    leader_min, follower_min = memory_requirements(video_batch_size)
    policy = dict(video_batch_size=video_batch_size, leader_free_mib=leader_min,
                  follower_free_mib=follower_min)
    try:
        devices = choose_devices(rows, requested_gpu_indices=requested_gpu_indices,
                                 video_batch_size=video_batch_size)
    except RuntimeError as exc:
        return dict(ready=False, devices=[], gpus=rows, reason=str(exc), context_parallel_size=4, **policy)
    return dict(ready=True, devices=devices, gpus=rows, reason=None, context_parallel_size=4, **policy)


def check_devices(devices, video_batch_size=1):
    if len(devices) not in SUPPORTED_COUNTS or len(set(devices)) != len(devices):
        raise ValueError('Fixed validation requires exactly four distinct local GPUs')
    rows = {r['index']:r for r in gpu_rows()}
    leader_min, follower_min = memory_requirements(video_batch_size)
    if any(index not in rows or rows[index]['free'] < follower_min for index in devices):
        raise RuntimeError('Available memory changed before validation started; existing jobs were left running')
    if rows[devices[0]]['free'] < leader_min:
        raise RuntimeError(f'The validation encoder leader no longer has {leader_min/1024:g} GiB free')
    

def parse_gpu_ids(value):
    try:
        devices = [int(part.strip()) for part in value.split(',')]
    except ValueError as exc:
        raise ValueError('--gpus must be comma-separated GPU indices') from exc
    if len(devices) != 4 or len(set(devices)) != 4 or any(index < 0 for index in devices):
        raise ValueError('--gpus requires exactly four distinct non-negative GPU indices')
    return devices


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--gpus')
    parser.add_argument('--video-batch-size', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    requested = parse_gpu_ids(args.gpus) if args.gpus is not None else None
    print(json.dumps(availability(requested, args.video_batch_size)) if args.json
      else ','.join(map(str, select_devices(requested, args.video_batch_size))))
