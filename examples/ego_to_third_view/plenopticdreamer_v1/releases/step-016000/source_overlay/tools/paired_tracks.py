"""Fixed sparse paired-track cache schema used by the trajectory conditioner."""
import os
from pathlib import Path

import numpy as np


VERSION = 2
VIEW_NAMES = ('source', 'target')


def pack_paired_tracks(view_positions, view_depths, view_confidences, length, maximum,
                       identities=None):
    """Pack stable per-point source/target trajectories into fixed-size arrays.

    Inputs use ``(target, source)`` order to match the geometry preparation code;
    serialized tensors use ``(source, target)`` order for the conditioner API.
    A track row is a stable ``(entity, instance, point)`` identity across time.
    """
    if maximum < 1:
        raise ValueError('Paired-track capacity must be positive')
    target_positions, source_positions = view_positions
    target_depths, source_depths = view_depths
    target_confidences, source_confidences = view_confidences
    rows = {}
    instance_numbers = {}

    def identity_at(t, entity):
        value = entity if identities is None else identities[t][entity]
        return None if value is None else str(value)

    for t in range(length):
        entity_count = len(target_positions[t])
        if entity_count != len(source_positions[t]):
            raise ValueError('Source and target entity counts differ')
        for entity in range(entity_count):
            identity = identity_at(t, entity)
            if identity is None:
                continue
            instance_key = (entity, identity)
            if instance_key not in instance_numbers:
                instance_numbers[instance_key] = len(instance_numbers)
            count = min(len(target_positions[t][entity]), len(source_positions[t][entity]),
                        len(target_depths[t][entity]), len(source_depths[t][entity]),
                        len(target_confidences[t][entity]), len(source_confidences[t][entity]))
            for point in range(count):
                key = (entity, identity, point)
                if key not in rows:
                    rows[key] = dict(entity=entity, instance=instance_numbers[instance_key],
                        point=point, coords=np.zeros((2, length, 2), np.float32),
                        depth=np.zeros((2, length), np.float32),
                        confidence=np.zeros((2, length), np.float32))
                row = rows[key]
                for view, (positions, depths, confidences) in enumerate((
                        (source_positions, source_depths, source_confidences),
                        (target_positions, target_depths, target_confidences))):
                    coordinate = np.asarray(positions[t][entity][point], dtype=np.float32)
                    depth = float(depths[t][entity][point])
                    confidence = float(confidences[t][entity][point])
                    if (coordinate.shape == (2,) and np.isfinite(coordinate).all()
                            and np.isfinite(depth) and depth > 0
                            and np.isfinite(confidence) and confidence > 0):
                        row['coords'][view, t] = coordinate
                        row['depth'][view, t] = depth
                        row['confidence'][view, t] = min(1., confidence)

    # Prefer tracks supported in both views, then longer source tracks. Stable
    # semantic ordering makes cache generation reproducible across processes.
    eligible = [row for row in rows.values()
                if np.any(row['confidence'][0] > 0) and np.any(row['confidence'][1] > 0)]
    ranked = sorted(eligible, key=lambda row: (
        -int(np.logical_and(row['confidence'][0] > 0,
                            row['confidence'][1] > 0).sum()),
        -int((row['confidence'][0] > 0).sum()),
        row['entity'], row['instance'], row['point']))[:maximum]
    coords = np.zeros((maximum, 2, length, 2), np.float32)
    depth = np.zeros((maximum, 2, length), np.float32)
    confidence = np.zeros((maximum, 2, length), np.float32)
    entity = np.zeros(maximum, np.uint8)
    instance = np.zeros(maximum, np.uint16)
    point = np.zeros(maximum, np.uint16)
    for index, row in enumerate(ranked):
        coords[index] = row['coords']
        depth[index] = row['depth']
        confidence[index] = row['confidence']
        entity[index] = row['entity']
        instance[index] = row['instance']
        point[index] = row['point']
    observations = int((confidence > 0).sum())
    paired_observations = int(np.logical_and(confidence[:, 0] > 0,
                                             confidence[:, 1] > 0).sum())
    return dict(track_coords=coords, track_depth=depth, track_confidence=confidence,
                track_entity=entity, track_instance=instance, track_point=point,
                track_count=len(ranked), track_observations=observations,
                paired_track_observations=paired_observations)


def write_paired_cache(path, *, signature, roi_mask, corr_coords, corr_confidence,
                       corr_kind, corr_entity, tracks):
    """Atomically publish a correspondence cache with paired sparse tracks."""
    from hoi_supervision import file_hash, retry_io
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        temporary = path.with_name(path.stem + f'.{os.getpid()}.partial.npz')
        np.savez_compressed(temporary, signature=np.asarray(signature),
            paired_track_version=np.asarray(VERSION, dtype=np.int64),
            roi_mask=np.asarray(roi_mask, dtype=np.uint8),
            corr_coords=np.asarray(corr_coords, dtype=np.float32),
            corr_confidence=np.asarray(corr_confidence, dtype=np.float32),
            corr_kind=np.asarray(corr_kind, dtype=np.uint8),
            corr_entity=np.asarray(corr_entity, dtype=np.uint8),
            track_coords=np.asarray(tracks['track_coords'], dtype=np.float32),
            track_depth=np.asarray(tracks['track_depth'], dtype=np.float32),
            track_confidence=np.asarray(tracks['track_confidence'], dtype=np.float32),
            track_entity=np.asarray(tracks['track_entity'], dtype=np.uint8),
            track_instance=np.asarray(tracks['track_instance'], dtype=np.uint16),
            track_point=np.asarray(tracks['track_point'], dtype=np.uint16))
        os.replace(temporary, path)

    retry_io(publish)
    stat = retry_io(path.stat)
    return {'cache': str(path), 'cache_sha256': file_hash(path),
            'cache_size': stat.st_size, 'cache_mtime_ns': stat.st_mtime_ns}


def validate_paired_cache(path, frames, max_correspondences, max_tracks):
    length = (frames - 1) // 4 + 1
    expected = {
        'roi_mask': [length, 54, 96],
        'corr_coords': [max_correspondences, 2, 3],
        'corr_confidence': [max_correspondences],
        'corr_kind': [max_correspondences],
        'corr_entity': [max_correspondences],
        'track_coords': [max_tracks, 2, length, 2],
        'track_depth': [max_tracks, 2, length],
        'track_confidence': [max_tracks, 2, length],
        'track_entity': [max_tracks],
        'track_instance': [max_tracks],
        'track_point': [max_tracks],
    }
    with np.load(path, allow_pickle=False) as data:
        version = int(data['paired_track_version'].item())
        shapes = {name:list(data[name].shape) for name in expected}
        if version != VERSION or shapes != expected:
            raise ValueError('Invalid paired-track cache schema or shape')
        for name in ('corr_coords', 'corr_confidence', 'track_coords',
                     'track_depth', 'track_confidence'):
            if not np.isfinite(data[name]).all():
                raise ValueError('Paired-track cache contains non-finite values')
        confidence = data['track_confidence']
        depth = data['track_depth']
        coords = data['track_coords']
        supported_coords = coords[np.repeat((confidence > 0)[..., None], 2, axis=-1)]
        if ((confidence < 0).any() or (confidence > 1).any()
                or (depth[confidence > 0] <= 0).any()
                or (supported_coords < 0).any() or (supported_coords > 1).any()
                or (data['track_entity'] >= 4).any()):
            raise ValueError('Paired-track confidence, depth, or entity is invalid')
        return {'signature': str(data['signature'].item()), 'version': version,
                'shapes': shapes, 'track_count': int(np.any(confidence > 0, axis=(1, 2)).sum()),
                'track_observations': int((confidence > 0).sum()),
                'paired_track_observations': int(np.logical_and(
                    confidence[:, 0] > 0, confidence[:, 1] > 0).sum())}
