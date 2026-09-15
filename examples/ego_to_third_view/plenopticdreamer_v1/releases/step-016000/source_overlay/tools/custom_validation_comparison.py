#!/usr/bin/env python3
"""Compare target-free custom cases and assemble segmented full-video runs."""
import plenoptic_paths as layout
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import prepare_plenoptic as prepare
from plenoptic_data import decode_video
from validation_suite import digest_file, digest_json


def rooted(value):
    path = (layout.rooted(Path(value))).resolve()
    layout.require_workspace_path(path)
    return path


def save(path, value):
    prepare.save_json(Path(path), value)


def probe_video(path, *, frames, width, height, fps=15):
    value = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate',
        '-of', 'json', str(path)], text=True))['streams'][0]
    observed = (int(value['nb_read_frames']), int(value['width']),
                int(value['height']), value['r_frame_rate'])
    expected = (frames, width, height, f'{fps}/1')
    if observed != expected:
        raise ValueError(f'Video geometry differs: observed={observed}, expected={expected}')
    return value


def render_custom_comparison(run, record, panel_width):
    """Create the per-case comparison imported by compare_inference.py."""
    if record.get('has_target_reference') is not False:
        raise ValueError('Custom comparison requires an explicitly absent target reference')
    scene = record['custom_scene']
    if set(scene['videos']) != {'source'}:
        raise ValueError('Custom input must contain only the source video')
    hw = (record['height'], record['width'])
    indices = list(range(record['frames']))

    def decode(path):
        return decode_video(path, indices, hw, calibrated_crop=True).permute(1, 2, 3, 0).numpy()

    source = decode(scene['videos']['source'])
    generated = decode(str(run / 'generated.mp4'))
    panel_w = min(panel_width, hw[1])
    panel_h = round(panel_w * hw[0] / hw[1] / 2) * 2
    width, height = 2 * panel_w, panel_h + 32
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    output = run / 'comparison.mp4'
    temporary = run / 'comparison.tmp.mp4'
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', '15', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', str(temporary)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in indices:
                canvas = Image.new('RGB', (width, height), '#171b24')
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 7), 'SOURCE VIDEO', font=font, fill='white')
                draw.text((panel_w + 8, 7),
                          f'GENERATED TARGET VIEW  step={record["checkpoint_step"]}',
                          font=font, fill='white')
                for column, clip in enumerate((source, generated)):
                    picture = Image.fromarray(clip[index]).resize(
                        (panel_w, panel_h), Image.Resampling.LANCZOS)
                    canvas.paste(picture, (column * panel_w, 32))
                encoder.stdin.write(np.asarray(canvas).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Custom comparison encoding failed')
        except BaseException:
            if encoder.poll() is None:
                encoder.terminate()
            raise
    temporary.replace(output)
    encoded = probe_video(output, frames=len(indices), width=width, height=height)
    manifest = dict(status='passed', qualitative_only=True, has_target_reference=False,
        columns=['source video', 'generated target view'], checkpoint_step=record['checkpoint_step'],
        k=record['k'], dataset='custom', scene_id=scene['scene_id'], frames=len(indices), fps=15,
        encoded=encoded, comparison=str(layout.relative(output)),
        camera_provenance=scene['camera_provenance'],
        scope='Visual-only user-video check; no reference video and no quality metric.',
        updated_at=datetime.now().astimezone().isoformat())
    save(run / 'comparison.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def concatenate_segments(output, segments):
    temporary = output.with_suffix('.tmp.mp4')
    command = ['ffmpeg', '-v', 'error', '-y']
    filters, labels = [], []
    for index, segment in enumerate(segments):
        command += ['-i', str(segment['video'])]
        label = f'v{index}'
        selected = segment.get('generated_frame_indices')
        if selected is not None:
            if (len(selected) != segment['valid_frames'] or any(type(i) is not int or not 0 <= i < 81 for i in selected)
                    or selected != sorted(set(selected))):
                raise ValueError('Invalid original-timeline frame selection')
            expression = '+'.join(f'eq(n,{i})' for i in selected)
            filters.append(f"[{index}:v]select='{expression}',setpts=N/(15*TB)[{label}]")
        else:
            filters.append(
                f'[{index}:v]trim=end_frame={segment["valid_frames"]},setpts=PTS-STARTPTS[{label}]')
        labels.append(f'[{label}]')
    filters.append(''.join(labels) + f'concat=n={len(labels)}:v=1:a=0[outv]')
    command += ['-filter_complex', ';'.join(filters), '-map', '[outv]', '-an',
                '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-threads', '4', str(temporary)]
    subprocess.run(command, check=True)
    temporary.replace(output)


def side_by_side(source, generated, output, frames):
    temporary = output.with_suffix('.tmp.mp4')
    subprocess.run([
        'ffmpeg', '-v', 'error', '-y', '-i', str(source), '-i', str(generated),
        '-filter_complex', '[0:v]trim=end_frame=%d,setpts=PTS-STARTPTS[s];'
                           '[1:v]trim=end_frame=%d,setpts=PTS-STARTPTS[g];[s][g]hstack=2[outv]'
                           % (frames, frames),
        '-map', '[outv]', '-an', '-c:v', 'libx264', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-threads', '4', str(temporary)], check=True)
    temporary.replace(output)


def assemble_group(run, group, summaries):
    group_id = group['id']
    directory = run / group_id
    expected_start = 0
    segment_inputs, inference_records, summary_records = [], [], []
    for segment in group['segments']:
        if segment['start_frame'] != expected_start or segment['valid_frames'] < 1:
            raise ValueError(f'{group_id}: segment coverage is not contiguous')
        expected_start += segment['valid_frames']
        case_id = segment['case_id']
        case_dir = run / case_id
        video, metadata = case_dir / 'generated.mp4', case_dir / 'inference.json'
        if not video.is_file() or not metadata.is_file():
            raise FileNotFoundError(f'{group_id}: incomplete segment {case_id}')
        record = json.loads(metadata.read_text())
        if (record.get('status') != 'generated' or record.get('case_id') != case_id
                or record.get('target_reference_used_during_generation') is not False
                or record.get('has_target_reference') is not False):
            raise ValueError(f'{group_id}: invalid segment inference record {case_id}')
        summary = summaries.get(case_id)
        if summary is None or summary.get('video_directory') != case_id:
            raise ValueError(f'{group_id}: missing segment validation summary {case_id}')
        segment_inputs.append(dict(video=video, valid_frames=segment['valid_frames'],
                                   case_id=case_id, sha256=digest_file(video),
                                   **({'generated_frame_indices':segment['generated_frame_indices']}
                                      if 'generated_frame_indices' in segment else {})))
        inference_records.append(record)
        summary_records.append(summary)
    if expected_start != group['total_frames']:
        raise ValueError(f'{group_id}: segments do not cover the declared full video')
    invariant = ('checkpoint_step', 'checkpoint_sha256', 'k', 'height', 'width',
                 'context_parallel_size', 'cuda_visible_devices', 'target_camera')
    for key in invariant:
        if len({json.dumps(record.get(key), sort_keys=True) for record in inference_records}) != 1:
            raise ValueError(f'{group_id}: segment inference mismatch for {key}')
    identity = dict(group_sha256=digest_json(group),
        segments=[dict(case_id=item['case_id'], generated_sha256=item['sha256'],
                       inference_sha256=digest_json(info))
                  for item, info in zip(segment_inputs, inference_records)])
    receipt_path = directory / 'assembly-receipt.json'
    if directory.exists():
        if not receipt_path.is_file():
            raise FileExistsError(f'Full-video output has no completion receipt: {directory}')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('identity') != identity:
            raise ValueError(f'{group_id}: completed assembly inputs changed')
        for filename, expected in receipt['files'].items():
            if digest_file(directory / filename) != expected:
                raise ValueError(f'{group_id}: completed assembly file changed: {filename}')
        return receipt['summary']
    directory.mkdir()
    generated = directory / 'generated.mp4'
    concatenate_segments(generated, segment_inputs)
    probe_video(generated, frames=group['total_frames'], width=768, height=432)
    source = rooted(group['full_scene']['videos']['source'])
    comparison = directory / 'comparison.mp4'
    side_by_side(source, generated, comparison, group['total_frames'])
    encoded = probe_video(comparison, frames=group['total_frames'], width=1536, height=432)
    first = inference_records[0]
    case_ids = [item['case_id'] for item in segment_inputs]
    generated_sha = digest_file(generated)
    inference = dict(first, case_id=group_id, scene_id=group['full_scene']['scene_id'],
        custom_scene=group['full_scene'], frames=group['total_frames'],
        video=str(layout.relative(generated)), input_sha256=group['input_sha256'],
        seconds=round(sum(float(item.get('seconds', 0.)) for item in inference_records), 3),
        assembled_from_segments=case_ids,
        segment_generated_sha256={item['case_id']: item['sha256'] for item in segment_inputs},
        generated_sha256=generated_sha, assembled_full_video=True,
        target_reference_used_during_generation=False, has_target_reference=False,
        qualitative_only=True, created_at=datetime.now().astimezone().isoformat())
    save(directory / 'inference.json', inference)
    comparison_record = dict(status='passed', qualitative_only=True, has_target_reference=False,
        columns=['normalized full source video', 'assembled generated target view'],
        checkpoint_step=first['checkpoint_step'], k=first['k'], dataset='custom',
        scene_id=group['full_scene']['scene_id'], frames=group['total_frames'], fps=group['fps'],
        encoded=encoded, comparison=str(layout.relative(comparison)),
        generated_sha256=generated_sha, assembled_from_segments=case_ids,
        camera_provenance=group['full_scene']['camera_provenance'],
        target_reference_used_during_generation=False,
        scope='Complete chronological user video assembled from non-overlapping generated segments; '
              'no target reference and no quality metric.',
        updated_at=datetime.now().astimezone().isoformat())
    save(directory / 'comparison.json', comparison_record)
    first_summary = summary_records[0]
    summary = dict(first_summary, case_id=group_id, scene_id=group['full_scene']['scene_id'],
        video_directory=group_id,
        seconds=round(sum(float(item.get('seconds', 0.)) for item in summary_records), 3),
        assembled_full_video=True, segment_cases=case_ids, total_frames=group['total_frames'])
    save(receipt_path, dict(status='complete', identity=identity, summary=summary,
         files={name:digest_file(directory / name) for name in
                ('generated.mp4', 'comparison.mp4', 'inference.json', 'comparison.json')}))
    return summary


def assemble_run(value):
    run = rooted(value)
    suite_path, report_path = run / 'custom_suite.json', run / 'validation.json'
    if not suite_path.is_file() or not report_path.is_file():
        raise FileNotFoundError('Run requires custom_suite.json and validation.json')
    suite, report = json.loads(suite_path.read_text()), json.loads(report_path.read_text())
    groups = suite.get('video_groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('Custom suite has no full-video groups')
    segments = report.get('qualitative_cases', [])
    summaries = {row['case_id']: row for row in segments}
    if len(summaries) != len(segments):
        raise ValueError('Qualitative segment case IDs are not unique')
    expected = {segment['case_id'] for group in groups for segment in group['segments']}
    if set(summaries) != expected:
        raise ValueError('Validation qualitative cases differ from the video-group segments')
    assembled = [assemble_group(run, group, summaries) for group in groups]
    report['segment_qualitative_cases'] = segments
    report['qualitative_cases'] = assembled
    report['generation_total_cases'] = report.get('total_cases')
    report['total_cases'] = len(report['cases']) + len(assembled)
    report['published_full_video_groups'] = [group['id'] for group in groups]
    report['full_video_assembly'] = dict(
        status='passed', target_reference_used_during_generation=False,
        generated_segment_cases=len(segments), published_full_videos=len(assembled),
        updated_at=datetime.now().astimezone().isoformat())
    save(report_path, report)
    result = dict(status='assembled', run=str(layout.relative(run)),
                  groups=report['published_full_video_groups'],
                  generated_segment_cases=len(segments), published_full_videos=len(assembled))
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assemble-run', required=True)
    args = parser.parse_args()
    assemble_run(args.assemble_run)


if __name__ == '__main__':
    main()
