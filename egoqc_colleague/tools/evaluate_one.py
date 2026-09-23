#!/usr/bin/env python3
"""Single-video I/O demo using the colleague's v3db prompt/sampler/rules.

Does not load evaluation labels, start an HTTP server, or delete source videos.
Uses one bounded API request, without the optional non-decision audit call.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parents[1]
ALLOWED_INPUTS = {'request_id', 'video_path', 'task_description'}
REQUIRED_CHECKS = ('q1_device_exposed', 'q2_task_completion', 'q3_blur',
                   'q4_temporal', 'q5_face', 'coverage', 'confidence')


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate_input(value):
    if not isinstance(value, dict) or set(value) != ALLOWED_INPUTS:
        raise ValueError('Input must contain only request_id, video_path, task_description; no ground truth.')
    for k in ALLOWED_INPUTS:
        if not isinstance(value[k], str) or not value[k].strip():
            raise ValueError(f'{k} must be a non-empty string')
    path = Path(value['video_path']).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('video_path must be a local regular file')
    return {**value, 'video_path': str(path)}


def load_engine():
    path = PROJECT / 'qc_run_v3.py'
    spec = importlib.util.spec_from_file_location('colleague_v3db', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'format=duration:stream=avg_frame_rate,width,height',
        '-of', 'json', video], capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    n, d = map(float, stream['avg_frame_rate'].split('/'))
    duration, fps = float(data['format']['duration']), n / d
    if not all(math.isfinite(v) and v > 0 for v in (duration, fps)):
        raise ValueError('Invalid video duration or frame rate')
    return {'duration_sec': duration, 'fps': fps, 'width': stream['width'], 'height': stream['height']}


def stream_call(endpoint, api_key, payload):
    """Single request: fail visibly on quota/auth errors instead of six retries."""
    started = time.monotonic()
    content, reasoning_chars, usage = [], 0, {}
    finish, response_model, response_id = None, None, None
    with requests.post(endpoint, json=payload, headers={
        'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
        stream=True, timeout=(15, 180), allow_redirects=False) as response:
        if response.status_code != 200:
            detail = response.text.replace(api_key, '[REDACTED]')[:1200]
            raise RuntimeError(f'Provider HTTP {response.status_code}: {detail}')
        for raw in response.iter_lines():
            if time.monotonic() - started > 360:
                raise TimeoutError('Provider streaming deadline exceeded')
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(json.dumps(event['error'], ensure_ascii=False).replace(api_key, '[REDACTED]'))
            response_model = event.get('model') or response_model
            response_id = event.get('id') or response_id
            usage = event.get('usage') or usage
            for choice in event.get('choices') or []:
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    content.append(delta['content'])
                reasoning_chars += len(delta.get('reasoning_content') or '')
    return {'content': ''.join(content), 'model': response_model, 'response_id': response_id,
            'finish_reason': finish, 'usage': usage, 'reasoning_chars': reasoning_chars,
            'elapsed_sec': round(time.monotonic() - started, 3)}


def validate_checks(checks, labels):
    errors = []
    if not isinstance(checks, dict):
        return ['model_output_not_json_object']
    for key in REQUIRED_CHECKS:
        if key not in checks:
            errors.append('missing_' + key)
    enums = {
        'q1_device_exposed': {'未出现', '疑似出现但看不清', '明确出现', '无法判断'},
        'q2_task_completion': {'完成', '部分完成', '未完成'},
        'q3_blur': {'清晰', '轻度模糊', '严重模糊', '中心区域模糊', '画面发白过曝'},
        'q5_face': {'出现', '未出现', '无法判断'},
    }
    for key, allowed in enums.items():
        item = checks.get(key)
        if not isinstance(item, dict) or item.get('verdict') not in allowed:
            errors.append('invalid_' + key)
    for key in REQUIRED_CHECKS[:5]:
        item = checks.get(key)
        if not isinstance(item, dict):
            errors.append('invalid_' + key); continue
        ids = item.get('frame_ids')
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in labels for i in ids):
            errors.append('invalid_frame_ids_' + key)
    temporal = checks.get('q4_temporal')
    if not isinstance(temporal, dict) or not isinstance(temporal.get('flags'), list):
        errors.append('invalid_temporal_flags')
    coverage = checks.get('coverage')
    if not isinstance(coverage, dict) or type(coverage.get('covers_whole_video')) is not bool:
        errors.append('invalid_coverage')
    confidence = checks.get('confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append('invalid_confidence')
    return sorted(set(errors))


def evaluate_one(value, output, env_file, model, think='auto', prepare_only=False):
    request = validate_input(value)
    config = {**dotenv_values(env_file), **os.environ}
    api_key = config.get('ARK_API_KEY') or ''
    base = config.get('ARK_BASE') or config.get('ARK_BASE_URL') or ''
    url = urlsplit(base)
    if (not prepare_only and not api_key) or url.scheme != 'https' or url.hostname != 'ark.cn-beijing.volces.com' or url.username or url.query:
        raise ValueError('Missing key or unexpected Ark endpoint; refusing to send credentials')
    endpoint = base.rstrip('/') + '/chat/completions'
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dump(output / 'input.json', request)
    try:
        engine = load_engine()
        video = probe(request['video_path'])
        if video['duration_sec'] < engine.MIN_DUR:
            raise ValueError('Demo requires a video >= 5 seconds to exercise the model, not the short-video shortcut')
        frame_dir = output / 'frames'
        labels, stamps = engine.extract(request['video_path'], str(frame_dir), video['duration_sec'])
        blocks = engine.build_blocks(str(frame_dir), labels, stamps)
        if not labels or any(not (frame_dir / (label + '.jpg')).is_file() for label in labels):
            raise ValueError('Frame extraction failed')
        frames = []
        for label in labels:
            group, index = label.split('_')
            path = frame_dir / (label + '.jpg')
            frames.append({'frame_id': label, 'estimated_timestamp_sec': stamps[group][int(index)],
                'relative_path': str(path.relative_to(output)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'view': 'bottom_35_percent_crop' if group == 'B' else 'full_frame'})
        counts = Counter(label.split('_')[0] for label in labels)
        groups = '，'.join(f'{g}{counts[g]}帧' for g in engine.GROUP_DESC if counts.get(g))
        head, tail = engine.accept_windows(video['duration_sec'])
        dur = video['duration_sec']
        prompt = engine.user_prompt(request['task_description'],
            f'时长约 {dur:.1f} 秒，{video["fps"]:g}fps；共提供 {len(labels)} 帧（{groups}）；'
            f'首尾豁免窗口 = 开头 {head:.1f}s 与 结尾 {tail:.1f}s，'
            f'即 [0, {head:.1f}s] 与 [{max(dur-tail,0):.1f}s, {dur:.1f}s] 属豁免区间，'
            f'({head:.1f}s, {max(dur-tail,0):.1f}s) 属中间区间')
        content = [{'type': 'text', 'text': prompt}]
        preview_content = list(content)
        for block in blocks:
            if isinstance(block, str):
                part = {'type': 'text', 'text': block}
                content.append(part); preview_content.append(part)
            else:
                label, path = block
                content.append({'type': 'image_url', 'image_url': {'url': engine.data_url(path)}})
                preview_content.append({'type': 'image_url', 'image_url': {'url': f'<JPEG base64 omitted: frames/{label}.jpg>'}})
        body = {'model': model, 'temperature': 0, 'max_tokens': 16000, 'stream': True,
            'stream_options': {'include_usage': True}, 'messages': [
            {'role': 'system', 'content': engine.SYSTEM}, {'role': 'user', 'content': content}]}
        if think != 'auto':
            body['thinking'] = {'type': 'enabled' if think == 'enabled' else 'disabled'}
        preview = {**body, 'messages': [body['messages'][0], {'role': 'user', 'content': preview_content}]}
        dump(output / 'sampling.json', {'video': video, 'frames': frames, 'group_counts': dict(counts),
            'timestamp_basis': 'legacy_s2_sampler_estimates_not_decoder_PTS',
            'coverage_gap_sec': engine.coverage_gap(engine.plan_sampling(dur), dur)})
        dump(output / 'model_request.json', body)
        dump(output / 'model_request.preview.json', preview)
        if prepare_only:
            prepared = {'request_id': request['request_id'], 'status': 'prepared_not_submitted',
                'endpoint': endpoint, 'model': model, 'frame_count': len(labels), 'video': video,
                'payload_bytes_utf8': len(json.dumps(body).encode('utf-8')),
                'ground_truth_sent': False, 'network_requests': 0,
                'prompt_version': engine.PROMPT_VERSION, 'sampling_version': engine.SAMPLE_VER,
                'note': 'Local input preview only. No model output exists until an authorized API call succeeds.'}
            dump(output / 'prepared.json', prepared)
            print(json.dumps(prepared, ensure_ascii=False), flush=True)
            return prepared
        print(json.dumps({'stage': 'calling_model', 'request_id': request['request_id'], 'model': model,
            'duration_sec': dur, 'frames': len(labels), 'output_dir': str(output)}, ensure_ascii=False), flush=True)
        raw = stream_call(endpoint, api_key, body)
        dump(output / 'model_response.json', raw)
        with (output / 'model_output.txt').open('x', encoding='utf-8') as f:
            f.write(raw['content'])
        try:
            checks = json.loads(raw['content'])
            errors = validate_checks(checks, set(labels))
        except (ValueError, TypeError):
            checks, errors = None, ['model_output_invalid_json']
        if raw['finish_reason'] != 'stop':
            errors.append('incomplete_model_generation')
        verdict, rule = ('needs_human', 'output_validation_failed') if errors else engine.decide(checks)
        warnings = ['v3db retains its fixed first/last 2-second exemption; not our newer crop policy.',
            'Sampled left-view frames only; timestamps are legacy sampler estimates.',
            'Evidence frame IDs do not constitute precise violation intervals.',
            'Optional audit call omitted; legacy audit does not change the final verdict.']
        response = {'request_id': request['request_id'], 'status': 'completed',
            'verdict': verdict, 'review_status': {'keep':'valid','delete':'invalid','needs_human':'needs_review'}[verdict],
            'triggered_rule': rule or None, 'checks': checks, 'validation_errors': errors,
            'crop': {'supported': False, 'start_sec': None, 'end_sec': None},
            'pipeline': {'prompt_version': engine.PROMPT_VERSION, 'sampling_version': engine.SAMPLE_VER,
                'requested_model': model, 'served_model': raw['model'], 'thinking': think,
                'prompt_sha256': hashlib.sha256((engine.SYSTEM + prompt).encode()).hexdigest(),
                'source_sha256': hashlib.sha256((PROJECT/'qc_run_v3.py').read_bytes()).hexdigest()},
            'video': video, 'frame_count': len(labels), 'usage': raw['usage'],
            'timing': {'model_sec': raw['elapsed_sec'], 'total_sec': round(time.monotonic()-started, 3)},
            'warnings': warnings}
        dump(output / 'output.json', response)
        print(json.dumps({'stage': 'completed', 'verdict': verdict, 'rule': rule,
            'validation_errors': errors, 'timing': response['timing']}, ensure_ascii=False), flush=True)
        return response
    except Exception as exc:
        error = {'request_id': request['request_id'], 'status': 'failed',
            'error_type': type(exc).__name__, 'detail': str(exc).replace(api_key, '[REDACTED]')}
        dump(output / 'error.json', error)
        print(json.dumps(error, ensure_ascii=False), flush=True)
        return error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--env-file', type=Path, default=Path('/data1/zhn/PhiAgent-ego-video-qc/.env'))
    ap.add_argument('--model', default='doubao-seed-2-0-pro-260215')
    ap.add_argument('--think', choices=['auto','enabled','disabled'], default='auto')
    ap.add_argument('--prepare-only', action='store_true', help='Only extract frames and save input; no network requests')
    args = ap.parse_args()
    result = evaluate_one(json.loads(args.input.read_text()), args.output, args.env_file, args.model, args.think, args.prepare_only)
    return 0 if result['status'] in ('completed', 'prepared_not_submitted') else 1


if __name__ == '__main__':
    raise SystemExit(main())
