#!/usr/bin/env python3
"""Generate resumable, per-video captions with the local Cosmos-Reason1 VLM.

torchrun may run independent GPU workers; no distributed process group needed.
Each worker owns disjoint videos and writes atomic records (no shared JSONL).
"""
import plenoptic_paths as layout
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
import re
import time

import prepare_plenoptic as prepare
from local_checkpoints import configure_local_checkpoints
from plenoptic_data import caption_key, decode_video, read_scenes, rooted

PROMPT = ('Write a factual video caption in English using 50 to 80 words in a single paragraph. '
          'Describe the setting, visible people or objects, their actions and motion, and camera motion. '
          'Only describe what is visible; do not infer intentions, identities, or hidden events. '
          'Return only the description, without analysis, headings, or XML tags.')
RECOVERY_POLICY = 'caption_length_recovery_v1'


def caption_spec(frames=8,max_new_tokens=384):
    spec=dict(model=prepare.JOBS['reason'][0],revision=prepare.JOBS['reason'][2],
              prompt=PROMPT,frames=frames,max_new_tokens=max_new_tokens,do_sample=False,
              decode_hw=[336,336],version=2)
    return spec,hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()


def caption_from_output(raw, ended):
    """Keep legacy successful captions; recover usable text when the token cap wins."""
    caption = re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip()
    answer = re.search(r'<answer>(.*?)</answer>',caption,re.S)
    if answer:
        caption = answer[1].strip()
    elif '<answer>' in caption and not ended:
        caption = caption.split('<answer>',1)[1].strip()
    if not caption or '<think>' in caption or len(caption.split())<6:
        raise ValueError('No usable caption text (empty, too short, or unfinished reasoning)')
    if ended:
        return caption,None
    # Missing EOS alone is not a failed annotation. Prefer complete sentences
    # from the model's actual description and discard the unfinished tail.
    caption = ' '.join(caption.split())
    prefixes = [caption[:m.end()].strip() for m in re.finditer(r'[.!?]["\u201d\u2019\']*(?=\s|$)',caption)]
    prefixes = [p for p in prefixes if len(p.split())>=6]
    short = [p for p in prefixes if len(p.split())<=80]
    if short:
        return short[-1],'complete_sentence_prefix'
    if prefixes:
        return prefixes[0],'complete_sentence_prefix'
    # A factual phrase is still a valid text condition even without punctuation.
    return ' '.join(caption.split()[:80]),'word_prefix'


def caption_one(model, processor, video, indices, args):
    import torch
    frames = decode_video(video, indices, (336, 336)).permute(1, 2, 3, 0).numpy()
    args.last_generation = None
    for attempt in range(2):
        prompt = PROMPT if attempt==0 else (
            PROMPT+' Do not think aloud. Give only two short factual sentences, at most 60 words.')
        limit = args.max_new_tokens if attempt==0 else max(args.max_new_tokens,768)
        messages = [{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
        text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs = processor(text=[text],videos=[frames],padding=True,return_tensors='pt',
                           fps=(args.frames-1)*15/80).to(model.device)
        with torch.inference_mode():
            result = model.generate(**inputs,max_new_tokens=limit,do_sample=False,temperature=None)
        generated = result[:,inputs['input_ids'].shape[1]:]
        raw = processor.batch_decode(generated,skip_special_tokens=True)[0]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else (eos or [])
        ended = int(result[0,-1]) in eos
        try:
            caption,recovery = caption_from_output(raw,ended)
        except ValueError as exc:
            print(json.dumps(dict(event='caption_retry' if attempt==0 else 'caption_unusable',
                video=str(video),attempt=attempt+1,ended=ended,generated_tokens=generated.shape[1],
                error=str(exc))),flush=True)
            if attempt==1:
                raise ValueError(f'No usable caption after two attempts for {video}: {raw}') from exc
            del result,generated,inputs
            continue
        details=dict(policy=RECOVERY_POLICY,attempt=attempt+1,ended=ended,
                     generated_tokens=generated.shape[1],max_new_tokens=limit,prompt=prompt,
                     recovery=recovery,caption_words=len(caption.split()))
        args.last_generation=details
        if recovery or attempt:
            print(json.dumps(dict(event='caption_recovered',video=str(video),**details)),flush=True)
        return caption,raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', nargs='+', default=['datasets/manifests/syncam_scenes.jsonl'])
    parser.add_argument('--output', default='datasets/captions/reason1')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--shard-id',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    args = parser.parse_args()
    if args.frames < 2 or args.frames % 2:
        parser.error('--frames must be a positive even number >= 2')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    if not 0 <= args.shard_id < args.num_shards:
        parser.error('Invalid external shard index/count')
    rank,world=rank+args.shard_id*world,world*args.num_shards
    device = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    configure_local_checkpoints(require=['reason'])
    scenes = read_scenes(args.manifests, args.split)
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    jobs = [(scene, video) for scene in scenes for _, video in sorted(scene['videos'].items())]
    if args.max_videos is not None:
        jobs = jobs[:args.max_videos]
    output = rooted(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spec,spec_hash=caption_spec(args.frames,args.max_new_tokens)
    jobs = [(s, v) for i, (s, v) in enumerate(jobs) if i % world == rank]
    pending = []
    for scene, video in jobs:
        path = output / (caption_key(video) + '.json')
        if path.exists():
            prior = json.loads(path.read_text())
            if prior['spec_hash'] != spec_hash or prior['video'] != video or not prior['caption'].strip():
                raise RuntimeError(f'Incompatible caption record: {path}; choose a new output directory')
        else:
            pending.append((scene, video, path))
    print(json.dumps(dict(rank=rank, assigned=len(jobs), pending=len(pending))), flush=True)
    if not pending:
        return
    local = layout.rooted(prepare.JOBS['reason'][3])
    processor = AutoProcessor.from_pretrained(local, local_files_only=True, min_pixels=224*224,
                                             max_pixels=336*336)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
        device_map={'': device}).eval().requires_grad_(False)
    indices = np.linspace(0, 80, args.frames).round().astype(int).tolist()
    failed, consecutive_failures = 0, 0
    for count, (scene, video, path) in enumerate(pending):
        started = time.monotonic()
        # Prevent duplicate processes from overwriting the same video's result.
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                continue
            try:
                caption,raw = caption_one(model,processor,video,indices,args)
            except (RuntimeError,ValueError) as exc:
                failed += 1
                consecutive_failures += 1
                prepare.save_json(path.with_suffix('.error.json'),dict(video=video,spec_hash=spec_hash,
                    error=str(exc),created_at=datetime.now().astimezone().isoformat()))
                print(json.dumps(dict(rank=rank,failed_video=video,error=str(exc))),flush=True)
                torch.cuda.empty_cache()
                if consecutive_failures >= 3:
                    raise RuntimeError('Three consecutive failures; check the caption worker log') from exc
                continue
            consecutive_failures = 0
            prepare.save_json(path, dict(video=video, scene_id=scene['scene_id'], dataset=scene['dataset'],
                split=scene['split'], caption=caption, raw_output=raw, frame_indices=indices,
                spec_hash=spec_hash, generation=spec, generation_details=args.last_generation,
                created_at=datetime.now().astimezone().isoformat()))
            path.with_suffix('.error.json').unlink(missing_ok=True)
        print(json.dumps(dict(rank=rank, completed=count+1, total=len(pending), video=video,
                              seconds=round(time.monotonic()-started, 2), caption=caption)), flush=True)
    print(json.dumps(dict(rank=rank,status='partial' if failed else 'completed',failed=failed)),flush=True)


if __name__ == '__main__':
    main()
