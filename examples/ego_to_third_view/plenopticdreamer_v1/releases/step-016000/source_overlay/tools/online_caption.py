"""Generate missing deterministic captions on the CP leader before VAE work.

The VLM is kept in CPU memory and occupies the GPU only during captioning.
Used-caption hashes are checkpointed so resume rejects changed annotations.
"""
import plenoptic_paths as layout
from datetime import datetime
import fcntl
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
import prepare_plenoptic as prepare
from caption_plenoptic import caption_one,caption_spec
from plenoptic_data import rooted,caption_key


class OnlineCaptioner:
    def __init__(self,directory):
        self.directory=rooted(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.spec,self.spec_hash=caption_spec()
        self.model=self.processor=None
        self.used={}
        self.generated=0
        self.args=SimpleNamespace(frames=8,max_new_tokens=384)

    def cached_path(self,video,expected=None):
        path=self.directory/(caption_key(video)+'.json')
        if expected is None:
            return path
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Checkpoint caption digest is invalid')
        if path.is_file():
            data=json.loads(path.read_text())
            if (data.get('video')==video and data.get('spec_hash')==self.spec_hash
                    and hashlib.sha256(data.get('caption','').encode()).hexdigest()==expected):
                return path
        # Independent CP leaders can have different deterministic-model
        # results in old local caches. Preserve each checkpoint's exact text.
        variant=self.directory/'.versions'/caption_key(video)/(expected+'.json')
        return variant if variant.is_file() else path

    def restore(self,ledger):
        for video,expected in ledger.items():
            data=json.loads(self.cached_path(video,expected).read_text())
            if data['video']!=video or data['spec_hash']!=self.spec_hash:
                raise ValueError('Checkpoint caption identity changed')
            if hashlib.sha256(data['caption'].encode()).hexdigest()!=expected:
                raise ValueError('A previously used training caption changed since checkpoint')
        self.used=dict(ledger)

    @torch.no_grad()
    def __call__(self,sample):
        video=sample['caption_video']
        path=self.directory/(caption_key(video)+'.json')
        started=time.monotonic()
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.cached_path(video,self.used.get(video))
            if video in self.used and not path.is_file():
                raise ValueError('A checkpoint caption is missing from the shared caption cache')
            if not path.exists():
                # Preserve training dropout/noise RNG even if a cache miss
                # constructs a model, or Transformers changes generation RNG.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    if self.model is None:
                        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
                        local=layout.rooted(prepare.JOBS['reason'][3])
                        self.processor=AutoProcessor.from_pretrained(local,local_files_only=True,
                            min_pixels=224*224,max_pixels=336*336)
                        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(local,
                            local_files_only=True,torch_dtype=torch.bfloat16,
                            attn_implementation='flash_attention_2',device_map={'':torch.cuda.current_device()}).eval().requires_grad_(False)
                    try:
                        self.model.to(torch.cuda.current_device())
                        indices=np.linspace(0,80,8).round().astype(int).tolist()
                        caption,raw=caption_one(self.model,self.processor,video,indices,self.args)
                    finally:
                        self.model.cpu()
                        torch.cuda.empty_cache()
                record=dict(video=video,scene_id=sample['scene_id'],dataset=sample['dataset'],
                    split=sample['split'],caption=caption,raw_output=raw,frame_indices=indices,
                    spec_hash=self.spec_hash,generation=self.spec,generation_details=self.args.last_generation,
                    created_at=datetime.now().astimezone().isoformat())
                prepare.save_json(path,record)
                self.generated+=1
                print(json.dumps(dict(event='online_caption',video=video,
                    seconds=round(time.monotonic()-started,3),generated=self.generated)),flush=True)
            data=json.loads(path.read_text())
            if data['video']!=video or data['spec_hash']!=self.spec_hash or data['split']!=sample['split'] or not data['caption'].strip():
                raise ValueError('Cached caption does not match the pinned generation specification')
        digest=hashlib.sha256(data['caption'].encode()).hexdigest()
        if video in self.used and self.used[video]!=digest:
            raise ValueError('An immutable training caption was modified')
        self.used[video]=digest
        return data['caption']
