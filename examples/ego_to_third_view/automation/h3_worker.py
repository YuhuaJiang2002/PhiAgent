"""JSONL worker: one model load, serial independent requests, explicit cache epochs.

Start through H3Client so GPU UUID selection and leases precede every CUDA import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import uuid
import math
import importlib.metadata
import socket

from .contracts import digest, read_json, write_json

# multiprocessing spawn imports this module on each rank. Registration must happen
# there as well as in the parent, before DiffGenerator resolves its model class.
if os.getenv('PHI_EGO_H3_BOOTSTRAP')=='1':
    from .sol_runtime import install
    install()


class ResidentGenerator:
    def __init__(self,factory,options):
        self.generator=None;self.factory=factory;self.options=options;self.loads=0

    def generate(self,payload):
        start=time.monotonic();load=0.
        if self.generator is None:
            self.generator=self.factory(**self.options);self.loads+=1
            load=time.monotonic()-start
        request_start=time.monotonic()
        result=self.generator.generate(sampling_params_kwargs=payload)
        return result,{'loading_seconds':load,'request_wall_seconds':time.monotonic()-request_start,
                       'worker_total_loads':self.loads,'cold_total_seconds':time.monotonic()-start}

    def close(self):
        if self.generator is not None:self.generator.shutdown();self.generator=None


def payload_for(request):
    from .media import probe
    sim=probe(request['sim'])
    if abs(sim['fps']-24)>1e-6 or not 4<=sim['duration_s']<=15:
        raise ValueError('Pinned H3 supports 4–15s at 24fps; longer clips need a separately reviewed continuity adapter')
    divisor=math.gcd(sim['width'],sim['height'])
    return {'prompt':request['prompt'],'task':'ref2va',
            'conditions':[{'type':'image','uri':Path(request['appearance']).resolve().as_uri(),'role':'reference'},
                          {'type':'video','uri':Path(request['reference']).resolve().as_uri(),'role':'reference'}],
            'target':{'short_edge':min(sim['width'],sim['height']),
                      'aspect_ratio':f'{sim["width"]//divisor}:{sim["height"]//divisor}',
                      'duration_seconds':sim['frames']/sim['fps']},
            'num_outputs_per_prompt':1,'num_inference_steps':request.get('steps',24),
            'flow_shift':request.get('flow_shift',12.),'audio_flow_shift':request.get('audio_flow_shift',3.),
            'quality':'lossless','seed':request['seed'],'output_path':request['output'],
            'output_file_name':'candidate.mp4','save_output':True,'return_file_paths_only':True}


def main():
    if os.getenv('PHI_EGO_H3_BOOTSTRAP')!='1' or not os.getenv('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Use the batch GPU launcher')
    config=read_json(sys.argv[1]);root=Path(config['worker_output'])
    write_json(root/'runtime_provenance.json',{'hostname':socket.gethostname(),'python':sys.version,
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'note':'SGLang quality=lossless is the native sampling selector; Sol/FBC are separately audited approximations, not a lossless-video claim.'})
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    options={'local_mode':True,'model_path':config['model_path'],'model_variant':'ref2va',
             'num_gpus':config.get('num_gpus',4),'tp_size':1,'ulysses_degree':config.get('num_gpus',4),
             'enable_cfg_parallel':False,'performance_mode':'speed','use_fsdp_inference':True,
             'text_encoder_cpu_offload':False,'enable_torch_compile':False,'regional_compile':False,
             'server_warmup':False,'master_port':config.get('master_port',30592)}
    generator=ResidentGenerator(DiffGenerator.from_pretrained,options)
    print('PHI_EGO_RPC '+json.dumps({'status':'ready'}),flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('command')=='shutdown':break
            out=Path(request['output']);out.mkdir(parents=True,exist_ok=False)
            epoch=request['id']+':'+uuid.uuid4().hex
            Path(os.environ['H3_REQUEST_EPOCH_FILE']).write_text(epoch+'\n')
            write_json(out/'request.json',request)
            payload=payload_for(request);write_json(out/'payload.json',payload)
            try:
                result,timing=generator.generate(payload)
                if result is None or isinstance(result,list):raise RuntimeError('Expected one H3 output')
                from .sol_runtime import audit
                evidence=audit(root,epoch,options['num_gpus'])
                write_json(out/'acceleration.json',evidence)
                if not evidence['verified_sol_and_cache_execution']:
                    raise RuntimeError('Missing real per-request Sol/FBC/gate execution evidence')
                record={'status':'ok','candidate':result.output_file_path,'timing':timing,
                        'epoch':epoch,'acceleration':str(out/'acceleration.json')}
                write_json(out/'result.json',record)
                print('PHI_EGO_RPC '+json.dumps(record),flush=True)
            except BaseException as exc:
                write_json(out/'failure.json',{'error':repr(exc),'epoch':epoch})
                print('PHI_EGO_RPC '+json.dumps({'status':'error','error':repr(exc)}),flush=True)
                # Never reuse a distributed worker after a partially failed request.
                raise
    finally:
        generator.close()


if __name__=='__main__':main()
