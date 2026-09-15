"""Resumable batch DAG with mandatory visual gates and bounded, routed repairs.

Adapters are trusted local argv arrays, never downloaded commands or shell text.
Missing perception/calibration is a visible NEEDS_INPUT, not a whiteboard fallback.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .contracts import (INVARIANTS,ISSUE_STAGE,digest,read_json,validate_events,
                        validate_manifest,validate_review,validate_scene,write_json)
from .gpu import Lease,inventory,select
from .media import prepare_reference,probe,retime,review_packet,run_ffmpeg,same_clock,threeway
from .prompt import build_prompt
from .timeline import event_anchors,pchip_map

ORDER=['reconstruct','stabilize','render','pre_review','generate','post_review','deliver']


def utc():return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def file_bindings(artifacts):
    bindings={key:digest(value) for key,value in artifacts.items() if isinstance(value,str) and Path(value).is_file()}
    if 'scene' in bindings:
        scene_path=Path(artifacts['scene'])
        scene=read_json(scene_path)
        meshes=[o['mesh'] for o in scene.get('objects',[]) if 'mesh' in o]
        meshes += [o['path'] for o in scene.get('static_meshes',[])]
        # Bind mesh-side materials/textures too: editing an external asset must
        # invalidate an accepted review even when scene.json is unchanged.
        suffixes={'.obj','.mtl','.ply','.stl','.glb','.gltf','.bin','.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff','.exr'}
        for mesh in meshes:
            path=(scene_path.parent/mesh).resolve()
            bindings['asset:'+str(path)]=digest(path)
            for dependency in sorted(path.parent.rglob('*')):
                if dependency.is_file() and dependency.suffix.lower() in suffixes:
                    bindings['asset:'+str(dependency.resolve())]=digest(dependency)
    return bindings


def provenance(command):
    record={'command':command,'hostname':socket.gethostname(),'python':sys.version,'started_at':utc()}
    for name,args in [('git_commit',['git','rev-parse','HEAD']),('git_status',['git','status','--porcelain'])]:
        result=subprocess.run(args,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
        record[name]=result.stdout.strip() if not result.returncode else result.stderr.strip()
    record['packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    return record


class Factory:
    def __init__(self,manifest,root,*,resume=False,client_factory=None):
        self.manifest=validate_manifest(manifest);self.root=Path(root).resolve()
        self.client=None;self.client_factory=client_factory
        identity=fingerprint(manifest)
        if self.root.exists():
            if not resume:raise FileExistsError('Existing batch directory; explicitly use --resume')
            if read_json(self.root/'manifest.json')!=manifest:raise ValueError('Manifest changed; create a new batch for changed inputs/policy')
        else:
            self.root.mkdir(parents=True)
            write_json(self.root/'manifest.json',manifest)
            write_json(self.root/'provenance.json',provenance(sys.argv))
        self.identity=identity

    def save(self,folder,state):
        state['updated_at']=utc();write_json(folder/'state.json',state)

    def log(self,folder,event,**fields):
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':utc(),'event':event,**fields},ensure_ascii=False)+'\n')

    def new_stage(self,folder,state,stage):
        count=state['attempts'].get(stage,0)+1
        state['attempts'][stage]=count
        target=folder/f'{stage}-{count:03d}'
        target.mkdir(exist_ok=False)
        self.save(folder,state)
        return target

    def context(self,clip,state):
        settings=json.loads(json.dumps(self.manifest.get('settings',{})))
        for issue in state.get('repair_history',[]):
            if issue['code']=='object_jitter':settings.setdefault('stabilize',{})['smoothing_s']=.075
            if issue['code']=='contact_slip':settings.setdefault('stabilize',{})['ramp_s']=.25
            if issue['code']=='elbow_flip':settings.setdefault('stabilize',{})['elbow_pole_speed_deg_s']=60.
        return {'clip':clip,'source':str(Path(clip['source']).resolve()),'source_sha256':state['source_sha256'],
                'artifacts':state['artifacts'],'settings':settings,'invariants':INVARIANTS,
                'repair_history':state.get('repair_history',[]),'seed':clip.get('seed',2026090631)}

    def run_adapter(self,stage,clip,folder,state):
        adapter=self.manifest.get('adapters',{}).get(stage)
        if adapter is None:
            if stage=='reconstruct' and not clip.get('bundle'):
                raise FileNotFoundError('No scene-general perception adapter or reviewed bundle supplied; source RGB alone is not calibrated 3-D')
            builtin={'reconstruct':'import_bundle','stabilize':'stabilize','render':'render'}[stage]
            adapter={'argv':[self.manifest.get('stage_python',sys.executable),'-m','automation.stages',builtin,
                             '--context','{context}','--output','{output}'],
                     'gpu_count':1 if stage=='render' else 0}
        target=self.new_stage(folder,state,stage)
        context=self.context(clip,state)
        command=[part.replace('{context}',str(target/'context.json')).replace('{output}',str(target))
                 .replace('{source}',context['source']) for part in adapter['argv']]
        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
        env['PYTHONHASHSEED']=str(context['seed']);env['CUDA_VISIBLE_DEVICES']=''
        lease=None
        if adapter.get('gpu_count',0):
            rows,topology=inventory()
            chosen=select(rows,adapter['gpu_count'],adapter.get('min_free_mib',4000),
                          adapter.get('max_utilization',10),adapter.get('allowed_gpus'))
            lease=Lease(chosen,self.manifest.get('lease_dir','/tmp/phiagent-ego-gpu-leases'))
            context['gpu_selection']={'inventory':rows,'selected':chosen,'topology':topology}
            env['CUDA_VISIBLE_DEVICES']=','.join(row['uuid'] for row in chosen)
        write_json(target/'context.json',context)
        record=provenance(command);record['gpu_selection']=context.get('gpu_selection')
        write_json(target/'execution.json',record)
        self.log(folder,'stage_started',stage=stage,directory=str(target))
        try:
            with (target/'stage.log').open('w') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=adapter.get('timeout_s',3600))
            record.update(returncode=result.returncode,finished_at=utc());write_json(target/'execution.json',record)
            if result.returncode:raise RuntimeError(f'{stage} failed: {target}/stage.log')
            result=read_json(target/'result.json')
            self.validate_stage(stage,result,state)
            state['artifacts'][stage]=result
            state['hashes'][stage]=file_bindings(result)
            self.log(folder,'stage_completed',stage=stage,artifacts=state['hashes'][stage])
        finally:
            if lease:lease.close()

    def validate_stage(self,stage,result,state):
        required={'reconstruct':{'scene','motion','events','ego'},
                  'stabilize':{'scene','motion','events','ego','rig','geometry_report'},
                  'render':{'scene','motion','events','ego','rig','geometry_report','sim'}}[stage]
        for key in required:
            if key not in result or not Path(result[key]).is_file():raise ValueError(f'{stage}: missing {key}')
        scene=validate_scene(read_json(result['scene']))
        if scene['source_sha256']!=state['source_sha256']:raise ValueError('Wrong source identity')
        events=read_json(result['events'])
        if events.get('source_sha256')!=state['source_sha256']:raise ValueError('Wrong event source')
        validate_events(events['events'],events['frames']/events['fps'])
        from .stages import load_motion
        load_motion(result['motion'],scene,events)
        if stage in ('stabilize','render'):
            import numpy as np
            from .geometry import audit_rig
            audit_rig(np.load(result['rig'],allow_pickle=False),scene,events['frames'])
        if stage=='render':
            sim=probe(result['sim']);ego=probe(result['ego'])
            if not same_clock(sim,ego) or sim['frames']!=events['frames'] or abs(sim['fps']-events['fps'])>1e-6:
                raise ValueError('Source/SIM/event clocks differ before DiT')
            report=read_json(result['geometry_report'])['rig']
            if any(abs(v)>1e-9 for v in report['root_translation_range_m']):raise ValueError('Torso translated')
            for side in ['left','right']:
                if side in report and max(report[side]['max_upper_length_error_m'],report[side]['max_forearm_length_error_m'])>1e-5:
                    raise ValueError('Limb length changed')

    def binding(self,state,phase):
        stages=['reconstruct','stabilize','render']+(['generate'] if phase=='post' else [])
        if phase=='post' and 'align' in state['artifacts']:stages.append('align')
        return {'source_sha256':state['source_sha256'],'manifest_sha256':self.identity,
                'frames':probe(state['artifacts']['render']['sim'])['frames'],
                'artifacts':{stage:file_bindings(state['artifacts'][stage]) for stage in stages}}

    def review(self,clip,folder,state,phase):
        key=phase+'_review';binding=self.binding(state,phase)
        pending=state.get('pending_review')
        if pending and pending['phase']==phase and pending['binding']==binding:
            target=Path(pending['directory'])
        else:
            target=self.new_stage(folder,state,key)
            rendered=state['artifacts']['render']
            videos={'ego':rendered['ego'],'sim':rendered['sim']}
            if phase=='post':videos['dit']=state['artifacts'].get('align',state['artifacts']['generate'])['dit']
            events=read_json(rendered['events'])['events']
            packet=review_packet(videos,target/'frames',events)
            write_json(target/'review_request.json',{'phase':phase,'binding':binding,
                'packet':str(target/'frames/packet.json'),'geometry_report':rendered['geometry_report'],
                'scene':rendered['scene'],'events':rendered['events'],
                'review_output':str(target/'review.json'),
                'instructions':'Inspect actual images and event neighborhoods. All checks required. No numerical proxy may substitute for visual review.',
                'required_checks':['actor_origin','limb_lengths','torso','contacts','identity','occlusion','camera','timing','background']})
            state['pending_review']={'phase':phase,'binding':binding,'directory':str(target)}
            self.save(folder,state)
        reviewer=self.manifest.get('adapters',{}).get('review_'+phase)
        if not (target/'review.json').is_file() and reviewer:
            command=[part.replace('{request}',str(target/'review_request.json')).replace('{output}',str(target/'review.json'))
                     for part in reviewer['argv']]
            with (target/'reviewer.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=reviewer.get('timeout_s',900))
            if result.returncode:raise RuntimeError('Visual reviewer command failed')
        if not (target/'review.json').is_file():
            state['status']='NEEDS_REVIEW';return False
        review=validate_review(read_json(target/'review.json'),binding,phase)
        self.log(folder,'visual_review',phase=phase,decision=review['decision'],review_sha256=digest(target/'review.json'))
        if review['decision']=='accept':
            state['reviews'][phase]={'path':str(target/'review.json'),'sha256':digest(target/'review.json'),'binding':binding}
            state.pop('pending_review',None);return True
        if review['decision']=='needs_input':
            state['status']='NEEDS_INPUT';return False
        if state['repairs']>=self.manifest.get('max_repairs',2):
            state['status']='REPAIR_BUDGET_EXHAUSTED';return False
        issues=review['issues'];stages={ISSUE_STAGE[i['code']] for i in issues}
        if 'review' in stages:
            state['status']='NEEDS_INPUT';return False
        repair='align' if stages=={'align'} else min(stages-{'align'},key=ORDER.index)
        if repair=='reconstruct' and clip.get('bundle') and not self.manifest.get('adapters',{}).get('reconstruct'):
            state.update(status='NEEDS_INPUT',error='The supplied reconstruction needs correction; configure a repair-capable frontend or provide a corrected bundle in a new batch')
            return False
        if phase=='pre' and repair in ('generate','align'):
            raise ValueError('Pre-DiT failures must be fixed upstream, not hidden by generation')
        state['repair_history'].extend(issues);state['repairs']+=1
        state.pop('pending_review',None)
        if repair=='align':
            generated=state['artifacts'].get('align',state['artifacts']['generate'])
            meta=probe(generated['dit']);sim_events=read_json(state['artifacts']['render']['events'])['events']
            anchors=event_anchors(sim_events,review['dit_events'],meta['frames'],meta['fps'])
            mapping=pchip_map(anchors,meta['frames'])
            align=self.new_stage(folder,state,'align')
            retime(generated['dit'],align/'dit.mp4',mapping)
            write_json(align/'mapping.json',mapping)
            state['artifacts']['align']={'dit':str(align/'dit.mp4'),'mapping':str(align/'mapping.json')}
            state['hashes']['align']=file_bindings(state['artifacts']['align'])
            state['step']='post_review'
        else:
            for stage in ORDER[ORDER.index(repair):]+['align']:
                state['artifacts'].pop(stage,None);state['hashes'].pop(stage,None)
            if ORDER.index(repair)<=ORDER.index('render'):state['reviews'].pop('pre',None)
            state['reviews'].pop('post',None);state['step']=repair
        state['status']='RUNNING';self.save(folder,state)
        return False

    def generate(self,clip,folder,state):
        # Review must still match exactly before any expensive GPU request.
        if state['reviews']['pre']['binding']!=self.binding(state,'pre'):
            raise ValueError('Pre-DiT review invalidated')
        rendered=state['artifacts']['render'];meta=probe(rendered['sim'])
        target=self.new_stage(folder,state,'generate')
        if clip.get('existing_dit'):
            if not clip.get('existing_dit_sha256') or digest(clip['existing_dit'])!=clip['existing_dit_sha256']:
                raise ValueError('Existing DiT must be explicitly hash-bound')
            source=clip['existing_dit']
            generation_record={'mode':'explicit_existing_dit','source_sha256':digest(source)}
        else:
            if not clip.get('appearance') or not Path(clip['appearance']).is_file():raise FileNotFoundError('Reviewed appearance reference required')
            padding=prepare_reference(rendered['sim'],target/'reference.mp4')
            write_json(target/'reference_alignment.json',padding)
            scene=read_json(rendered['scene']);events=read_json(rendered['events'])['events']
            request={'id':clip['id'],'sim':rendered['sim'],'reference':str(target/'reference.mp4'),
                     'appearance':clip['appearance'],'seed':clip.get('seed',2026090631),
                     'prompt':build_prompt(scene,events,meta['duration_s']),'output':str(target/'inference'),
                     'steps':self.manifest.get('h3',{}).get('steps',24)}
            if self.client is None:
                if self.client_factory:self.client=self.client_factory()
                else:
                    from .h3_client import H3Client
                    self.client=H3Client(self.manifest['h3'],self.root/f'worker-{time.time_ns()}')
            generation_record=self.client.generate(request);source=generation_record['candidate']
        raw=probe(source)
        if raw['frames']<meta['frames'] or abs(raw['fps']-meta['fps'])>1e-6:
            raise ValueError('H3 output is too short or wrong fps; never speed up/pad to conceal failure')
        if (raw['width'],raw['height'])!=(meta['width'],meta['height']):raise ValueError('H3 output projection size differs from control')
        run_ffmpeg(['-i',source,'-vf',f'trim=end_frame={meta["frames"]},setpts=PTS-STARTPTS',
                    '-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',target/'dit.mp4'])
        write_json(target/'generation.json',generation_record)
        state['artifacts']['generate']={'dit':str(target/'dit.mp4'),'generation':str(target/'generation.json')}
        state['hashes']['generate']=file_bindings(state['artifacts']['generate'])

    def advance(self,clip):
        folder=self.root/clip['id'];folder.mkdir(exist_ok=True)
        path=folder/'state.json'
        state=read_json(path) if path.is_file() else {'clip_id':clip['id'],'status':'RUNNING','step':'reconstruct',
            'source_sha256':digest(clip['source']),'artifacts':{},'hashes':{},'reviews':{},'attempts':{},
            'repairs':0,'repair_history':[]}
        try:
            if digest(clip['source'])!=state['source_sha256']:raise ValueError('Source changed; new batch required')
            for stage,hashes in state['hashes'].items():
                if file_bindings(state['artifacts'][stage])!=hashes:raise ValueError(f'Artifact changed after {stage}; new batch required')
            if state['status']=='COMPLETE':return state
            if state['status']=='REPAIR_BUDGET_EXHAUSTED':return state
            state['status']='RUNNING'
            while state['status']=='RUNNING':
                stage=state['step']
                if stage in ('reconstruct','stabilize','render'):self.run_adapter(stage,clip,folder,state)
                elif stage in ('pre_review','post_review'):
                    if not self.review(clip,folder,state,stage.split('_')[0]):
                        if state['status']=='RUNNING':continue
                        break
                elif stage=='generate':self.generate(clip,folder,state)
                elif stage=='deliver':
                    if state['reviews']['post']['binding']!=self.binding(state,'post'):raise ValueError('Post review invalidated')
                    target=self.new_stage(folder,state,'deliver');rendered=state['artifacts']['render']
                    dit=state['artifacts'].get('align',state['artifacts']['generate'])['dit']
                    threeway(rendered['ego'],rendered['sim'],dit,target/'threeway.mp4')
                    state['artifacts']['deliver']={'threeway':str(target/'threeway.mp4'),'dit':dit,'sim':rendered['sim'],'ego':rendered['ego']}
                    state['hashes']['deliver']=file_bindings(state['artifacts']['deliver'])
                    state['status']='COMPLETE';self.log(folder,'delivered',hashes=state['hashes']['deliver']);break
                state['step']=ORDER[ORDER.index(stage)+1]
                self.save(folder,state)
        except FileNotFoundError as exc:state.update(status='NEEDS_INPUT',error=str(exc))
        except RuntimeError as exc:
            state.update(status='WAITING_GPU' if 'same-NUMA' in str(exc) or 'leased' in str(exc) else 'FAILED',error=str(exc))
            if state.get('step')=='generate' and self.client:
                self.client.close();self.client=None
        except Exception as exc:state.update(status='FAILED',error=repr(exc))
        self.save(folder,state)
        return state

    def run(self):
        summary={}
        try:
            for clip in self.manifest['clips']:
                try:
                    result=self.advance(clip);summary[clip['id']]={'status':result['status'],'step':result['step'],'error':result.get('error')}
                except Exception as exc:summary[clip['id']]={'status':'FAILED','error':repr(exc)}
                write_json(self.root/'summary.json',summary)
        finally:
            if self.client:self.client.close()
        return summary
