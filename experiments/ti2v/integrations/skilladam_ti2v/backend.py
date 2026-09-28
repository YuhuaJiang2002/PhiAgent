"""Remote Qwen/MiniMax execution; official scores enter only after selection."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from collections.abc import Mapping

from .adapter import METRICS


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False)); temp.replace(path)


@contextmanager
def file_lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


TEXT = {'type': 'string'}
GATES = ('task_completion', 'entity_identity', 'camera_preservation', 'motion_continuity', 'visible_action_order')
PLAN_SCHEMA = obj({'terminal_goal': TEXT, 'prompt': TEXT})
PATCH_SCHEMA = obj({'reasoning': TEXT, 'patch': TEXT})
GATE_SCHEMA = obj({'terminal_goal': TEXT, 'gates': obj({g: obj({
    'status': {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']}, 'reason': TEXT,
    'frame_ids': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 16}}
}) for g in GATES})})
JUDGE = '''Judge the literal task from visible evidence only. Image0 is the real initial image;
images1..16 are time-ordered frames across the full generated eight-second clip,
including both endpoints. Each frame has its actual normalized time. Do not infer
hidden motion, contact force, geometry or robot feasibility. UNKNOWN means insufficient
evidence, never PASS. A pickup ends holding; placement ends supported. Do not require
any action, return, release or endpoint absent from the task. Assess task_completion,
entity_identity, camera_preservation, motion_continuity and visible_action_order.
Every PASS or FAIL needs nonempty frame_ids; reasons must cite visible observations.
No future reference, method identity, generating prompt or official score is supplied.
Return terminal_goal and all five gates as JSON. TASK: '''


def choose(base, candidate):
    """Prospectively strengthened v2 policy; sparse audits are not certification."""
    all_pass = all(candidate['gates'][g]['status'] == 'PASS' for g in GATES)
    repaired = [g for g in GATES if base['gates'][g]['status'] == 'FAIL'
                and candidate['gates'][g]['status'] == 'PASS']
    return {'selected': 'candidate' if all_pass and repaired else 'base',
            'candidate_all_pass': all_pass, 'repaired': repaired,
            'physical_success_established': False}


def stage_initial_image(pool, source, expected_sha256):
    """Keep native references inside the backend's declared input boundary."""
    root = Path(pool)/'inputs'; root.mkdir(exist_ok=True)
    target = root/(expected_sha256+'.png')
    assert target.resolve().is_relative_to(root.resolve())
    assert sha(source) == expected_sha256
    if not target.exists():
        shutil.copy2(source, target)
    assert sha(target) == expected_sha256
    return target


class TI2VBackend:
    def __init__(self, root, method):
        self.root = Path(root); self.method = method
        self.cfg = json.loads((self.root / 'protocol.json').read_text())
        assert socket.gethostname() == self.cfg.get('expected_hostname', 'yxys-node-214-41-3-2')
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        self.out = self.root / method / 'execution'; self.out.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.state = {'status': 'READY', 'native_calls': 0, 'auxiliary_calls': 0,
                      'score_jobs': [], 'completed_rollouts': [], 'logical_base_reuses': 0,
                      'hostname': socket.gethostname(), 'started_at': time.time(),
                      'protocol_sha256': sha(self.root / 'protocol.json')}
        self.deadline = time.monotonic() + self.cfg['max_seconds']
        self.update()

    def update(self):
        with self.lock:
            self.state['updated_at'] = time.time(); save(self.out / 'state.json', self.state)

    def public_metadata(self):
        return {'adapter': 'ewm-ti2v-native-v1', 'method_identity': self.method,
                'auxiliary': self.cfg['auxiliary']['model'], 'auxiliary_revision': self.cfg['auxiliary']['revision'],
                'generator': self.cfg['generator']['model'], 'generator_revision': self.cfg['generator']['revision'],
                'scope': 'opened development; no SOTA claim'}

    def query(self, messages, schema, stage, seed=20260916, max_tokens=4096):
        import jsonschema
        with self.lock:
            assert self.state['auxiliary_calls'] < self.cfg['max_auxiliary_calls_per_method']
            self.state['auxiliary_calls'] += 1; number = self.state['auxiliary_calls']; self.update()
        folder = self.out / 'calls' / f'{number:05d}-{stage}'; folder.mkdir(parents=True)
        # Compile locally on the authorized server: no GPU or alternate model involved.
        command = [self.cfg['grammar_python'], '-c',
                   'import sys,json,xgrammar;print(str(xgrammar.Grammar.from_json_schema(json.load(sys.stdin),any_whitespace=False)))']
        grammar = subprocess.check_output(command, input=json.dumps(schema), text=True, timeout=90)
        payload = {'model': self.cfg['auxiliary']['served_model'], 'messages': plain(messages),
                   'temperature': 0, 'seed': seed, 'max_tokens': max_tokens,
                   'chat_template_kwargs': {'enable_thinking': False},
                   'structured_outputs': {'grammar': grammar}}
        save(folder / 'schema.json', schema); save(folder / 'request.json', payload)
        with file_lock(self.root / 'qwen.lock'):
            request = urllib.request.Request(self.cfg['auxiliary']['endpoint'], data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = response.read()
        (folder / 'response.json').write_bytes(raw); response = json.loads(raw)
        assert response['model'] == self.cfg['auxiliary']['served_model']
        assert response['choices'][0]['finish_reason'] == 'stop', response['choices'][0]['finish_reason']
        value = json.loads(response['choices'][0]['message']['content']); jsonschema.validate(value, schema)
        usage = response.get('usage', {}); save(folder / 'usage.json', usage); save(folder / 'value.json', value)
        return value, usage

    def visual_query(self, prompt, images, schema, stage, seed, positions=None):
        content = [{'type': 'text', 'text': prompt}]
        for i, p in enumerate(images):
            label = 'Image0 REAL INITIAL' if i == 0 else f'Image{i} GENERATED normalized_time={positions[i-1]:.5f}'
            content.append({'type': 'text', 'text': label})
            mime = 'image/png' if Path(p).suffix == '.png' else 'image/jpeg'
            content.append({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(Path(p).read_bytes()).decode()}})
        return self.query([{'role': 'user', 'content': content}], schema, stage, seed, 2048)

    def frames(self, video, output, mode='uniform', event_positions=None):
        output.mkdir(parents=True, exist_ok=False)
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                   '-show_entries', 'stream=nb_read_frames,duration', '-of', 'json', str(video)]
        probe = json.loads(subprocess.check_output(command, text=True, timeout=90)); n = int(probe['streams'][0]['nb_read_frames'])
        indices = [round(i * (n-1) / 15) for i in range(16)]
        if mode == 'event' and event_positions:
            chosen = {round(i * (n-1) / 11) for i in range(12)}
            for t in event_positions:
                for offset in (-2, 0, 2):
                    if len(chosen) < 16:
                        chosen.add(max(0, min(n-1, round(t*(n-1)) + offset)))
            for i in indices:
                if len(chosen) < 16:
                    chosen.add(i)
            indices = sorted(chosen)
        assert len(set(indices)) == 16 and indices[0] == 0 and indices[-1] == n-1
        select = '+'.join(f'eq(n\\,{i})' for i in indices)
        cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video), '-vf', f'select={select},scale=320:240',
               '-vsync', 'vfr', str(output / '%02d.jpg')]
        subprocess.run(cmd, check=True, timeout=120)
        files = sorted(output.glob('*.jpg')); assert len(files) == 16
        save(output / 'sampling.json', {'video_sha256': sha(video), 'indices': indices, 'probe': probe,
                                       'commands': [command, cmd], 'frame_hashes': [sha(p) for p in files]})
        return files, [i/(n-1) for i in indices]

    def audit(self, row, video, folder, mode='uniform'):
        folder.mkdir(parents=True, exist_ok=False)
        images, positions = self.frames(video, folder / 'uniform')
        usages = []
        if mode == 'event':
            schema = obj({'event_positions': {'type': 'array', 'maxItems': 2, 'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}, 'evidence': TEXT})
            events, usage = self.visual_query('Locate up to two visibly changing contacts for the literal task. Return their normalized times from the supplied frames. Empty list when uncertain; never guess unseen events. TASK: ' + row['instruction'], [row['initial']] + images, schema, 'event-locator', row['seed'], positions)
            usages.append(usage); save(folder / 'event-locator.json', events)
            if events['event_positions']:
                images, positions = self.frames(video, folder / 'event', 'event', events['event_positions'])
        judge = JUDGE.replace('eight-second', self.cfg.get('duration_phrase', 'eight-second'))
        value, usage = self.visual_query(judge + row['instruction'], [row['initial']] + images, GATE_SCHEMA, 'blind-audit', row['seed'], positions)
        usages.append(usage)
        assert all(g['status'] == 'UNKNOWN' or g['frame_ids'] for g in value['gates'].values())
        save(folder / 'audit.json', value)
        return value, usages

    def native(self, row, prompt, folder):
        with file_lock(self.root / 'native-dispatch.lock'):
            ready = []
            for path in self.cfg['native_pools']:
                pool = Path(path); state = json.loads((pool/'execution.json').read_text())
                if state['status'] == 'FRAMEWORK_POOL_READY':
                    pending = sum(not p.with_suffix('.result.json').exists() for p in (pool/'queue').glob('*.job.json'))
                    ready.append((pending, str(pool)))
            assert ready, 'No ready native pool'
            pool = Path(min(ready)[1]); config = json.loads((pool/'config.json').read_text()); server = config['servers'][0]
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader'], text=True)
            for i, u in zip(server['physical_gpu_indices'], server['physical_gpu_uuids']):
                assert any(line.split(',')[0].strip() == str(i) and u in line for line in inventory.splitlines())
            save(folder/'gpu-selection.json', {'inventory': inventory, 'pool_config_sha256': sha(pool/'config.json'),
                 'physical_gpu_indices': server['physical_gpu_indices'], 'CUDA_VISIBLE_DEVICES': ','.join(server['physical_gpu_uuids']), 'at': time.time()})
            with self.lock:
                assert self.state['native_calls'] < self.cfg['max_native_calls_per_method']
                self.state['native_calls'] += 1; self.update()
            # Native reference loader is deliberately confined to its pool root.
            # Copy the hash-bound first image into that authorized input directory.
            initial = stage_initial_image(pool, row['initial'], row['initial_sha256'])
            job = pool/'queue'/(self.method+'-'+uuid.uuid4().hex+'.job.json')
            request = {'task': 'fl2va', 'prompt': prompt, 'seed': row['seed'],
                       'target': self.cfg.get('media_target', {'aspect_ratio': '4:3', 'duration_seconds': 8.0, 'short_edge': 768}),
                       'num_inference_steps': 50, 'references': [{'path': str(initial), 'kind': 'image', 'role': 'keyframe', 'frame_index': 0}],
                       'stage': self.method, 'requested_media': {'case_id': row['case_id'], 'protocol_sha256': sha(self.root/'protocol.json')}}
            save(folder/'native-request.json', request); save(job, request)
        receipt = job.with_suffix('.result.json')
        while not receipt.exists() and time.monotonic() < self.deadline:
            pool_state = json.loads((pool/'execution.json').read_text())
            if pool_state['status'] in ('BLOCKED','FRAMEWORK_POOL_STOPPED'):
                raise RuntimeError('Native service stopped before producing a receipt: '+str(pool_state.get('error',pool_state['status'])))
            time.sleep(5)
        result = json.loads(receipt.read_text()); save(folder/'native-receipt.json', result)
        assert result['status'] == 'COMPLETE', result
        video = result['result']['path']; assert sha(video) == result['result']['sha256']
        return video

    def rollout(self, row, skill, mode='uniform'):
        key = digest({'row': row, 'skill': skill, 'mode': mode, 'protocol': sha(self.root/'protocol.json')})
        folder = self.out/'rollouts'/key
        # Same immutable inputs within one method may reuse exact complete rollouts.
        if (folder/'selection.json').exists():
            saved = json.loads((folder/'selection.json').read_text())
            assert sha(saved['video']) == saved['sha256']; return saved
        folder.mkdir(parents=True, exist_ok=False)
        assert sha(row['initial']) == row['initial_sha256'] and sha(row['base']) == row['base_sha256']
        plan_instruction = ('Execute this reusable skill on the literal task and initial image. Return only terminal_goal and prompt JSON, at most 180 words in prompt. Immutable constraints: preserve literal named arm/object/action/endpoint, exact first frame and camera, one continuous eight-second clip. Never add an unrequested action or invent hidden state.\nSKILL:\n').replace('eight-second', self.cfg.get('duration_phrase', 'eight-second'))
        plan, usage = self.visual_query(plan_instruction+skill+'\nTASK:\n'+row['instruction'], [row['initial']], PLAN_SCHEMA, 'plan', row['seed'])
        assert 5 <= len(plan['prompt'].split()) <= 180
        save(folder/'plan.json', plan)
        literal = row['instruction'] if self.cfg.get('literal_instruction_passthrough', False) else row['instruction'].split('Keep the first')[0]
        prompt = literal.strip()+' '+plan['prompt']+' Preserve the exact provided first frame and fixed camera.'
        candidate = self.native(row, prompt, folder)
        base_audit, ub = self.audit(row, row['base'], folder/'base-audit', mode)
        candidate_audit, uc = self.audit(row, candidate, folder/'candidate-audit', mode)
        decision = choose(base_audit, candidate_audit); video = candidate if decision['selected'] == 'candidate' else row['base']
        record = {'case_id': row['case_id'], 'seed': row['seed'], 'video': video, 'sha256': sha(video),
                  'candidate': candidate, 'candidate_sha256': sha(candidate), 'skill_sha256': hashlib.sha256(skill.encode()).hexdigest(),
                  'decision': decision, 'base_audit': base_audit, 'candidate_audit': candidate_audit,
                  'trajectory': [{'role': 'user', 'content': row['instruction']},
                                 {'role': 'assistant', 'content': json.dumps({'plan': plan, 'audits': [base_audit, candidate_audit], 'decision': decision})}],
                  'usage': [usage]+ub+uc, 'logical_native_calls': 2, 'real_native_calls': 1, 'audit_mode': mode}
        save(folder/'selection.json', record)
        with self.lock:
            self.state['completed_rollouts'].append(key); self.state['logical_base_reuses'] += 1; self.update()
        return record

    def score(self, records, stage):
        name = self.method+'-'+stage.replace('/', '_')+'-'+uuid.uuid4().hex[:12]
        job = self.root/'score-queue'/(name+'.request.json')
        request = {'job_id': name, 'selected': [{k:r[k] for k in ('case_id','seed','video','sha256')} for r in records],
                   'expected_videos': len(records), 'method': self.method, 'selection_committed_before_score': True}
        save(job, request)
        with self.lock:
            self.state['score_jobs'].append(name); self.update()
        result_path = job.with_name(name+'.result.json')
        while not result_path.exists() and time.monotonic() < self.deadline:
            time.sleep(5)
        result = json.loads(result_path.read_text()); assert result['status'] == 'SCORED', result
        return result

    def execute_batch(self, batch):
        from skilladam.execution.contracts import RawRollout
        requests = list(batch.requests)
        assert all(r.case.metadata['split'] in ('train','validation') for r in requests), 'Final cases never enter optimizer'
        with ThreadPoolExecutor(max_workers=3) as executor:
            records = list(executor.map(lambda r: self.rollout(plain(r.case.payload), r.skill), requests))
        result = self.score(records, batch.stage)
        outputs = []
        for request, record in zip(requests, records):
            key = record['case_id']+':'+str(record['seed'])
            outputs.append(RawRollout(request.case.case_id, {'case_id': request.case.case_id,
                'metrics': result['metrics'][key], 'selected_sha256': record['sha256'], 'trajectory': record['trajectory'],
                'usage': record['usage'], 'official_evidence_sha256': result['official_evidence_sha256']}))
        return tuple(outputs)

    def generate(self, request):
        from skilladam.execution.contracts import GenerationResult, ToolCall
        from skilladam.usage import normalize_usage
        tools = plain(request.metadata.get('tools', ()))
        messages = plain(request.messages)
        if tools:
            variants = []
            for tool in tools:
                function = tool.get('function', tool)
                variants.append(obj({'name': {'type': 'string', 'enum': [function['name']]}, 'arguments': function['parameters']}))
            schema = obj({'text': TEXT, 'tool_calls': {'type': 'array', 'items': {'anyOf': variants}}})
            messages.append({'role':'user','content':'Transport only: encode the official tool calls in this JSON envelope: {"text":"brief rationale","tool_calls":[{"name":"official tool name","arguments":{...}}]}. Use exactly the supplied schemas.\n'+json.dumps(tools)})
            value, usage = self.query(messages, schema, request.stage, max_tokens=8192)
            return GenerationResult(text=value['text'], usage=(normalize_usage(usage),),
                                    tool_calls=tuple(ToolCall(c['name'], c['arguments']) for c in value['tool_calls']))
        value, usage = self.query(messages, PATCH_SCHEMA, request.stage, max_tokens=8192)
        return GenerationResult(text=json.dumps(value), usage=(normalize_usage(usage),))
