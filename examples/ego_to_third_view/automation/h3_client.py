"""Own-process resident worker lifecycle, cooperative leases, bounded request waits."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

from .contracts import write_json
from .gpu import Lease,inventory,select


class H3Client:
    def __init__(self,config,output):
        self.process=None;self.lease=None;self.log=None;self.messages=queue.Queue()
        self.config=dict(config);self.output=Path(output)

    def start(self):
        if self.process is not None:return
        self.output.mkdir(parents=True,exist_ok=False)
        rows,topology=inventory()
        selected=select(rows,self.config.get('num_gpus',4),self.config.get('min_free_mib',50000),
                        self.config.get('max_utilization',10),self.config.get('allowed_gpus'))
        self.lease=Lease(selected,self.config.get('lease_dir','/tmp/phiagent-ego-gpu-leases'))
        try:
            self.config['worker_output']=str(self.output.resolve())
            write_json(self.output/'config.json',self.config)
            write_json(self.output/'gpu_selection.json',{'inventory':rows,'selected':selected,'topology':topology,
                'hostname':socket.gethostname(),'scope':'cooperative lease; unrelated shared-host occupancy can still change'})
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=','.join(row['uuid'] for row in selected),
                       PHI_EGO_H3_BOOTSTRAP='1',PHI_EGO_SOL_ROOT=self.config['sol_root'],
                       H3_REQUEST_EPOCH_FILE=str(self.output/'request_epoch.txt'),
                       H3_SOL_EVENT_LOG=str(self.output/'sol_events_rank{rank}.jsonl'))
            (self.output/'request_epoch.txt').write_text('startup\n')
            # Package root is independent of the caller's working directory.
            package_root=Path(__file__).resolve().parents[1]
            env['PYTHONPATH']=str(package_root)+os.pathsep+env.get('PYTHONPATH','')
            python=self.config.get('python',sys.executable)
            command=[python,'-u','-m','automation.h3_worker',str(self.output/'config.json')]
            write_json(self.output/'launch.json',{'command':command,'selected_uuids':[r['uuid'] for r in selected]})
            self.log=(self.output/'worker.log').open('w')
            self.process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          stdin=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
            def reader():
                for line in self.process.stdout:
                    self.log.write(line);self.log.flush()
                    if line.startswith('PHI_EGO_RPC '):
                        try:self.messages.put(json.loads(line[len('PHI_EGO_RPC '):]))
                        except json.JSONDecodeError:self.messages.put({'status':'error','error':'Malformed worker response'})
                self.messages.put({'status':'error','error':'Worker exited; inspect worker.log'})
            self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
            if self.wait(self.config.get('startup_timeout_s',1800)).get('status')!='ready':
                raise RuntimeError('Worker did not become ready')
        except BaseException:
            self.close();raise

    def wait(self,timeout):
        try:message=self.messages.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('Worker deadline exceeded; inspect logs, not sparse progress-bar cadence')
        if message.get('status')=='error':raise RuntimeError(message['error'])
        return message

    def generate(self,request):
        self.start()
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        return self.wait(self.config.get('request_timeout_s',3600))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"command":"shutdown"}\n');self.process.stdin.flush()
                    self.process.wait(timeout=30)
                except (BrokenPipeError,subprocess.TimeoutExpired):
                    # Only the session we created; never pkill/nvidia-smi reset.
                    try:os.killpg(self.process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:self.process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(self.process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        self.process.wait()
            # A failed coordinator may leave spawned ranks alive in our session.
            # Reap that owned group even when the coordinator already exited.
            try:os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if hasattr(self,'reader'):self.reader.join(timeout=3)
            for stream in (self.process.stdin,self.process.stdout):
                if stream is not None:stream.close()
            self.process=None
        if self.log is not None:self.log.close();self.log=None
        if self.lease is not None:self.lease.close();self.lease=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
