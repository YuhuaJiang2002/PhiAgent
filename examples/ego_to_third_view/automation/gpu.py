"""Physical UUID leases and same-NUMA selection; never stop another user's process."""
from __future__ import annotations

import csv
import fcntl
import itertools
import os
from pathlib import Path
import subprocess


def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.free,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in csv.reader(raw.splitlines(),skipinitialspace=True):
        index,uuid,name,bus,free,total,util=row
        # nvidia-smi uses 8 domain digits, Linux sysfs normally uses 4.
        bus=bus.lower(); domain,tail=bus.split(':',1)
        numa_file=Path('/sys/bus/pci/devices')/(domain[-4:]+':'+tail)/'numa_node'
        numa=int(numa_file.read_text()) if numa_file.is_file() else -1
        result.append({'index':int(index),'uuid':uuid,'name':name,'free_mib':int(free),
                       'total_mib':int(total),'utilization':int(util),'numa':numa})
    topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
    return result,topo


def select(rows,count,min_free_mib=50000,max_utilization=10,allowed=None):
    candidates=[r for r in rows if r['numa']>=0 and r['free_mib']>=min_free_mib and r['utilization']<=max_utilization
                and (allowed is None or r['uuid'] in allowed or str(r['index']) in allowed)]
    groups=[group for group in itertools.combinations(candidates,count) if len({r['numa'] for r in group})==1]
    if not groups:
        raise RuntimeError('No eligible same-NUMA GPU group; queue/retry later, never silently cross NUMA or evict jobs')
    return list(max(groups,key=lambda group:(min(r['free_mib'] for r in group),-sum(r['utilization'] for r in group))))


class Lease:
    """Cooperative local-process lock, not exclusive ownership of the GPU host."""
    def __init__(self,rows,directory):
        self.files=[]
        Path(directory).mkdir(parents=True,exist_ok=True)
        try:
            for row in sorted(rows,key=lambda x:x['uuid']):
                file=(Path(directory)/(row['uuid']+'.lock')).open('a+')
                try:
                    fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BaseException:
                    file.close()
                    raise
                self.files.append(file)
        except BaseException:
            self.close()
            raise RuntimeError('Selected GPUs leased by another pipeline worker')

    def close(self):
        for file in self.files:
            fcntl.flock(file,fcntl.LOCK_UN)
            file.close()
        self.files=[]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
