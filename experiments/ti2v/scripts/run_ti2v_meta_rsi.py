"""Reserved remote MetaRSI-inspired skill/policy adaptation and final scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    if socket.gethostname()!='yxys-node-214-41-3-2' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Declared H200 CPU controller with GPU visibility disabled required')
    sys.path.insert(0,str(root/'source'))
    from phiagent.experiments import audit_experiment_timestamps
    from integrations.skilladam_ti2v.backend import TI2VBackend, save, sha
    from integrations.skilladam_ti2v.meta_rsi import run_meta_rsi
    audit=audit_experiment_timestamps(root,require_reservation=True)
    save(root/'timestamp-audit.json',audit.to_dict())
    if not audit.valid:raise RuntimeError(audit.issues)
    for name,expected in json.loads((root/'source-manifest.json').read_text()).items():
        if sha(root/name)!=expected:raise RuntimeError('Frozen source mismatch: '+name)
    cfg=json.loads((root/'config.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    records=json.loads((root/'inputs.json').read_text())['records']
    assert len(records)==60 and len({r['case_id'] for r in records})==20
    method='phiagent_metarsi';out=root/method;out.mkdir(exist_ok=False)
    state={'status':'WAITING_FOR_NATIVE_POOLS','at':time.time(),'hostname':socket.gethostname(),
           'argv':sys.argv,'scope':cfg['scope'],'final_outputs_expected':60,
           'protocol_sha256':sha(root/'protocol.json'),'source_manifest_sha256':sha(root/'source-manifest.json')}
    save(out/'state.json',state)
    try:
        deadline=time.monotonic()+2400
        while time.monotonic()<deadline:
            pools={p:json.loads((Path(p)/'execution.json').read_text()).get('status')
                   if (Path(p)/'execution.json').exists() else 'STARTING' for p in protocol['native_pools']}
            if all(v=='FRAMEWORK_POOL_READY' for v in pools.values()):break
            if any(v in ('BLOCKED','FRAMEWORK_POOL_STOPPED') for v in pools.values()):
                raise RuntimeError('A frozen native pool failed: '+json.dumps(pools))
            time.sleep(10)
        else:raise TimeoutError('Native pool startup deadline')
        backend=TI2VBackend(root,method)
        initial=(root/'initial-skill.md').read_text();policy=(root/'initial-policy.md').read_text()
        optrows=[r for r in records if r['seed']==cfg['optimizer_seed']]
        state.update(status='META_RSI_OPTIMIZATION',started_at=time.time());save(out/'state.json',state)
        release=run_meta_rsi(backend,optrows,initial,policy,sha(root/'protocol.json'),out/'optimizer',terms=cfg['terms'])
        (out/'final-skill.md').write_text(release['skill']);(out/'final-policy.md').write_text(release['policy'])
        state.update(status='FINAL_GENERATION',frozen_skill_sha256=sha(out/'final-skill.md'),
                     frozen_policy_sha256=sha(out/'final-policy.md'),optimizer_finished_at=time.time())
        save(out/'state.json',state)
        with ThreadPoolExecutor(max_workers=4) as executor:
            selected=list(executor.map(lambda r:backend.rollout(r,release['skill']),records))
        save(out/'final-selection.json',{'records':selected,'release':release,'all_selections_committed':True})
        state.update(status='FINAL_SCORING',selected_outputs=len(selected));save(out/'state.json',state)
        result=backend.score(selected,'final60');save(out/'final-scores.json',result)
        # Read-only reporting after release; these final scores never reach the optimizer.
        keys={r['case_id']+':'+str(r['seed']) for r in records}
        if set(result['metrics'])!=keys:raise RuntimeError('Incomplete final score coverage')
        from phiagent.harness.video_meta_rsi import METRICS
        means={m:sum(v[m] for v in result['metrics'].values())/len(keys) for m in METRICS}
        isolated={r['case_id']+':'+str(r['seed']) for r in records if r['split']=='test'}
        isolated_means={m:sum(result['metrics'][k][m] for k in isolated)/len(isolated) for m in METRICS}
        save(out/'summary.json',{'all60_development_means':means,'optimizer_isolated24_means':isolated_means,
            'optimizer_isolated_cases_historically_opened':True,'official_evidence_sha256':result['official_evidence_sha256'],
            'new_native_calls':backend.state['native_calls'],'auxiliary_calls':backend.state['auxiliary_calls'],
            'full_public_benchmark':False,'SOTA_established':False})
        state.update(status='DEVELOPMENT_COMPLETE',means=means,finished_at=time.time());save(out/'state.json',state)
    except BaseException as error:
        state.update(status='BLOCKED',error=f'{type(error).__name__}: {error}',finished_at=time.time())
        save(out/'state.json',state);(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        # Only this run's owned pools; never stop another campaign's services.
        for pool in protocol['native_pools']:
            p=Path(pool)
            if p.parent==root:(p/'STOP').touch()


if __name__=='__main__':main()
