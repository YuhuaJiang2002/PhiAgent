"""Authorized H200 campaign: official SkillAdam core and bounded PhiAgent loop."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback


def bounded_edit(skill, edits):
    if len(edits) > 2:
        raise ValueError('At most two exact whole-line replacements')
    lines = skill.splitlines()
    seen = set()
    for edit in edits:
        old, new = edit['old'], edit['new']
        if not old.strip() or old.startswith('#') or '\n' in old or '\n' in new or lines.count(old) != 1 or old in seen:
            raise ValueError('Edits require unique existing non-heading lines')
        if len(new.split()) > 55:
            raise ValueError('Replacement must be concise')
        seen.add(old); lines[lines.index(old)] = new
    candidate = '\n'.join(line for line in lines if line.strip())+'\n'
    if len(candidate.split()) > 300:
        raise ValueError('Skill too long')
    if any(word in candidate.lower() for word in ('gt_dataset/', 'bleuscore', 'clipscore', 'ndtw', '649524')):
        raise ValueError('No case/evaluator identities in the reusable skill')
    return candidate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--method', choices=['skilladam','phiagent'], required=True)
    args = ap.parse_args(); root, method = args.root, args.method
    assert socket.gethostname() == 'yxys-node-214-41-3-2' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    sys.path.insert(0, str(root/'source')); sys.path.insert(0, str(root/'upstream'))
    from integrations.skilladam_ti2v.adapter import register, METRICS
    from integrations.skilladam_ti2v.backend import TI2VBackend, save, sha, plain, obj, TEXT, choose
    from skilladam.types import BenchmarkCase
    from skilladam.execution.contracts import BackendContext, BatchExecution
    from skilladam.core.feedback_loop import FeedbackLoopConfig, EditBudgetConfig
    from skilladam.methods.skilladam import SkillAdamRunConfig, SkillAdamRunner
    Adapter = register(); cfg = json.loads((root/'protocol.json').read_text())
    for relative, expected in json.loads((root/'source-manifest.json').read_text()).items():
        assert sha(root/relative) == expected, relative
    rows = json.loads((root/'inputs.json').read_text())['records']
    initial = (root/'initial-skill.md').read_text(); out = root/method; out.mkdir(exist_ok=False)
    (out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    state = {'status':'STARTING','method':method,'hostname':socket.gethostname(),'argv':sys.argv,
             'started_at':time.time(),'scope':'Opened DEV only; final optimizer-isolated cases were historically opened',
             'protocol_sha256':sha(root/'protocol.json'),'iterations':[],'stage0':'Explicit shared initial skill; official supported Stage0 bypass',
             'claim_boundary':'No SOTA/public benchmark win/real robot success established'}
    save(out/'state.json',state)
    try:
        backend = TI2VBackend(root,method); adapter = Adapter(root)
        cases = tuple(BenchmarkCase(r['case_id']+':'+str(r['seed']), r, metadata={'split':r['split']})
                      for r in rows if r['seed']==20260910 and r['split'] in ('train','validation'))
        by_id = {c.case_id:c for c in cases}
        context = BackendContext('ewm_ti2v','skilladam',cfg['auxiliary']['served_model'],'none',3,cfg['seed'])
        config = SkillAdamRunConfig(benchmark='ewm_ti2v',split='train',scope=None,
            train_size=2,validation_size=2,stage0_size=2,validation_split='validation',seed=cfg['seed'],
            feedback_loop=FeedbackLoopConfig(max_iterations=2,min_iterations=2,patch_attempts=2,
                edit_budget=EditBudgetConfig(metric_key='ndtw',v_max=.01,base=4,minimum=1,beta=.9)),
            trajectory_compression='deterministic',optimizer_history_turns=2)
        runner = SkillAdamRunner(config=config,adapter=adapter,backend=backend,context=context,output_dir=out/'optimizer')
        batches = tuple(runner._build_sampler(cases))
        save(out/'sampled-batches.json',[asdict(b) for b in batches])
        state['status']='OPTIMIZING';save(out/'state.json',state)
        if method == 'skilladam':
            result = runner.run(cases=cases,initial_skill=initial)
            final = result.final_skill
            state['iterations'] = [r.to_dict() for r in result.feedback_loop.iterations]
            state['stop_reason'] = result.feedback_loop.stop_reason
        else:
            # PhiAgent is an explicit bounded residual optimizer, not a renamed SkillAdam core.
            from skilladam.execution.evaluate import EvaluationRunner
            evaluator = EvaluationRunner()
            final = initial
            def evaluate(ids,skill,stage):
                selected = [by_id[i] for i in ids]
                requests = tuple(adapter.build_rollout_request(c,method='skilladam',split=c.metadata['split'],skill=skill,seed=cfg['seed']) for c in selected)
                raw = backend.execute_batch(BatchExecution(requests,context,stage))
                parsed = [adapter.parse_result(c,r.result) for c,r in zip(selected,raw)]
                from skilladam.types import MetricResult
                per = {r.case_id:dict(r.metadata['metrics']) for r in parsed}
                values = {m:sum(v[m] for v in per.values())/len(per) for m in METRICS}
                return MetricResult(values['ndtw'],values,per,len(per)), parsed
            for index,batch in enumerate(batches,1):
                training,trajectories = evaluate(batch.training_case_ids,final,f'phi-{index}-train')
                baseline,_ = evaluate(batch.validation_case_ids,final,f'phi-{index}-validation-base')
                proposal_schema = obj({'reasoning':TEXT,'edits':{'type':'array','maxItems':2,'items':obj({'old':TEXT,'new':TEXT})}})
                prompt = ('Improve this reusable robot video skill with at most two exact whole-line replacements or deletions. '
                    'Preserve every unrelated line, headings, literal endpoint, identities and visible contact order. '
                    'Prefer correcting a specific observed failure; remove unsupported or redundant motion detail. '
                    'Do not add case IDs, evaluator names, scores, reference futures, or scene-specific memorized answers. '
                    'The generator only receives the task and first image. Choose task-general instructions. '
                    'No extra action or unsupported terminal state. A new line may contain at most55words. '
                    'Return reasoning and edits [{old:exact existing line,new:replacement or empty}].\nSKILL:\n'+final+
                    '\nTRAINING ONLY:\n'+json.dumps([{'trajectory':plain(t.trajectory),'metrics':dict(t.metadata['metrics'])} for t in trajectories]))
                iteration = out/f'iteration-{index:02d}';iteration.mkdir()
                error=''; candidate=final; valid=False
                for attempt in range(2):
                    proposal,usage = backend.query([{'role':'user','content':prompt+'\nPrevious syntax error: '+error}],proposal_schema,f'phi-patch-{index}-{attempt}')
                    save(iteration/f'proposal-{attempt}.json',{'proposal':proposal,'usage':usage})
                    try:
                        candidate=bounded_edit(final,proposal['edits']);valid=True;break
                    except ValueError as exc:
                        error=str(exc);save(iteration/f'rejected-syntax-{attempt}.json',{'error':error})
                (iteration/'candidate-skill.md').write_text(candidate)
                candidate_metric,_ = evaluate(batch.validation_case_ids,candidate,f'phi-{index}-validation-candidate')
                decision = adapter.gate(baseline,candidate_metric)
                if decision.accepted and valid: final=candidate
                record={'iteration':index,'accepted':decision.accepted and valid,'reason':decision.reason,
                        'baseline':asdict(baseline),'candidate':asdict(candidate_metric),'edit_parse_valid':valid,
                        'training_case_ids':list(batch.training_case_ids),'validation_case_ids':list(batch.validation_case_ids),
                        'current_skill_sha256':hashlib.sha256(final.encode()).hexdigest()}
                save(iteration/'decision.json',record);state['iterations'].append(record);save(out/'state.json',state)
                (out/'current-skill.md').write_text(final)
        (out/'final-skill.md').write_text(final)
        state.update(status='FINAL_GENERATION',frozen_final_skill_sha256=sha(out/'final-skill.md'),optimizer_finished_at=time.time())
        save(out/'state.json',state)
        # Freeze before any final score; no test results feed either optimizer.
        with ThreadPoolExecutor(max_workers=3) as executor:
            records=list(executor.map(lambda r:backend.rollout(r,final),rows))
        save(out/'final-uniform-selection.json',{'records':records})
        state['status']='FINAL_SCORING';save(out/'state.json',state)
        score=backend.score(records,'final-uniform');save(out/'final-uniform-scores.json',score)
        if method=='phiagent':
            state['status']='EVENT_SAMPLING_ABLATION';save(out/'state.json',state)
            def reselect(item):
                row,record=item; folder=out/'event-ablation'/(row['case_id'].replace('/','-')+'-'+str(row['seed']))
                base,ub=backend.audit(row,row['base'],folder/'base','event')
                candidate,uc=backend.audit(row,record['candidate'],folder/'candidate','event')
                decision=choose(base,candidate);video=record['candidate'] if decision['selected']=='candidate' else row['base']
                r={**record,'video':video,'sha256':sha(video),'decision':decision,'base_audit':base,'candidate_audit':candidate,
                   'audit_mode':'event','additional_auxiliary_usage':ub+uc,'additional_native_calls':0}
                save(folder/'selection.json',r);return r
            with ThreadPoolExecutor(max_workers=3) as executor:
                event_records=list(executor.map(reselect,zip(rows,records)))
            save(out/'final-event-selection.json',{'records':event_records})
            save(out/'final-event-scores.json',backend.score(event_records,'final-event'))
        state.update(status='DEVELOPMENT_COMPLETE',finished_at=time.time());save(out/'state.json',state)
    except BaseException as error:
        state.update(status='BLOCKED',error=f'{type(error).__name__}: {error}',finished_at=time.time())
        (out/'failure.txt').write_text(traceback.format_exc());save(out/'state.json',state);raise


if __name__=='__main__':
    main()
