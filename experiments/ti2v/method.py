"""Public entry point for the task-grounded relational-repair candidate.

This is method-side prompt compilation. Video gates, selection and official
metrics remain in the unchanged backend. Running campaigns keep frozen sources.
"""
from copy import deepcopy
from pathlib import Path
from integrations.skilladam_ti2v.backend import save
from integrations.skilladam_ti2v.repair_factorial import (
    literal_relation_precondition, prepare_factorial_repair,
)

METHOD_ID = 'task_grounded_relational_repair_v1'
PRIMARY_ARM = 'combined_extended'
ABLATION_ARM = 'factored_extended'
INPUT_FIELDS = ('instruction', 'initial', 'initial_sha256', 'base_sha256', 'seed')


def propose(backend, row, base_audit, skill, folder, images, positions, *,
            arm=PRIMARY_ARM):
    """Return one bounded edit or the unchanged skill, preserving raw evidence.

    The caller provides the remote backend and sixteen uniformly sampled frames.
    No official scores, future reference or competitor output enters this call.
    The returned skill is a proposal; it is not a quality or physical-success label.
    """
    if arm not in (PRIMARY_ARM, ABLATION_ARM):
        raise ValueError('Use the declared primary method or its factored ablation')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'method-decision.json'
    if destination.exists():
        raise FileExistsError('Preserve the prior decision; use a new call directory')
    audit = deepcopy(base_audit)
    result = prepare_factorial_repair(
        backend, {key: row[key] for key in INPUT_FIELDS}, audit, skill,
        folder, arm, images, positions,
    )
    if audit != base_audit:
        raise ValueError('The proposer mutated its inherited audit')
    result = deepcopy(result)
    result.update(method_id=METHOD_ID, video_quality_improvement_established=False)
    if result['status'] == 'APPLIED':
        precondition = literal_relation_precondition(row['instruction'], result['relation_id'])
        result['literal_task_precondition'] = precondition
        if precondition != 'PASS':
            result.update(status='ABSTAINED', reason='LITERAL_TASK_PRECONDITION_UNRESOLVED',
                          skill=skill, residual='', edits=[])
    # The underlying plan.json is retained verbatim, including rejected proposals.
    save(destination, result)
    return result
