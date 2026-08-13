"""Built-in workflow registry."""

from __future__ import annotations

from typing import Callable

from .checkpoint import CheckpointStore
from .core import CompiledGraph
from .flower import FLOWER_WORKFLOW_NAME, build_flower_long_video_workflow


WorkflowFactory = Callable[..., CompiledGraph]

_WORKFLOWS: dict[str, WorkflowFactory] = {
    FLOWER_WORKFLOW_NAME: build_flower_long_video_workflow,
}


def workflow_names() -> tuple[str, ...]:
    return tuple(sorted(_WORKFLOWS))


def create_workflow(name: str, *, checkpointer: CheckpointStore | None = None) -> CompiledGraph:
    try:
        factory = _WORKFLOWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown workflow {name!r}; available={list(workflow_names())}") from exc
    return factory(checkpointer=checkpointer)
