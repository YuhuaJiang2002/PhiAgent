"""A small, inspectable StateGraph runtime for reproducible research workflows.

The public API intentionally resembles the useful center of LangGraph: named
nodes, typed state updates, conditional edges, thread-scoped checkpoints,
streaming events, and resumable interrupts.  It is not a compatibility shim.
Keeping the runtime in the Python standard library lets every PhiAgent workflow
be imported and tested without CUDA, PyTorch, a simulator, or a checkpoint.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .checkpoint import CheckpointStore, MemoryCheckpointer, json_clone


START = "__start__"
END = "__end__"


class GraphStatus(str, Enum):
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class GraphDefinitionError(ValueError):
    """Raised when a graph cannot be compiled safely."""


class GraphExecutionError(RuntimeError):
    """Raised after a node failure has been persisted."""


class ThreadStateError(RuntimeError):
    """Raised when an invocation conflicts with persisted thread state."""


class _NoValue:
    pass


_NO_VALUE = _NoValue()


@dataclass(frozen=True)
class Command:
    """A node transition or an external resume/retry command."""

    update: Mapping[str, Any] = field(default_factory=dict)
    goto: str | None = None
    resume: Any = field(default=_NO_VALUE, repr=False)
    retry: bool = False

    @property
    def has_resume(self) -> bool:
        return self.resume is not _NO_VALUE


@dataclass(frozen=True)
class Interrupt:
    interrupt_id: str
    node: str
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "interrupt_id": self.interrupt_id,
            "node": self.node,
            "payload": json_clone(self.payload),
        }


class _InterruptRaised(Exception):
    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(interrupt.interrupt_id)
        self.interrupt = interrupt


@dataclass(frozen=True)
class GraphResult:
    thread_id: str
    status: GraphStatus
    state: dict[str, Any]
    next_node: str
    step: int
    checkpoint_id: int
    interrupts: tuple[Interrupt, ...] = ()


class NodeContext:
    """Stable per-node runtime context.

    Nodes should put side effects below ``node_dir``.  A node is restarted from
    its beginning after an interrupt, so work before ``interrupt`` must be
    idempotent, just as it is in durable workflow engines.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        node: str,
        step: int,
        execution_id: int,
        artifact_root: Path | None,
        resume_value: Any = _NO_VALUE,
    ) -> None:
        self.thread_id = thread_id
        self.node = node
        self.step = step
        self.execution_id = execution_id
        self._artifact_root = artifact_root
        self._resume_value = resume_value

    @property
    def node_dir(self) -> Path:
        if self._artifact_root is None:
            raise RuntimeError("this graph invocation did not configure artifact_root")
        path = (
            self._artifact_root
            / self.thread_id
            / f"{self.step:06d}-{self.node}"
            / f"attempt-{self.execution_id:08d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def interrupt(self, payload: Any, *, key: str = "review") -> Any:
        """Pause durably or consume the value supplied by ``Command(resume=...)``."""

        if self._resume_value is not _NO_VALUE:
            value = self._resume_value
            self._resume_value = _NO_VALUE
            return value
        digest = hashlib.sha256(
            f"{self.thread_id}:{self.node}:{self.step}:{key}".encode("utf-8")
        ).hexdigest()[:20]
        raise _InterruptRaised(
            Interrupt(interrupt_id=digest, node=self.node, payload=json_clone(payload))
        )


Node = Callable[..., Mapping[str, Any] | Command | None]
Router = Callable[..., str]
Reducer = Callable[[Any, Any], Any]


class StateGraph:
    """Builder for a deterministic, checkpointed state machine."""

    def __init__(self, *, reducers: Mapping[str, Reducer] | None = None) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, str] = {}
        self._conditionals: dict[str, tuple[Router, dict[str, str]]] = {}
        self._reducers = dict(reducers or {})

    def add_node(self, name: str, action: Node) -> "StateGraph":
        self._validate_node_name(name)
        if name in self._nodes:
            raise GraphDefinitionError(f"duplicate node: {name}")
        if not callable(action):
            raise TypeError("node action must be callable")
        self._nodes[name] = action
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        if source == END or target == START:
            raise GraphDefinitionError(f"invalid edge {source!r} -> {target!r}")
        if source in self._edges or source in self._conditionals:
            raise GraphDefinitionError(f"node {source!r} already has an outgoing transition")
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: Router,
        path_map: Mapping[str, str],
    ) -> "StateGraph":
        if source in self._edges or source in self._conditionals:
            raise GraphDefinitionError(f"node {source!r} already has an outgoing transition")
        if not path_map:
            raise GraphDefinitionError("conditional edges require a non-empty path_map")
        if not callable(router):
            raise TypeError("conditional router must be callable")
        self._conditionals[source] = (router, dict(path_map))
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    def compile(
        self,
        *,
        name: str = "workflow",
        version: str = "1.0.0",
        checkpointer: CheckpointStore | None = None,
        max_steps: int = 256,
    ) -> "CompiledGraph":
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._validate()
        return CompiledGraph(
            name=name,
            version=version,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditionals=dict(self._conditionals),
            reducers=dict(self._reducers),
            checkpointer=checkpointer or MemoryCheckpointer(),
            max_steps=max_steps,
        )

    @staticmethod
    def _validate_node_name(name: str) -> None:
        if not name or name in {START, END}:
            raise GraphDefinitionError(f"invalid node name: {name!r}")
        if any(character.isspace() for character in name):
            raise GraphDefinitionError("node names cannot contain whitespace")

    def _validate(self) -> None:
        if START not in self._edges:
            raise GraphDefinitionError("graph requires an entry point")
        known = set(self._nodes) | {START, END}
        destinations = set(self._edges.values())
        for _, path_map in self._conditionals.values():
            destinations.update(path_map.values())
        unknown = sorted(destinations - known)
        if unknown:
            raise GraphDefinitionError(f"transitions reference unknown nodes: {unknown}")
        sources = set(self._edges) | set(self._conditionals)
        invalid_sources = sorted(sources - known)
        if invalid_sources:
            raise GraphDefinitionError(f"transitions start at unknown nodes: {invalid_sources}")
        missing_transitions = sorted(set(self._nodes) - sources)
        if missing_transitions:
            raise GraphDefinitionError(
                f"nodes require explicit outgoing transitions: {missing_transitions}"
            )
        reachable = {START}
        frontier = [START]
        while frontier:
            source = frontier.pop()
            targets: Iterable[str]
            if source in self._edges:
                targets = (self._edges[source],)
            elif source in self._conditionals:
                targets = self._conditionals[source][1].values()
            else:
                targets = ()
            for target in targets:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        unreachable = sorted(set(self._nodes) - reachable)
        if unreachable:
            raise GraphDefinitionError(f"unreachable nodes: {unreachable}")
        if END not in reachable:
            raise GraphDefinitionError("graph has no statically reachable END transition")


class CompiledGraph:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        nodes: Mapping[str, Node],
        edges: Mapping[str, str],
        conditionals: Mapping[str, tuple[Router, dict[str, str]]],
        reducers: Mapping[str, Reducer],
        checkpointer: CheckpointStore,
        max_steps: int,
    ) -> None:
        self.name = name
        self.version = version
        self._nodes = dict(nodes)
        self._edges = dict(edges)
        self._conditionals = dict(conditionals)
        self._reducers = dict(reducers)
        self.checkpointer = checkpointer
        self.max_steps = max_steps

    def invoke(
        self,
        input_state: Mapping[str, Any] | Command,
        *,
        config: Mapping[str, Any],
    ) -> GraphResult:
        result = None
        for event in self.stream(input_state, config=config):
            if event["type"] in {"graph_completed", "graph_interrupted"}:
                result = GraphResult(
                    thread_id=str(event["thread_id"]),
                    status=GraphStatus(str(event["status"])),
                    state=dict(event["state"]),
                    next_node=str(event["next_node"]),
                    step=int(event["step"]),
                    checkpoint_id=int(event["checkpoint_id"]),
                    interrupts=tuple(
                        Interrupt(
                            interrupt_id=str(item["interrupt_id"]),
                            node=str(item["node"]),
                            payload=item["payload"],
                        )
                        for item in event.get("interrupts", [])
                    ),
                )
        if result is None:
            raise GraphExecutionError("graph ended without a terminal event")
        return result

    def stream(
        self,
        input_state: Mapping[str, Any] | Command,
        *,
        config: Mapping[str, Any],
    ) -> Iterable[dict[str, Any]]:
        thread_id = str(config.get("thread_id", ""))
        if not thread_id:
            raise ValueError("config.thread_id is required for every invocation")
        artifact_root_raw = config.get("artifact_root")
        artifact_root = (
            Path(str(artifact_root_raw)).expanduser().resolve()
            if artifact_root_raw is not None
            else None
        )
        checkpoint_metadata = config.get("checkpoint_metadata", {})
        if not isinstance(checkpoint_metadata, Mapping):
            raise TypeError("config.checkpoint_metadata must be an object")

        latest = self.checkpointer.load_latest(thread_id)
        resume_value = _NO_VALUE
        if isinstance(input_state, Command):
            if input_state.update or input_state.goto:
                raise ThreadStateError("external Command supports only resume or retry")
            if input_state.has_resume == input_state.retry:
                raise ThreadStateError("external Command must choose exactly one of resume or retry")
            if latest is None:
                raise ThreadStateError(f"thread {thread_id!r} has no checkpoint to resume")
            if input_state.has_resume:
                if latest["status"] != GraphStatus.WAITING.value:
                    raise ThreadStateError("only a WAITING thread accepts Command(resume=...)")
                resume_value = input_state.resume
            elif latest["status"] != GraphStatus.ERROR.value:
                raise ThreadStateError("only an ERROR thread accepts Command(retry=True)")
            state = dict(latest["state"])
            current = str(latest["next_node"])
            step = int(latest["step"])
        else:
            if latest is not None:
                raise ThreadStateError(
                    f"thread {thread_id!r} already exists with status {latest['status']}"
                )
            state = json_clone(dict(input_state))
            current = self._edges[START]
            step = 0
            latest = self.checkpointer.save(
                thread_id,
                state=state,
                next_node=current,
                status=GraphStatus.RUNNING.value,
                step=step,
                metadata=checkpoint_metadata,
            )
            yield self._event("checkpoint", latest)

        while current != END:
            if step >= self.max_steps:
                error = {
                    "type": "StepLimitExceeded",
                    "message": f"graph exceeded max_steps={self.max_steps}",
                }
                latest = self.checkpointer.save(
                    thread_id,
                    state=state,
                    next_node=current,
                    status=GraphStatus.ERROR.value,
                    step=step,
                    error=error,
                    metadata=checkpoint_metadata,
                )
                yield self._event("graph_error", latest)
                raise GraphExecutionError(error["message"])
            context = NodeContext(
                thread_id=thread_id,
                node=current,
                step=step,
                execution_id=int(latest["checkpoint_id"]),
                artifact_root=artifact_root,
                resume_value=resume_value,
            )
            resume_value = _NO_VALUE
            yield {
                "type": "node_started",
                "thread_id": thread_id,
                "node": current,
                "step": step,
            }
            try:
                raw_result = self._call(self._nodes[current], state, context)
                update, goto = self._normalize_node_result(raw_result)
                next_state = self._merge_state(state, update)
                next_node = goto or self._route(current, next_state, context)
                if next_node != END and next_node not in self._nodes:
                    raise GraphExecutionError(
                        f"node {current!r} requested unknown transition {next_node!r}"
                    )
            except _InterruptRaised as raised:
                latest = self.checkpointer.save(
                    thread_id,
                    state=state,
                    next_node=current,
                    status=GraphStatus.WAITING.value,
                    step=step,
                    interrupt=raised.interrupt.to_dict(),
                    metadata=checkpoint_metadata,
                )
                yield self._event(
                    "graph_interrupted",
                    latest,
                    interrupts=[raised.interrupt.to_dict()],
                )
                return
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc), "node": current}
                latest = self.checkpointer.save(
                    thread_id,
                    state=state,
                    next_node=current,
                    status=GraphStatus.ERROR.value,
                    step=step,
                    error=error,
                    metadata=checkpoint_metadata,
                )
                yield self._event("graph_error", latest)
                raise GraphExecutionError(
                    f"node {current!r} failed: {type(exc).__name__}: {exc}"
                ) from exc

            state = next_state
            current = next_node
            step += 1
            status = GraphStatus.COMPLETED if current == END else GraphStatus.RUNNING
            latest = self.checkpointer.save(
                thread_id,
                state=state,
                next_node=current,
                status=status.value,
                step=step,
                metadata=checkpoint_metadata,
            )
            yield {
                "type": "node_completed",
                "thread_id": thread_id,
                "node": context.node,
                "next_node": current,
                "step": step,
                "update": update,
                "checkpoint_id": latest["checkpoint_id"],
            }

        yield self._event("graph_completed", latest)

    def describe(self) -> dict[str, Any]:
        conditional = {
            source: dict(path_map)
            for source, (_, path_map) in sorted(self._conditionals.items())
        }
        return {
            "name": self.name,
            "version": self.version,
            "entry_point": self._edges[START],
            "nodes": sorted(self._nodes),
            "edges": dict(sorted(self._edges.items())),
            "conditional_edges": conditional,
            "reducers": sorted(self._reducers),
            "max_steps": self.max_steps,
        }

    def mermaid(self) -> str:
        lines = ["flowchart TD"]
        for source, target in sorted(self._edges.items()):
            lines.append(f'    {self._mermaid_id(source)}["{source}"] --> {self._mermaid_id(target)}["{target}"]')
        for source, (_, path_map) in sorted(self._conditionals.items()):
            for label, target in sorted(path_map.items()):
                lines.append(
                    f'    {self._mermaid_id(source)} -->|"{label}"| '
                    f'{self._mermaid_id(target)}["{target}"]'
                )
        return "\n".join(lines) + "\n"

    def get_state(self, thread_id: str) -> dict[str, Any] | None:
        return self.checkpointer.load_latest(thread_id)

    def get_state_history(self, thread_id: str) -> list[dict[str, Any]]:
        return self.checkpointer.history(thread_id)

    @staticmethod
    def _call(action: Callable[..., Any], state: Mapping[str, Any], context: NodeContext) -> Any:
        parameters = [
            parameter
            for parameter in inspect.signature(action).parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        frozen_state = MappingProxyType(json_clone(dict(state)))
        if len(parameters) == 1:
            return action(frozen_state)
        if len(parameters) == 2:
            return action(frozen_state, context)
        raise TypeError("workflow callables must accept (state) or (state, context)")

    @staticmethod
    def _normalize_node_result(
        result: Mapping[str, Any] | Command | None,
    ) -> tuple[dict[str, Any], str | None]:
        if result is None:
            return {}, None
        if isinstance(result, Command):
            if result.has_resume or result.retry:
                raise TypeError("node Command cannot contain resume or retry")
            return json_clone(dict(result.update)), result.goto
        if not isinstance(result, Mapping):
            raise TypeError("node result must be a mapping, Command, or None")
        return json_clone(dict(result)), None

    def _merge_state(self, state: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(state)
        for key, value in update.items():
            if key in merged and key in self._reducers:
                merged[key] = self._reducers[key](merged[key], value)
            else:
                merged[key] = value
        return json_clone(merged)

    def _route(self, node: str, state: Mapping[str, Any], context: NodeContext) -> str:
        if node in self._edges:
            return self._edges[node]
        router, path_map = self._conditionals[node]
        label = str(self._call(router, state, context))
        if label not in path_map:
            raise GraphExecutionError(
                f"router for {node!r} returned {label!r}; expected one of {sorted(path_map)}"
            )
        return path_map[label]

    @staticmethod
    def _event(event_type: str, checkpoint: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "type": event_type,
            "thread_id": checkpoint["thread_id"],
            "status": checkpoint["status"],
            "state": checkpoint["state"],
            "next_node": checkpoint["next_node"],
            "step": checkpoint["step"],
            "checkpoint_id": checkpoint["checkpoint_id"],
            **extra,
        }

    @staticmethod
    def _mermaid_id(name: str) -> str:
        return "n_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
