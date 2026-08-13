import json
import sys

import pytest

from phiagent.workflows import (
    END,
    Command,
    CommandSpec,
    GraphExecutionError,
    GraphStatus,
    JsonFileCheckpointer,
    StateGraph,
    SubprocessNode,
)


def test_state_graph_routes_reduces_and_persists(tmp_path):
    graph = StateGraph(reducers={"events": lambda left, right: [*left, *right]})
    graph.add_node("measure", lambda state: {"score": state["value"] * 2, "events": ["m"]})
    graph.add_node("accept", lambda state: {"decision": "accepted", "events": ["a"]})
    graph.add_node("reject", lambda state: {"decision": "rejected", "events": ["r"]})
    graph.set_entry_point("measure")
    graph.add_conditional_edges(
        "measure", lambda state: "pass" if state["score"] >= 4 else "fail", {"pass": "accept", "fail": "reject"}
    )
    graph.add_edge("accept", END)
    graph.add_edge("reject", END)
    checkpointer = JsonFileCheckpointer(tmp_path / "checkpoints")
    app = graph.compile(name="quality", checkpointer=checkpointer)

    result = app.invoke(
        {"value": 2, "events": []},
        config={"thread_id": "candidate-1", "artifact_root": tmp_path / "artifacts"},
    )

    assert result.status is GraphStatus.COMPLETED
    assert result.state["decision"] == "accepted"
    assert result.state["events"] == ["m", "a"]
    assert [row["status"] for row in app.get_state_history("candidate-1")] == [
        "RUNNING",
        "RUNNING",
        "COMPLETED",
    ]


def test_interrupt_resumes_same_node_from_start(tmp_path):
    calls = []

    def review(state, context):
        calls.append(state["candidate"])
        verdict = context.interrupt({"question": "pass?"})
        return {"verdict": verdict}

    graph = StateGraph()
    graph.add_node("review", review)
    graph.set_entry_point("review")
    graph.add_edge("review", END)
    app = graph.compile(checkpointer=JsonFileCheckpointer(tmp_path / "checkpoints"))

    waiting = app.invoke({"candidate": "x"}, config={"thread_id": "review-1"})
    assert waiting.status is GraphStatus.WAITING
    assert waiting.next_node == "review"
    assert len(waiting.interrupts) == 1

    completed = app.invoke(Command(resume="PASS"), config={"thread_id": "review-1"})
    assert completed.status is GraphStatus.COMPLETED
    assert completed.state["verdict"] == "PASS"
    assert calls == ["x", "x"]


def test_error_checkpoint_can_retry(tmp_path):
    should_fail = {"value": True}

    def unstable(state):
        if should_fail["value"]:
            raise RuntimeError("transient")
        return {"recovered": True}

    graph = StateGraph()
    graph.add_node("unstable", unstable)
    graph.set_entry_point("unstable")
    graph.add_edge("unstable", END)
    app = graph.compile(checkpointer=JsonFileCheckpointer(tmp_path / "checkpoints"))

    with pytest.raises(GraphExecutionError, match="transient"):
        app.invoke({}, config={"thread_id": "retry-1"})
    assert app.get_state("retry-1")["status"] == "ERROR"

    should_fail["value"] = False
    result = app.invoke(Command(retry=True), config={"thread_id": "retry-1"})
    assert result.state["recovered"] is True


def test_subprocess_node_records_shell_free_command(tmp_path):
    def command_builder(state, context):
        return CommandSpec(argv=(sys.executable, "-c", "print('workflow-ok')"))

    graph = StateGraph()
    graph.add_node("command", SubprocessNode(command_builder))
    graph.set_entry_point("command")
    graph.add_edge("command", END)
    app = graph.compile(checkpointer=JsonFileCheckpointer(tmp_path / "checkpoints"))

    result = app.invoke(
        {},
        config={"thread_id": "command-1", "artifact_root": tmp_path / "artifacts"},
    )

    record = result.state["last_command"]
    assert record["returncode"] == 0
    assert record["gpu_selection"] is None
    command_record = json.loads(
        (
            tmp_path
            / "artifacts"
            / "command-1"
            / "000000-command"
            / "attempt-00000000"
            / "command.json"
        ).read_text()
    )
    assert command_record["argv"][:2] == [sys.executable, "-c"]
