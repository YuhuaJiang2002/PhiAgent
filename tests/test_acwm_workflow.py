from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from phiagent.acwm.adapters import (
    ACWMRenderRequest,
    ACWMRenderResult,
    BWMConfig,
    BWMRenderer,
    OSCAR_COSMOS_REASON_REVISION,
    OSCARConfig,
    OSCAR_MODEL_REVISION,
    OSCAR_REPOSITORY_COMMIT,
    OSCAR_WAN_VAE_REVISION,
    OSCARRenderer,
    _pair_backend_results,
)
from phiagent.acwm.schema import ACWMActionCondition, ACWMCase, ActionRepresentation
from phiagent.agent.acwm import (
    ACWMProposal,
    ACWMScorecard,
    ACWMThresholds,
    ACWMEvaluationRequest,
    AgenticACWMController,
    AgenticACWMRequest,
    CommandACWMEvaluator,
)
from scripts.build_oscar_bowl_skeleton_conditions import (
    repaired_wrist_source_xy,
    skeleton_row,
    vertical_template_xy,
)
from scripts.evaluate_acwm_candidate import action_adherence_score
from scripts.run_agentic_acwm import initial_proposals, select_cases


def _file(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _action(tmp_path: Path, *, representation=ActionRepresentation.KINEMATIC_SKELETON_2D):
    visual = _file(tmp_path / "skeleton.mp4")
    frame = "camera:test_pixels" if representation.requires_camera_frame else "robot_base:test"
    channels = (
        ("x", "y")
        if representation.requires_camera_frame
        else tuple(f"action_{index}" for index in range(14))
    )
    values = (
        tuple(0.0 for _ in channels),
        tuple(1.0 for _ in channels),
    )
    return ACWMActionCondition(
        label="slide-left",
        instruction="move left",
        timeline="start; move; hold",
        representation=representation,
        coordinate_frame=frame,
        timestamps_s=(0.0, 0.1),
        channels=channels,
        values=values,
        visual_condition=visual if representation.requires_camera_frame else None,
    )


def _case(tmp_path: Path, action: ACWMActionCondition | None = None) -> ACWMCase:
    return ACWMCase(
        case_id="slide-left",
        first_frame=_file(tmp_path / "first.png"),
        source_video=_file(tmp_path / "source.mp4"),
        action=action or _action(tmp_path),
        prompt="A robot moves a bowl left.",
    )


def test_action_contract_rejects_screen_pixels_as_eef(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="robot-base"):
        ACWMActionCondition(
            label="slide-left",
            instruction="move left",
            timeline="move",
            representation=ActionRepresentation.EEF_ABSOLUTE,
            coordinate_frame="camera:test_pixels",
            timestamps_s=(0.0, 0.1),
            channels=tuple(f"action_{index}" for index in range(14)),
            values=(tuple([0.0] * 14), tuple([1.0] * 14)),
        )


def test_action_json_round_trip_resolves_relative_visual_condition(tmp_path: Path) -> None:
    action = _action(tmp_path)
    path = tmp_path / "condition" / "action.json"
    action.to_json(path)

    loaded = ACWMActionCondition.from_json(path)

    assert loaded == action
    assert json.loads(path.read_text())["visual_condition"].startswith("..")


def test_oscar_and_bwm_route_only_native_representations(tmp_path: Path) -> None:
    oscar = OSCARRenderer(OSCARConfig(tmp_path / "oscar", tmp_path / "checkpoint"))
    bwm = BWMRenderer(
        BWMConfig(
            tmp_path / "bwm",
            tmp_path / "base",
            tmp_path / "bwm.safetensors",
            tmp_path / "stat.json",
        )
    )
    skeleton_case = _case(tmp_path)
    eef_case = _case(tmp_path, _action(tmp_path, representation=ActionRepresentation.EEF_ABSOLUTE))

    assert oscar.supports(skeleton_case).supported
    assert not bwm.supports(skeleton_case).supported
    assert not oscar.supports(eef_case).supported
    assert bwm.supports(eef_case).supported


def test_oscar_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "oscar"
    python = repository / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    (repository / ".phiagent-source-revision").write_text(OSCAR_REPOSITORY_COMMIT + "\n")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / ".phiagent-model-revision").write_text(OSCAR_MODEL_REVISION + "\n")
    renderer = OSCARRenderer(OSCARConfig(repository, checkpoint))

    report = renderer.preflight(select_cuda_device=False)

    assert report["python"] == str(python.absolute())
    assert report["python"] != str(python.resolve())


def test_oscar_preflight_pins_offline_runtime_dependencies(tmp_path: Path) -> None:
    repository = tmp_path / "oscar"
    python = repository / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    (repository / ".phiagent-source-revision").write_text(OSCAR_REPOSITORY_COMMIT + "\n")
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / ".phiagent-model-revision").write_text(OSCAR_MODEL_REVISION + "\n")
    cosmos = tmp_path / "runtime" / "Cosmos-Reason1-7B"
    cosmos.mkdir(parents=True)
    (cosmos / ".phiagent-model-revision").write_text(OSCAR_COSMOS_REASON_REVISION + "\n")
    vae = _file(tmp_path / "runtime" / "Wan2.1_VAE.pth")
    (vae.parent / ".phiagent-wan-vae-revision").write_text(OSCAR_WAN_VAE_REVISION + "\n")
    renderer = OSCARRenderer(
        OSCARConfig(
            repository,
            checkpoint,
            cosmos_reason_path=cosmos,
            wan_vae_path=vae,
            offline=True,
        )
    )

    report = renderer.preflight(select_cuda_device=False)

    runtime = report["model"]["runtime_dependencies"]
    assert runtime["cosmos_reason_revision"] == OSCAR_COSMOS_REASON_REVISION
    assert runtime["wan_vae_revision"] == OSCAR_WAN_VAE_REVISION
    assert report["model"]["offline"] is True


def test_skeleton_compiler_preserves_terminal_direction() -> None:
    left = skeleton_row((200.0, 300.0))
    right = skeleton_row((600.0, 300.0))
    lifted = skeleton_row((400.0, 100.0))

    assert left[6] < right[6]
    assert lifted[7] < left[7]
    assert len(left) == 12


def test_terminal_wrist_repair_is_smooth_and_frame_explicit() -> None:
    samples = [{"hand_contact_xy": [520.0, 285.0]} for _ in range(81)]

    before = repaired_wrist_source_xy(samples, 19, (100.0, 309.0))
    middle = repaired_wrist_source_xy(samples, 39, (100.0, 309.0))
    terminal = repaired_wrist_source_xy(samples, 80, (100.0, 309.0))

    assert before == (520.0, 285.0)
    assert terminal == pytest.approx((140.0, 329.6))
    assert terminal[0] < middle[0] < before[0]


def test_vertical_template_keeps_target_x_and_borrows_lift_y() -> None:
    assert vertical_template_xy((610.0, 330.0), (390.0, 120.0)) == (610.0, 120.0)


def test_action_score_rewards_directional_progress_not_exact_endpoint() -> None:
    right, right_progress, _, _ = action_adherence_score((174.0, 19.0), (197.0, -43.0))
    lift, lift_progress, _, _ = action_adherence_score((-2.0, -164.0), (-5.0, -202.0))
    stalled, stalled_progress, _, _ = action_adherence_score((-138.0, 19.0), (-11.0, 5.0))

    assert right >= 0.75
    assert lift >= 0.75
    assert right_progress > 1.0
    assert lift_progress > 1.0
    assert stalled < 0.5
    assert stalled_progress < 0.15


def test_prompt_repair_batch_keeps_one_case_and_same_seed(tmp_path: Path) -> None:
    left = _case(tmp_path / "left")
    right = ACWMCase(
        case_id="slide-right",
        first_frame=left.first_frame,
        source_video=left.source_video,
        action=left.action,
        prompt="move right",
    )
    selected = select_cases((left, right), ["slide-right"])
    renderers = {"oscar": _FakeRenderer()}

    proposals = initial_proposals(
        selected,
        renderers,
        seed=20260810,
        prompt_suffixes=["keep one articulated hand", "preserve continuous joint motion"],
    )

    assert [item.case_id for item in proposals] == ["slide-right", "slide-right"]
    assert {item.seed for item in proposals} == {20260810}
    assert len({item.prompt_suffix for item in proposals}) == 2


def test_backend_result_pairing_preserves_duplicate_case_order(tmp_path: Path) -> None:
    case = _case(tmp_path)
    requests = tuple(
        ACWMRenderRequest(
            case=case,
            output=tmp_path / f"candidate-{index}.mp4",
            experiment_dir=tmp_path / "run",
        )
        for index in range(2)
    )
    payload = [
        {"case_id": "slide-left", "output": "first.mp4"},
        {"case_id": "slide-left", "output": "second.mp4"},
    ]

    pairs = _pair_backend_results(requests, payload)

    assert [item[1]["output"] for item in pairs] == ["first.mp4", "second.mp4"]


def test_case_selection_rejects_unknown_case(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown AC-WM cases"):
        select_cases((_case(tmp_path),), ["lift-up"])


def test_command_evaluator_uses_rendered_canonical_action_json(tmp_path: Path) -> None:
    case = _case(tmp_path)
    condition = tmp_path / "backend-runs" / "oscar" / "conditions" / "slide-left.json"
    case.action.to_json(condition)
    output = _file(tmp_path / "generated.mp4")
    metadata = tmp_path / "generated.json"
    metadata.write_text(json.dumps({"condition": str(condition.resolve())}) + "\n")
    evaluator = CommandACWMEvaluator(
        (
            sys.executable,
            "-c",
            (
                "import json,sys; print(json.dumps({{"
                "'action_adherence': 1, 'embodiment_consistency': 1, "
                "'object_interaction': 1, 'temporal_consistency': 1, "
                "'background_consistency': 1, 'human_review_passed': None, "
                "'evidence': sys.argv[1]}}))"
            ),
            "{condition}",
        )
    )

    score = evaluator.evaluate(
        ACWMEvaluationRequest(
            case=case,
            result=ACWMRenderResult(
                backend="oscar",
                case_id=case.case_id,
                output=output,
                metadata=metadata,
                experiment_dir=tmp_path,
            ),
        )
    )

    assert score.evidence == condition.resolve()


class _FakeRenderer:
    name = "oscar"

    def supports(self, case: ACWMCase):
        return OSCARRenderer(OSCARConfig(Path("repo"), Path("checkpoint"))).supports(case)

    def render_batch(self, requests: list[ACWMRenderRequest]):
        results = []
        for request in requests:
            request.output.parent.mkdir(parents=True, exist_ok=True)
            request.output.write_bytes(b"generated")
            metadata = request.output.with_suffix(".json")
            metadata.write_text("{}\n")
            results.append(
                ACWMRenderResult(
                    backend=self.name,
                    case_id=request.case.case_id,
                    output=request.output,
                    metadata=metadata,
                    experiment_dir=request.experiment_dir,
                )
            )
        return tuple(results)


class _Evaluator:
    def __init__(self, human_review: bool | None) -> None:
        self.human_review = human_review

    def evaluate(self, request):
        return ACWMScorecard(
            evaluator="test",
            action_adherence=0.9,
            embodiment_consistency=0.9,
            object_interaction=0.9,
            temporal_consistency=0.9,
            background_consistency=0.9,
            human_review_passed=self.human_review,
        )


@pytest.mark.parametrize(
    ("human_review", "status"),
    ((None, "pending_human_review"), (True, "accepted")),
)
def test_agentic_acwm_requires_human_review(
    tmp_path: Path, human_review: bool | None, status: str
) -> None:
    case = _case(tmp_path)
    controller = AgenticACWMController(
        {"oscar": _FakeRenderer()},
        _Evaluator(human_review),
        project_root=Path(__file__).resolve().parents[1],
    )

    outcome = controller.run(
        AgenticACWMRequest(
            cases=(case,),
            initial_proposals=(ACWMProposal(case.case_id, "oscar", seed=7),),
            experiment_root=tmp_path / "experiments",
            thresholds=ACWMThresholds(),
            maximum_rounds=1,
        )
    )

    assert outcome.status == status
    assert outcome.accepted is (human_review is True)
    trace = json.loads(outcome.trace_path.read_text())
    assert trace["human_review_required_for_acceptance"] is True
