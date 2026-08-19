#!/usr/bin/env python3
"""Run the native AC-WM branch over a compiled real-scene action set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phiagent.acwm.adapters import (  # noqa: E402
    BWMConfig,
    BWMRenderer,
    FlowWAMConfig,
    FlowWAMRenderer,
    Kinema4DConfig,
    Kinema4DRenderer,
    MiniMaxH3Config,
    MiniMaxH3Renderer,
    OSCARConfig,
    OSCARRenderer,
)
from phiagent.harness.task_reasoning import (  # noqa: E402
    TSHIRT_FOLD_TASK,
    TaskReasoningPlan,
)
from phiagent.harness.test_time_scaling import (  # noqa: E402
    HardGateTestTimeScalingRepairAgent,
    compile_task_reasoning_prompt,
    initial_scaled_proposals,
    load_test_time_scaling_policy,
)
from phiagent.rendering.minimax_h3 import file_sha256  # noqa: E402
from phiagent.acwm.schema import ACWMActionCondition, ACWMCase  # noqa: E402
from phiagent.agent.acwm import (  # noqa: E402
    ACWMProposal,
    ACWMThresholds,
    AgenticACWMController,
    AgenticACWMRequest,
    CommandACWMEvaluator,
)


PROMPTS = {
    "slide-left": (
        "In a fixed real laboratory camera view, a rigid silver dexterous robot arm "
        "grasps the yellow handled bowl, slides it clearly to the left along the metal "
        "table, and holds the left terminal state. Preserve one robot, one bowl, causal "
        "contact, the real lighting, and the unchanged background."
    ),
    "slide-right": (
        "In a fixed real laboratory camera view, a rigid silver dexterous robot arm "
        "grasps the yellow handled bowl, slides it clearly to the right along the metal "
        "table, and holds the right terminal state. Preserve one robot, one bowl, causal "
        "contact, the real lighting, and the unchanged background."
    ),
    "lift-up": (
        "In a fixed real laboratory camera view, a rigid silver dexterous robot arm "
        "grasps the yellow handled bowl, lifts it visibly off the metal table, and holds "
        "it in the upper terminal state. Preserve one robot, one bowl, causal contact, "
        "the real lighting, and the unchanged background."
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-manifest", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only the named case; repeat to keep multiple cases.",
    )
    parser.add_argument(
        "--prompt-suffix",
        action="append",
        help=(
            "Append one morphology or interaction constraint to the base prompt. "
            "Repeat to render matched prompt-repair candidates in one model load."
        ),
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("oscar", "minimax-h3", "bwm", "kinema4d", "flowwam"),
    )
    parser.add_argument("--task-reasoning-plan", type=Path)
    parser.add_argument("--test-time-scaling-config", type=Path)
    parser.add_argument("--tshirt-fold-tracking-contract", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/acwm-open-models"))
    parser.add_argument("--maximum-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--human-review-dir", type=Path)
    parser.add_argument("--oscar-repo", type=Path)
    parser.add_argument("--oscar-checkpoint", type=Path)
    parser.add_argument("--oscar-cosmos-reason", type=Path)
    parser.add_argument("--oscar-wan-vae", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--h3-repo", type=Path)
    parser.add_argument("--h3-model-base", type=Path)
    parser.add_argument("--h3-python", type=Path)
    parser.add_argument("--h3-steps", type=int, default=20)
    parser.add_argument("--h3-width", type=int, default=832)
    parser.add_argument("--h3-height", type=int, default=480)
    parser.add_argument("--h3-num-frames", type=int, default=124)
    parser.add_argument("--bwm-repo", type=Path)
    parser.add_argument("--bwm-base-model", type=Path)
    parser.add_argument("--bwm-checkpoint", type=Path)
    parser.add_argument("--bwm-action-stats", type=Path)
    parser.add_argument("--kinema-repo", type=Path)
    parser.add_argument("--kinema-base-transformer", type=Path)
    parser.add_argument("--kinema-checkpoint", type=Path)
    parser.add_argument("--kinema-dataset-root", type=Path)
    parser.add_argument("--kinema-episode-list", type=Path)
    parser.add_argument("--flowwam-repo", type=Path)
    parser.add_argument("--flowwam-base-model", type=Path)
    parser.add_argument("--flowwam-checkpoint", type=Path)
    parser.add_argument("--flowwam-python", type=Path)
    for name in (
        "action",
        "embodiment",
        "object",
        "temporal",
        "background",
    ):
        parser.add_argument(f"--{name}-threshold", type=float, default=0.75)
    return parser


def _required(parser: argparse.ArgumentParser, values: dict[str, Path | None]) -> None:
    missing = [name for name, value in values.items() if value is None]
    if missing:
        parser.error(f"selected backend is missing: {', '.join(missing)}")


def _manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def select_cases(cases: tuple[ACWMCase, ...], requested: list[str] | None) -> tuple[ACWMCase, ...]:
    """Keep an explicit case subset without silently ignoring misspellings."""

    if not requested:
        return cases
    wanted = tuple(dict.fromkeys(requested))
    available = {case.case_id for case in cases}
    unknown = set(wanted) - available
    if unknown:
        raise ValueError(f"unknown AC-WM cases: {sorted(unknown)}")
    return tuple(case for case in cases if case.case_id in wanted)


def initial_proposals(
    cases: tuple[ACWMCase, ...],
    renderers: dict[str, object],
    *,
    seed: int,
    prompt_suffixes: list[str] | None,
) -> tuple[ACWMProposal, ...]:
    """Build a same-seed prompt-repair batch that shares one backend load."""

    suffixes = tuple(dict.fromkeys(prompt_suffixes or ("",)))
    if any(not suffix.strip() for suffix in suffixes) and len(suffixes) > 1:
        raise ValueError("an empty prompt suffix cannot be mixed with repair suffixes")
    return tuple(
        ACWMProposal(
            case.case_id,
            name,
            seed,
            num_inference_steps=(renderer.config.steps if name == "minimax-h3" else 35),
            guidance_scale=1.0 if name == "minimax-h3" else 6.0,
            prompt_suffix=suffix,
        )
        for case in cases
        for name, renderer in renderers.items()
        if renderer.supports(case).supported
        for suffix in suffixes
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    reasoning_plan_path = (
        args.task_reasoning_plan.expanduser().resolve()
        if args.task_reasoning_plan is not None
        else None
    )
    reasoning_plan = None
    if reasoning_plan_path is not None:
        payload = json.loads(reasoning_plan_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("task reasoning plan must contain one JSON object")
        reasoning_plan = TaskReasoningPlan.from_dict(payload)
    scaling_policy_path = (
        args.test_time_scaling_config.expanduser().resolve()
        if args.test_time_scaling_config is not None
        else None
    )
    scaling_policy = (
        load_test_time_scaling_policy(scaling_policy_path)
        if scaling_policy_path is not None
        else None
    )
    tracking_contract_path = (
        args.tshirt_fold_tracking_contract.expanduser().resolve()
        if args.tshirt_fold_tracking_contract is not None
        else None
    )
    if scaling_policy is not None and reasoning_plan is None:
        parser.error("--test-time-scaling-config requires --task-reasoning-plan")
    if reasoning_plan is not None and reasoning_plan.task_type == TSHIRT_FOLD_TASK:
        if tracking_contract_path is None or not tracking_contract_path.is_file():
            parser.error("T-shirt reasoning requires --tshirt-fold-tracking-contract")
    if scaling_policy is not None and args.prompt_suffix:
        parser.error("test-time scaling owns prompt repair suffixes")
    selected = tuple(dict.fromkeys(args.backend or ("oscar",)))
    project_root = Path(__file__).resolve().parents[1]
    renderers = {}
    if "oscar" in selected:
        _required(
            parser,
            {"--oscar-repo": args.oscar_repo, "--oscar-checkpoint": args.oscar_checkpoint},
        )
        renderers["oscar"] = OSCARRenderer(
            OSCARConfig(
                repository=args.oscar_repo,
                checkpoint_dir=args.oscar_checkpoint,
                cosmos_reason_path=args.oscar_cosmos_reason,
                wan_vae_path=args.oscar_wan_vae,
                offline=args.offline,
                gpu_index=args.gpu,
            ),
            project_root=project_root,
        )
    if "minimax-h3" in selected:
        _required(
            parser,
            {"--h3-repo": args.h3_repo, "--h3-model-base": args.h3_model_base},
        )
        renderers["minimax-h3"] = MiniMaxH3Renderer(
            MiniMaxH3Config(
                repository=args.h3_repo,
                model_base_path=args.h3_model_base,
                python_executable=args.h3_python,
                gpu_index=args.gpu,
                steps=args.h3_steps,
                width=args.h3_width,
                height=args.h3_height,
                num_frames=args.h3_num_frames,
            ),
            project_root=project_root,
        )
    if "bwm" in selected:
        _required(
            parser,
            {
                "--bwm-repo": args.bwm_repo,
                "--bwm-base-model": args.bwm_base_model,
                "--bwm-checkpoint": args.bwm_checkpoint,
                "--bwm-action-stats": args.bwm_action_stats,
            },
        )
        renderers["bwm"] = BWMRenderer(
            BWMConfig(
                repository=args.bwm_repo,
                base_model_dir=args.bwm_base_model,
                checkpoint_path=args.bwm_checkpoint,
                action_stats=args.bwm_action_stats,
                gpu_index=args.gpu,
            ),
            project_root=project_root,
        )
    if "kinema4d" in selected:
        _required(
            parser,
            {
                "--kinema-repo": args.kinema_repo,
                "--kinema-base-transformer": args.kinema_base_transformer,
                "--kinema-checkpoint": args.kinema_checkpoint,
                "--kinema-dataset-root": args.kinema_dataset_root,
                "--kinema-episode-list": args.kinema_episode_list,
            },
        )
        renderers["kinema4d"] = Kinema4DRenderer(
            Kinema4DConfig(
                repository=args.kinema_repo,
                base_transformer=args.kinema_base_transformer,
                lora_path=args.kinema_checkpoint,
                dataset_root=args.kinema_dataset_root,
                episode_list=args.kinema_episode_list,
                gpu_index=args.gpu,
            ),
            project_root=project_root,
        )
    if "flowwam" in selected:
        _required(
            parser,
            {
                "--flowwam-repo": args.flowwam_repo,
                "--flowwam-base-model": args.flowwam_base_model,
                "--flowwam-checkpoint": args.flowwam_checkpoint,
            },
        )
        renderers["flowwam"] = FlowWAMRenderer(
            FlowWAMConfig(
                repository=args.flowwam_repo,
                base_model_root=args.flowwam_base_model,
                checkpoint_path=args.flowwam_checkpoint,
                python_executable=args.flowwam_python,
                gpu_index=args.gpu,
            ),
            project_root=project_root,
        )

    manifest_path = args.condition_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    first_frame = _manifest_path(manifest_path, str(manifest["first_frame"]))
    source_video = _manifest_path(manifest_path, str(manifest["source_video"]))
    plan_prompt = (
        compile_task_reasoning_prompt(reasoning_plan)
        if reasoning_plan is not None
        else ""
    )
    top_level_auxiliary = dict(manifest.get("auxiliary_inputs", {}))
    cases = select_cases(tuple(
        ACWMCase(
            case_id=str(item["label"]),
            first_frame=first_frame,
            source_video=source_video,
            action=ACWMActionCondition.from_json(
                _manifest_path(manifest_path, str(item["condition"]))
            ),
            prompt=str(
                item.get("prompt")
                or PROMPTS.get(str(item["label"]), str(item["instruction"]))
            ) + plan_prompt,
            auxiliary_inputs=tuple(
                (
                    str(key),
                    _manifest_path(manifest_path, str(value)),
                )
                for key, value in {
                    **top_level_auxiliary,
                    **dict(item.get("auxiliary_inputs", {})),
                }.items()
            ),
        )
        for item in manifest["variants"]
    ), args.cases)
    if reasoning_plan is not None and any(
        case.action.coordinate_frame != reasoning_plan.coordinate_frame for case in cases
    ):
        raise ValueError(
            "task reasoning plan and action conditions must use the same named coordinate frame"
        )
    support = {
        case.case_id: {
            name: as_report.__dict__
            for name, renderer in renderers.items()
            if (as_report := renderer.supports(case))
        }
        for case in cases
    }
    if args.plan_only:
        print(
            json.dumps(
                {
                    "support": support,
                    "task_reasoning_plan": (
                        {
                            "path": str(reasoning_plan_path),
                            "sha256": file_sha256(reasoning_plan_path),
                            "plan_sha256": reasoning_plan.plan_sha256,
                        }
                        if reasoning_plan_path is not None and reasoning_plan is not None
                        else None
                    ),
                    "test_time_scaling": (
                        {
                            "path": str(scaling_policy_path),
                            "sha256": file_sha256(scaling_policy_path),
                            "policy": scaling_policy.to_dict(),
                        }
                        if scaling_policy_path is not None and scaling_policy is not None
                        else None
                    ),
                    "tracking_contract": (
                        {
                            "path": str(tracking_contract_path),
                            "sha256": file_sha256(tracking_contract_path),
                        }
                        if tracking_contract_path is not None
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_only:
        preflight = {
            name: renderer.preflight()
            for name, renderer in renderers.items()
            if any(renderer.supports(case).supported for case in cases)
        }
        print(json.dumps({"support": support, "preflight": preflight}, indent=2, sort_keys=True))
        return 0

    proposals = (
        initial_scaled_proposals(
            cases,
            renderers,
            policy=scaling_policy,
            base_seed=args.seed,
        )
        if scaling_policy is not None
        else initial_proposals(
            cases,
            renderers,
            seed=args.seed,
            prompt_suffixes=args.prompt_suffix,
        )
    )
    if not proposals:
        raise ValueError("none of the selected models accepts the compiled action representation")
    if reasoning_plan is not None and reasoning_plan.task_type == TSHIRT_FOLD_TASK:
        assert reasoning_plan_path is not None
        assert tracking_contract_path is not None
        evaluator_command = [
            sys.executable,
            str(project_root / "scripts" / "evaluate_tshirt_fold_candidate.py"),
            "--candidate",
            "{candidate}",
            "--first-frame",
            "{first_frame}",
            "--plan",
            str(reasoning_plan_path),
            "--tracking-contract",
            str(tracking_contract_path),
            "--metadata",
            "{metadata}",
        ]
    else:
        evaluator_command = [
            sys.executable,
            str(project_root / "scripts" / "evaluate_acwm_candidate.py"),
            "--candidate",
            "{candidate}",
            "--condition",
            "{condition}",
            "--first-frame",
            "{first_frame}",
            "--source",
            "{source}",
            "--metadata",
            "{metadata}",
        ]
    if args.human_review_dir is not None:
        evaluator_command.extend(
            [
                "--human-review",
                str(args.human_review_dir.expanduser().resolve() / "{case_id}.json"),
            ]
        )
    controller = AgenticACWMController(
        renderers,
        CommandACWMEvaluator(tuple(evaluator_command)),
        repair_agent=(
            HardGateTestTimeScalingRepairAgent(scaling_policy, base_seed=args.seed)
            if scaling_policy is not None
            else None
        ),
        project_root=project_root,
    )
    outcome = controller.run(
        AgenticACWMRequest(
            cases=cases,
            initial_proposals=proposals,
            experiment_root=args.experiment_root,
            thresholds=ACWMThresholds(
                action_adherence=args.action_threshold,
                embodiment_consistency=args.embodiment_threshold,
                object_interaction=args.object_threshold,
                temporal_consistency=args.temporal_threshold,
                background_consistency=args.background_threshold,
            ),
            maximum_rounds=(
                len(scaling_policy.rounds)
                if scaling_policy is not None
                else args.maximum_rounds
            ),
            frozen_inputs=tuple(
                (name, path)
                for name, path in (
                    ("task_reasoning_plan", reasoning_plan_path),
                    ("test_time_scaling_policy", scaling_policy_path),
                    ("tshirt_fold_tracking_contract", tracking_contract_path),
                )
                if path is not None
            ),
        )
    )
    print(
        json.dumps(
            {
                "status": outcome.status,
                "experiment_dir": str(outcome.experiment_dir),
                "trace": str(outcome.trace_path),
                "best_outputs": {
                    item.proposal.case_id: str(item.result.output) for item in outcome.best_by_case
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if outcome.status == "accepted":
        return 0
    if outcome.status == "pending_human_review":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
