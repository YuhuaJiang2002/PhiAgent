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
    OSCARConfig,
    OSCARRenderer,
)
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
        choices=("oscar", "bwm", "kinema4d", "flowwam"),
    )
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
        ACWMProposal(case.case_id, name, seed, prompt_suffix=suffix)
        for case in cases
        for name, renderer in renderers.items()
        if renderer.supports(case).supported
        for suffix in suffixes
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
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
            ),
            auxiliary_inputs=tuple(
                (
                    str(key),
                    _manifest_path(manifest_path, str(value)),
                )
                for key, value in dict(item.get("auxiliary_inputs", {})).items()
            ),
        )
        for item in manifest["variants"]
    ), args.cases)
    support = {
        case.case_id: {
            name: as_report.__dict__
            for name, renderer in renderers.items()
            if (as_report := renderer.supports(case))
        }
        for case in cases
    }
    if args.preflight_only:
        preflight = {
            name: renderer.preflight()
            for name, renderer in renderers.items()
            if any(renderer.supports(case).supported for case in cases)
        }
        print(json.dumps({"support": support, "preflight": preflight}, indent=2, sort_keys=True))
        return 0

    proposals = initial_proposals(
        cases,
        renderers,
        seed=args.seed,
        prompt_suffixes=args.prompt_suffix,
    )
    if not proposals:
        raise ValueError("none of the selected models accepts the compiled action representation")
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
            maximum_rounds=args.maximum_rounds,
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
