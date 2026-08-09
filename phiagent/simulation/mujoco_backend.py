"""MuJoCo physics backend for actuated joint-trajectory replay."""

from __future__ import annotations

import math
import os
from typing import Any

from phiagent.simulation.base import (
    SimulationEvent,
    SimulationRequest,
    SimulationResult,
)


class MujocoBackend:
    """Replay named joints through their MuJoCo actuators and measure contacts."""

    def __init__(self, camera: str | int | None = None) -> None:
        self.camera = camera

    @staticmethod
    def _imports() -> tuple[Any, Any, Any]:
        if "DISPLAY" not in os.environ:
            os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            import imageio.v2 as imageio
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo backend requires the phiagent[simulation] extra"
            ) from exc
        return mujoco, np, imageio

    @staticmethod
    def _name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str:
        return mujoco.mj_id2name(model, object_type, object_id) or f"id:{object_id}"

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        mujoco, np, imageio = self._imports()
        model = mujoco.MjModel.from_xml_path(str(request.model_xml))
        data = mujoco.MjData(model)
        trajectory = request.trajectory
        reachability: list[dict[str, Any]] = []
        joint_ids: list[int] = []
        actuator_ids: list[int] = []
        for joint_name in trajectory.embodiment.joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                reachability.append(
                    {"joint": joint_name, "reason": "joint_missing_from_model"}
                )
                continue
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                reachability.append(
                    {
                        "joint": joint_name,
                        "reason": "expected_exactly_one_joint_actuator",
                        "actuator_count": int(len(matches)),
                    }
                )
                continue
            joint_ids.append(joint_id)
            actuator_ids.append(int(matches[0]))
        if reachability:
            return SimulationResult(
                backend=f"mujoco-{mujoco.__version__}",
                physically_valid=False,
                task_success=None,
                collision_events=(),
                contact_events=(),
                joint_limit_violations=trajectory.joint_limit_violations(),
                reachability_failures=tuple(reachability),
                slip_events=(),
                object_pose_trajectories={},
                rendered_rollout=None,
                metrics={"simulated_steps": 0},
            )

        for joint_id, value in zip(joint_ids, trajectory.joint_positions_rad[0]):
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)

        body_ids: dict[str, int] = {}
        for body_name in request.object_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                reachability.append(
                    {"body": body_name, "reason": "object_body_missing_from_model"}
                )
            else:
                body_ids[body_name] = body_id
        for goal in request.object_position_goals:
            if goal.body_name in body_ids:
                continue
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, goal.body_name
            )
            if body_id < 0:
                reachability.append(
                    {"body": goal.body_name, "reason": "goal_body_missing_from_model"}
                )
            else:
                body_ids[goal.body_name] = body_id
        object_poses: dict[str, list[dict[str, Any]]] = {
            name: [] for name in body_ids
        }
        contacts: list[SimulationEvent] = []
        collisions: list[SimulationEvent] = []
        seen_contacts: set[tuple[int, int]] = set()
        observed_named_pairs: set[tuple[str, str]] = set()
        forbidden_pairs = set(request.forbidden_contact_pairs)
        rendered_frames: list[Any] = []
        renderer = None
        if request.render_output is not None:
            renderer = mujoco.Renderer(
                model, height=request.render_height, width=request.render_width
            )
        next_render_s = 0.0
        render_period_s = 1.0 / request.render_fps
        simulated_steps = 0

        def record_state(timestamp_s: float) -> None:
            nonlocal next_render_s
            for body_name, body_id in body_ids.items():
                object_poses[body_name].append(
                    {
                        "timestamp_s": timestamp_s,
                        "translation_m": [float(value) for value in data.xpos[body_id]],
                        "quaternion_wxyz": [
                            float(value) for value in data.xquat[body_id]
                        ],
                    }
                )
            if renderer is not None and timestamp_s + 1e-9 >= next_render_s:
                renderer.update_scene(
                    data, camera=-1 if self.camera is None else self.camera
                )
                rendered_frames.append(renderer.render())
                next_render_s += render_period_s

        record_state(trajectory.timestamps_s[0])
        for segment in range(len(trajectory.timestamps_s) - 1):
            start_s = trajectory.timestamps_s[segment]
            end_s = trajectory.timestamps_s[segment + 1]
            start = np.asarray(trajectory.joint_positions_rad[segment], dtype=float)
            end = np.asarray(trajectory.joint_positions_rad[segment + 1], dtype=float)
            duration = end_s - start_s
            steps = max(1, math.ceil(duration / model.opt.timestep))
            for step in range(1, steps + 1):
                alpha = step / steps
                target = start + alpha * (end - start)
                data.ctrl[actuator_ids] = target
                mujoco.mj_step(model, data)
                simulated_steps += 1
                timestamp_s = start_s + alpha * duration
                active: set[tuple[int, int]] = set()
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                    active.add(pair)
                    if pair not in seen_contacts:
                        entities = tuple(
                            self._name(
                                mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                            )
                            for geom_id in pair
                        )
                        contacts.append(
                            event := SimulationEvent(
                                timestamp_s=timestamp_s,
                                event_type="contact_begin",
                                entities=entities,
                                details={"distance_m": float(contact.dist)},
                            )
                        )
                        named_pair = tuple(sorted(entities))
                        observed_named_pairs.add(named_pair)  # type: ignore[arg-type]
                        if named_pair in forbidden_pairs:
                            collisions.append(
                                SimulationEvent(
                                    timestamp_s=timestamp_s,
                                    event_type="forbidden_collision",
                                    entities=event.entities,
                                    details=event.details,
                                )
                            )
                seen_contacts = active
                record_state(timestamp_s)

        rendered_path: str | None = None
        if renderer is not None:
            renderer.close()
            assert request.render_output is not None
            request.render_output.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                request.render_output,
                rendered_frames,
                fps=request.render_fps,
                macro_block_size=1,
            )
            rendered_path = str(request.render_output)
        violations = trajectory.joint_limit_violations()
        goal_errors: dict[str, float] = {}
        for goal in request.object_position_goals:
            samples = object_poses.get(goal.body_name, [])
            if not samples:
                continue
            final_position = samples[-1]["translation_m"]
            goal_errors[goal.body_name] = math.dist(
                final_position, goal.target_translation_m
            )
        required_contacts_met = all(
            pair in observed_named_pairs for pair in request.required_contact_pairs
        )
        object_goals_met = all(
            goal_errors.get(goal.body_name, math.inf) <= goal.tolerance_m
            for goal in request.object_position_goals
        )
        has_task_criteria = bool(
            request.required_contact_pairs or request.object_position_goals
        )
        task_success = (
            required_contacts_met and object_goals_met if has_task_criteria else None
        )
        physically_valid = not violations and not reachability and not collisions
        return SimulationResult(
            backend=f"mujoco-{mujoco.__version__}",
            physically_valid=physically_valid,
            task_success=task_success,
            collision_events=tuple(collisions),
            contact_events=tuple(contacts),
            joint_limit_violations=violations,
            reachability_failures=tuple(reachability),
            slip_events=(),
            object_pose_trajectories={
                name: tuple(samples) for name, samples in object_poses.items()
            },
            rendered_rollout=rendered_path,
            metrics={
                "simulated_steps": simulated_steps,
                "contact_events": len(contacts),
                "forbidden_collisions": len(collisions),
                "required_contacts_met": required_contacts_met,
                "object_goals_met": object_goals_met,
                "object_goal_max_error_m": max(goal_errors.values(), default=0.0),
                "model_timestep_s": float(model.opt.timestep),
            },
        )
