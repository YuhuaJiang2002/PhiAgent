"""Deterministic physical-state extractor from aligned teacher observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from phiagent.perception.schema import (
    HandObservation,
    ObjectObservation,
    PerceptionSequence,
)
from phiagent.physical_language.schema import (
    ContactState,
    EPLChunk,
    EPLSequence,
    ManipulationPhase,
    MotionSE3,
    Point3D,
    Relation,
)


@dataclass(frozen=True)
class PhysicalStateExtractorConfig:
    contact_distance_m: float = 0.045
    moving_distance_m: float = 0.005

    def __post_init__(self) -> None:
        if self.contact_distance_m <= 0 or self.moving_distance_m <= 0:
            raise ValueError("extractor distance thresholds must be positive")


def _motion(current: tuple[float, ...], following: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(after - before for before, after in zip(current, following))


def _relative_quaternion(
    current: tuple[float, float, float, float],
    following: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    cx, cy, cz, cw = current
    inverse = (-cx, -cy, -cz, cw)
    ix, iy, iz, iw = inverse
    fx, fy, fz, fw = following
    value = (
        fw * ix + fx * iw + fy * iz - fz * iy,
        fw * iy - fx * iz + fy * iw + fz * ix,
        fw * iz + fx * iy - fy * ix + fz * iw,
        fw * iw - fx * ix - fy * iy - fz * iz,
    )
    norm = math.sqrt(sum(component * component for component in value))
    return tuple(component / norm for component in value)  # type: ignore[return-value]


class PhysicalStateExtractor:
    def __init__(self, config: PhysicalStateExtractorConfig | None = None) -> None:
        self.config = config or PhysicalStateExtractorConfig()

    def _contact(
        self, hand: HandObservation, object_observation: ObjectObservation | None
    ) -> tuple[ContactState, tuple[Point3D, ...]]:
        if object_observation is None:
            return ContactState.UNKNOWN, ()
        object_center = object_observation.pose.translation_m
        distances = [
            math.dist(point.xyz_m, object_center) for point in hand.fingertips
        ]
        points = tuple(
            point
            for point, distance in zip(hand.fingertips, distances)
            if distance <= self.config.contact_distance_m
        )
        return (ContactState.STABLE if points else ContactState.NONE), points

    def extract(self, observations: PerceptionSequence, source_video: str) -> EPLSequence:
        chunks: list[EPLChunk] = []
        previous_contact = ContactState.NONE
        for index in range(len(observations.hands) - 1):
            hand = observations.hands[index]
            following_hand = observations.hands[index + 1]
            obj = observations.objects[index]
            following_obj = observations.objects[index + 1]
            eef_translation = _motion(
                hand.wrist_pose.translation_m, following_hand.wrist_pose.translation_m
            )
            eef_motion = MotionSE3(
                expressed_in=hand.wrist_pose.target_frame,
                translation_m=eef_translation,  # type: ignore[arg-type]
                quaternion_xyzw=_relative_quaternion(
                    hand.wrist_pose.quaternion_xyzw,
                    following_hand.wrist_pose.quaternion_xyzw,
                ),
            )
            contact_state, contact_points = self._contact(hand, obj)
            object_delta = None
            state_changes: tuple[str, ...] = ()
            object_moved = False
            if obj is not None and following_obj is not None:
                translation = _motion(
                    obj.pose.translation_m, following_obj.pose.translation_m
                )
                object_moved = math.sqrt(sum(value * value for value in translation)) >= (
                    self.config.moving_distance_m
                )
                object_delta = MotionSE3(
                    expressed_in=obj.pose.target_frame,
                    translation_m=translation,  # type: ignore[arg-type]
                    quaternion_xyzw=_relative_quaternion(
                        obj.pose.quaternion_xyzw, following_obj.pose.quaternion_xyzw
                    ),
                )
                if obj.state != following_obj.state:
                    state_changes = (f"{obj.state}->{following_obj.state}",)
            hand_moved = math.sqrt(
                sum(value * value for value in eef_translation)
            ) >= self.config.moving_distance_m
            if previous_contact is ContactState.STABLE and contact_state is ContactState.NONE:
                phase = ManipulationPhase.RELEASE
            elif contact_state is ContactState.STABLE and object_moved:
                phase = ManipulationPhase.MANIPULATE
            elif contact_state is ContactState.STABLE:
                phase = ManipulationPhase.GRASP
            elif hand_moved:
                phase = ManipulationPhase.APPROACH
            else:
                phase = ManipulationPhase.IDLE
            relations = ()
            if obj is not None and contact_state is ContactState.STABLE:
                relations = (
                    Relation(
                        subject=hand.wrist_pose.source_frame.name,
                        predicate="contacting",
                        object=obj.object_id,
                        confidence=min(hand.confidence, obj.confidence),
                    ),
                )
            confidence_values = [hand.confidence, following_hand.confidence]
            if obj is not None:
                confidence_values.append(obj.confidence)
            if following_obj is not None:
                confidence_values.append(following_obj.confidence)
            chunks.append(
                EPLChunk(
                    start_s=hand.timestamp_s,
                    end_s=following_hand.timestamp_s,
                    phase=phase,
                    eef_delta=eef_motion,
                    wrist_pose=hand.wrist_pose,
                    fingertips=hand.fingertips,
                    hand_aperture_m=hand.aperture_m,
                    hand_articulation=hand.articulation,
                    contact_state=contact_state,
                    contact_points=contact_points,
                    object_pose=obj.pose if obj is not None else None,
                    object_delta=object_delta,
                    object_state_changes=state_changes,
                    relations=relations,
                    confidence=min(confidence_values),
                )
            )
            previous_contact = contact_state
        return EPLSequence("0.1.0", source_video, tuple(chunks))

