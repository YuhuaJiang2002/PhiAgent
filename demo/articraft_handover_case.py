"""ArtiCraft SDK model for the first handover-oriented asset demo."""

from __future__ import annotations

from build123d import Box

from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    Material,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("handover_case")

    base = model.part("case")
    base.add(
        Box(0.14, 0.065, 0.045),
        name="case_body",
        material=Material.ABS_PLASTIC,
        color=(0.12, 0.32, 0.58),
    )
    base.add(
        Box(0.018, 0.069, 0.025).translate((-0.052, 0.0, 0.0)),
        name="left_grip_band",
        material=Material.RUBBER,
        color=(0.05, 0.06, 0.07),
    )
    base.add(
        Box(0.018, 0.069, 0.025).translate((0.052, 0.0, 0.0)),
        name="right_grip_band",
        material=Material.RUBBER,
        color=(0.05, 0.06, 0.07),
    )

    lid = model.part("lid")
    lid.add(
        Box(0.14, 0.065, 0.012).translate((0.0, 0.0325, 0.0055)),
        name="lid_panel",
        material=Material.ABS_PLASTIC,
        color=(0.82, 0.48, 0.10),
    )

    model.articulation(
        "lid_hinge",
        ArticulationType.REVOLUTE,
        base,
        lid,
        origin=Origin(xyz=(0.0, -0.0325, 0.0225)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=0.0, upper=1.5708),
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.allow_overlap(
        "case",
        "lid",
        reason="the lid edge has a small designed hinge embed",
        shape_a="case_body",
        shape_b="lid_panel",
    )
    ctx.expect_contact("case", "lid")
    return ctx.report()
