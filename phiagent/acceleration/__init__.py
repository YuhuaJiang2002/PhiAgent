"""Optional, evidence-gated runtime acceleration contracts."""

from .sol_engine import (
    H3ABPlan,
    H3ABResult,
    SolEngineH3Config,
    assess_h3_ab_result,
    plan_h3_ab_experiment,
    write_h3_quality_evidence_template,
    validate_sol_engine_source,
    validate_matched_h3_benchmarks,
)

__all__ = [
    "H3ABPlan",
    "H3ABResult",
    "SolEngineH3Config",
    "assess_h3_ab_result",
    "plan_h3_ab_experiment",
    "write_h3_quality_evidence_template",
    "validate_sol_engine_source",
    "validate_matched_h3_benchmarks",
]
