"""Dependency-light PhiAgent-Bench contracts and evaluators."""

from phiagent.benchmark.metrics import BenchmarkPolicy, evaluate_submission
from phiagent.benchmark.schema import BenchmarkSuite, Submission

__all__ = [
    "BenchmarkPolicy",
    "BenchmarkSuite",
    "Submission",
    "evaluate_submission",
]
