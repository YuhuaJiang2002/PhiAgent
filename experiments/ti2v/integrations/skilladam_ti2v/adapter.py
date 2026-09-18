"""TI2V metrics and domain registration around an unchanged SkillAdam runner.

Imports of the optional upstream dependency happen only at this adapter boundary.
The private registry has no extension API in the pinned release; the one added
entry and package search path are explicit adaptations, not upstream patches.
"""
from pathlib import Path
import math

METRICS = ('BLEUScore', 'CLIPScore', 'hsd', 'dyn', 'ndtw')
BENCHMARK = 'ewm_ti2v'


def register():
    import skilladam.benchmarks
    from skilladam.benchmarks.base import BenchmarkManifest
    from skilladam.benchmarks.registry import BenchmarkRegistration, _REGISTRATIONS
    from skilladam.core.acceptance_gate import AcceptanceGate, ImprovementRule, NonRegressionRule
    from skilladam.types import MetricResult, RolloutRequest, RolloutResult
    from skilladam.usage import normalize_usage

    location = str(Path(__file__).parent)
    if location not in skilladam.benchmarks.__path__:
        skilladam.benchmarks.__path__.append(location)

    class TI2VAdapter:
        manifest = BenchmarkManifest(BENCHMARK, 'EWMBench TI2V adaptation',
                                     'Opened real robot video development; not a full benchmark release.',
                                     frozenset(('train', 'validation', 'test')))

        def __init__(self, data_root):
            self.data_root = Path(data_root)
            self.acceptance = AcceptanceGate(
                improvements=tuple(ImprovementRule(m) for m in METRICS),
                non_regressions=tuple(NonRegressionRule(m) for m in METRICS))
            # The pinned runner serializes this standard adapter attribute into
            # its resume signature; preserve the actual five-metric gate object.
            self._gate = self.acceptance

        def validate_dependencies(self):
            pass

        def load_cases(self, split):
            raise RuntimeError('The remote runner must supply its hash-bound, case-grouped manifest explicitly')

        def build_rollout_request(self, case, *, method, split, skill, seed):
            if case.metadata['split'] != split:
                raise ValueError('Case split differs from rollout split')
            return RolloutRequest(case=case, method=method, split=split, skill=skill,
                                  seed=seed, metadata={'output_protocol': 'ti2v-full-native-video'})

        def parse_result(self, case, raw_result):
            values = dict(raw_result['metrics'])
            if set(values) != set(METRICS) or not all(math.isfinite(float(x)) for x in values.values()):
                raise ValueError('Require all five finite official metrics')
            if raw_result['case_id'] != case.case_id or not raw_result.get('official_evidence_sha256'):
                raise ValueError('Missing case-matched official score evidence')
            usage = tuple(normalize_usage(x) for x in raw_result.get('usage', ()))
            return RolloutResult(case_id=case.case_id, output=raw_result['selected_sha256'],
                                 trajectory=tuple(raw_result['trajectory']), usage=usage,
                                 metadata={'metrics': values,
                                           'official_evidence_sha256': raw_result['official_evidence_sha256']})

        def evaluate(self, result):
            values = dict(result.metadata['metrics'])
            return MetricResult(primary=values['ndtw'], metrics=values,
                                case_metrics={result.case_id: values}, sample_count=1)

        def gate(self, baseline, candidate):
            return self.acceptance.judge(baseline, candidate)

    existing = _REGISTRATIONS.get(BENCHMARK)
    if existing is not None and existing.manifest != TI2VAdapter.manifest:
        raise ValueError('Conflicting TI2V registration')
    _REGISTRATIONS[BENCHMARK] = BenchmarkRegistration(TI2VAdapter.manifest, 0, TI2VAdapter)
    return TI2VAdapter
