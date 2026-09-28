"""Three-arm, bounded RSI comparison. Pure control logic; no model imports.

Frontier membership permits research exploration, never a release promotion.
Only the fixed five-coordinate non-regression gate can promote a release.
"""
from dataclasses import dataclass
import json

from .video_meta_rsi import (MetaRSI, METRICS, ScoreCard, digest, text_hash,
                             dominates, validate_patch, validate_policy, _sha, _text)

ARMS = ('rsi', 'pareto_rsi', 'pareto_meta_rsi')
SCALES = (1.0, 100.0, 1.0, 1.0, 1.0)


def stratified_cases(base_status, families, salt='pareto-rsi-v1'):
    """Prospective, score-blind split; source cases and all their seeds stay together.

    Freeze precisely two FAIL and two non-FAIL cases in each adaptation split.
    Spread task families by selecting the least represented available family,
    then a salted hash. No alternative split is tried if this design is infeasible.
    """
    if len(base_status) != 20 or set(base_status) != set(families):
        raise ValueError('Require the declared complete 20-case development universe')
    if set(base_status.values()) - {'FAIL', 'PASS', 'UNKNOWN'}:
        raise ValueError('Invalid base status')
    groups = {True: [], False: []}
    for case, status in base_status.items():
        groups[status == 'FAIL'].append(case)
    if min(map(len, groups.values())) < 6:
        raise ValueError('Cannot freeze the required informative balanced splits')
    result = {}; global_counts = {}; split_counts = {}
    for name in ('train', 'validation', 'meta_validation'):
        split_counts[name] = {}
        for failed in (True, False):
            for _ in range(2):
                def rank(c):
                    f = families[c]
                    return (split_counts[name].get(f, 0), global_counts.get(f, 0),
                            text_hash(salt + ':' + name + ':' + c))
                case = min(groups[failed], key=rank); groups[failed].remove(case)
                result[case] = name; f = families[case]
                split_counts[name][f] = split_counts[name].get(f, 0) + 1
                global_counts[f] = global_counts.get(f, 0) + 1
    result.update({c: 'test' for c in base_status if c not in result})
    return result


@dataclass(frozen=True)
class Assessment:
    card: ScoreCard
    # Full selected-output phenotype, independent of skill text and score-job id.
    outputs: tuple[tuple[str, int, str], ...]

    def verify(self, skill, part, protocol):
        vector = self.card.means(skill, part, protocol)
        keys = [(c, s) for c, s, _ in self.outputs]
        if len(keys) != len(set(keys)) or set(keys) != set(part.members):
            raise ValueError('Output phenotype must cover the exact partition')
        for _, _, value in self.outputs:
            _sha(value)
        return vector, digest(sorted(self.outputs))


class Frontier:
    def __init__(self, initial):
        self.entries = {initial['sha256']: initial}
        self.uses = {}
        self.reference = initial['inner']

    def worst_gain(self, item):
        return min((x - b) / s for x, b, s in zip(item['inner'], self.reference, SCALES))

    def add(self, item):
        if any(e['phenotype'] == item['phenotype'] for e in self.entries.values()):
            return 'duplicate_selected_outputs'
        if any(e['inner'] == item['inner'] or dominates(e['inner'], item['inner'])
               for e in self.entries.values()):
            return 'dominated_or_equal_vector'
        self.entries = {k: e for k, e in self.entries.items()
                        if not dominates(item['inner'], e['inner'])}
        self.entries[item['sha256']] = item
        return 'retained'

    def parent(self):
        # Exploration precedes exploitation, so complementary tradeoffs are used.
        key = min(self.entries, key=lambda k: (self.uses.get(k, 0),
                  -self.worst_gain(self.entries[k]), k))
        self.uses[key] = self.uses.get(key, 0) + 1
        return self.entries[key]


class ParetoRSI(MetaRSI):
    """Reuse the existing isolation/observation contract, with a fixed arm toggle.

    Each term consumes two proposal slots even for invalid/duplicate proposals.
    Four skill proposals per arm; no validation numbers enter model contexts.
    Meta-policy trials share the same parent, training observations and seed
    schedule as the fixed-policy trials. Every valid unique child gets the same
    train/inner/meta coverage, irrespective of whether it can be promoted.
    """
    def _assess(self, skill, split, stage):
        if split not in ('validation', 'meta_validation'):
            raise ValueError('Final evaluation is outside adaptation')
        part = self.parts[split]
        result = self.evaluate(skill, part, stage)
        vector, phenotype = result.verify(skill, part, self.protocol)
        self.record(stage, {'skill_sha256': text_hash(skill), 'split': split,
                    'partition_sha256': part.sha256, 'means': dict(zip(METRICS, vector)),
                    'phenotype_sha256': phenotype, 'outputs': result.outputs,
                    'evidence_sha256': result.card.evidence_sha256})
        return vector, phenotype

    def run(self, skill, policy, *, arm, terms=2):
        if arm not in ARMS or terms != 2:
            raise ValueError('This protocol freezes three arms and exactly two terms')
        self._writable_text(_text(skill, 300)); self._writable_text(_text(policy, 220))
        measured = {}; training = {}; proposal_history = []
        def measure(value, stage):
            key = text_hash(value)
            if key not in measured:
                inner, phenotype = self._assess(value, 'validation', stage + '-inner')
                meta, _ = self._assess(value, 'meta_validation', stage + '-meta')
                measured[key] = {'skill': value, 'sha256': key, 'inner': inner,
                                 'meta': meta, 'phenotype': phenotype}
            return measured[key]
        def observe(item, stage):
            key = item['sha256']
            if key not in training:
                training[key] = self._observe(item['skill'], stage)
                self.record(stage, {'training': training[key], 'skill_sha256': key})
            return training[key]
        clone = lambda x: json.loads(json.dumps(x))
        incumbent = measure(skill, 'initial'); frontier = Frontier(incumbent)
        for term in range(1, terms + 1):
            prefix = f'term-{term:02d}'
            parent = incumbent if arm == 'rsi' else frontier.parent()
            signal = observe(parent, prefix + '-parent-train')
            # The archive's scores and membership decisions are never serialized here.
            context = {'training': signal, 'training_history': [
                {'skill_sha256': k, 'training': v} for k, v in training.items()][-5:],
                'prior_proposals': clone(proposal_history[-4:])}
            next_policy = policy; policy_valid = False
            if arm == 'pareto_meta_rsi':
                raw = self.propose_policy(policy, clone(context), prefix + '-policy')
                self.record(prefix + '-policy-proposal', raw)
                try:
                    next_policy = self._writable_text(validate_policy(raw))
                    policy_valid = next_policy != policy
                except (ValueError, TypeError) as exc:
                    self.record(prefix + '-policy-rejected', {'error': str(exc)})
            children = []
            for slot in range(2):
                stage = prefix + f'-slot-{slot + 1}'
                trial_policy = next_policy if slot == 1 else policy
                # Slot 2 sees slot 1 text only, avoiding deterministic duplicate edits.
                child_context = {**clone(context), 'prior_proposals': clone(proposal_history[-4:]),
                                 'proposal_slot': (term - 1) * 2 + slot + 1}
                raw = self.propose_skill(trial_policy, parent['skill'], child_context, stage)
                self.record(stage + '-proposal', raw)
                proposal_history.append({'parent_skill_sha256': parent['sha256'], 'proposal': raw})
                reason = 'valid'; child = parent; candidate = None
                try:
                    candidate = self.apply_patch(parent['skill'], validate_patch(raw))
                    self._writable_text(_text(candidate, 300))
                except (ValueError, TypeError, KeyError) as exc:
                    candidate = None
                    reason = 'invalid_proposal: ' + str(exc)
                if candidate is not None:
                    if text_hash(candidate) in measured:
                        reason = 'duplicate_skill_consumed_slot'
                    # Never swallow model/coverage/evaluator failures as bad proposals.
                    child = measure(candidate, stage)
                    observe(child, stage + '-train')
                children.append(child)
                membership = frontier.add(child) if arm != 'rsi' else 'single_incumbent_arm'
                self.record(stage + '-candidate', {'reason': reason, 'skill_sha256': child['sha256'],
                            'parent_sha256': parent['sha256'], 'policy_sha256': text_hash(trial_policy),
                            'frontier_decision': membership, 'frontier_size': len(frontier.entries)})
            old_skill, old_policy = incumbent['sha256'], text_hash(policy)
            qualifies = lambda c: dominates(c['inner'], incumbent['inner']) and dominates(c['meta'], incumbent['meta'])
            policy_accepted = (arm == 'pareto_meta_rsi' and policy_valid and qualifies(children[1])
                               and dominates(children[1]['meta'], children[0]['meta']))
            eligible = ([children[1]] if policy_accepted else [children[0]]) if arm == 'pareto_meta_rsi' else children
            eligible = [c for c in eligible if qualifies(c)]
            if eligible:
                incumbent = min(eligible, key=lambda c: (-frontier.worst_gain(c), c['sha256']))
            if policy_accepted:
                policy = next_policy
            self.record(prefix + '-export', {'skill_sha256': incumbent['sha256'],
                'policy_sha256': text_hash(policy), 'skill_accepted': old_skill != incumbent['sha256'],
                'policy_accepted': policy_accepted, 'previous_policy_sha256': old_policy,
                'search_parent_sha256': parent['sha256'], 'frontier': list(frontier.entries.values())})
        release = {'skill': incumbent['skill'], 'policy': policy, 'arm': arm, 'terms': terms,
            'skill_sha256': incumbent['sha256'], 'policy_sha256': text_hash(policy),
            'proposal_slots_consumed': len(proposal_history), 'unique_skills_evaluated': len(measured),
            'protocol_sha256': self.protocol, 'final_test_evaluated': False}
        self.record('frozen-release', release)
        return release
