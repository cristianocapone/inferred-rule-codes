"""Finite-sample controls for discrete communication codes (NumPy only).

Rule labels are used only for evaluation. Decoder fitting and evaluation use
disjoint episodes within each rule; this does not measure transfer to new rules.
"""
from collections import Counter
import numpy as np


def _entropy(ids):
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _information(rule_ids, message_ids):
    pairs = np.column_stack([rule_ids, message_ids])
    _, joint = np.unique(pairs, axis=0, return_inverse=True)
    return max(0.0, _entropy(rule_ids) + _entropy(message_ids) - _entropy(joint))


def evaluate_code(rules, messages, *, permutations=200, seed=0):
    """Report descriptive code statistics, permutation controls and held-out decoding.

The null permutes rule labels, retaining rule counts and message frequencies.
If null information equals label entropy (e.g. every message is a singleton),
chance-adjusted information is undefined and returned as None, never as 1.
Nearest-neighbour ties contribute fractional correctness, avoiding label-order
artefacts. The split is reproducible and stratified, with >=2 episodes per rule.
"""
    if permutations < 1:
        raise ValueError("At least one permutation is required")
    rules = np.asarray(rules)
    messages = np.asarray(messages)
    if messages.ndim != 2 or len(rules) != len(messages) or len(rules) == 0:
        raise ValueError("Expected aligned nonempty rule labels and [episodes, positions] messages")
    _, ri = np.unique(rules, return_inverse=True)
    _, mi, mc = np.unique(messages, axis=0, return_inverse=True, return_counts=True)
    hr, hm = _entropy(ri), _entropy(mi)
    info = _information(ri, mi)
    rng = np.random.default_rng(seed)
    null = np.array([_information(rng.permutation(ri), mi) for _ in range(permutations)])
    null_mean = float(null.mean())
    denominator = hr - null_mean
    by_rule = [np.flatnonzero(ri == r) for r in np.unique(ri)]
    modal = [Counter(mi[idx]).most_common(1)[0][0] for idx in by_rule]
    purity = [Counter(mi[idx]).most_common(1)[0][1] / len(idx) for idx in by_rule]
    result = dict(
        n_episodes=len(ri), n_rules=len(by_rule), message_length=messages.shape[1],
        n_unique_messages=len(mc), unique_messages_per_rule=len(mc) / len(by_rule),
        n_distinct_modal_messages=len(set(modal)),
        distinct_modal_messages_per_rule=len(set(modal)) / len(by_rule),
        mean_modal_purity=float(np.mean(purity)),
        min_episodes_per_rule=min(map(len, by_rule)), max_episodes_per_rule=max(map(len, by_rule)),
        singleton_episode_fraction=float(np.sum(mc == 1) / len(ri)),
        singleton_message_fraction=float(np.mean(mc == 1)),
        rule_entropy_bits=hr, message_entropy_bits=hm, empirical_information_bits=info,
        empirical_nmi=info / hr if hr > 0 else None,
        null_information_mean_bits=null_mean,
        null_nmi_mean=null_mean / hr if hr > 0 else None,
        excess_information_bits=info - null_mean,
        adjusted_information=(info - null_mean) / denominator if denominator > 1e-10 else None,
        permutation_p=float((1 + np.sum(null >= info - 1e-12)) / (permutations + 1)),
        permutations=permutations, seed=seed,
    )
    train, test = [], []
    for indices in by_rule:
        if len(indices) < 2:
            continue
        indices = rng.permutation(indices)
        cut = len(indices) // 2
        train.extend(indices[:cut]); test.extend(indices[cut:])
    if not test:
        result["heldout_decoder"] = None
        return result
    train, test = np.asarray(train), np.asarray(test)
    trm, try_ = messages[train], ri[train]
    prior = Counter(try_)
    lookup = {}
    for idx in train:
        lookup.setdefault(tuple(messages[idx]), Counter())[ri[idx]] += 1
    lookup_score, nn_score, seen = [], [], []
    for idx in test:
        counts = lookup.get(tuple(messages[idx]))
        seen.append(counts is not None)
        counts = counts or prior
        winners = [r for r, c in counts.items() if c == max(counts.values())]
        lookup_score.append(float(ri[idx] in winners) / len(winners))
        distances = np.count_nonzero(trm != messages[idx], axis=1)
        tied_labels = try_[distances == distances.min()]
        nn_score.append(float(np.mean(tied_labels == ri[idx])))
    result["heldout_decoder"] = dict(
        n_fit_episodes=len(train), n_test_episodes=len(test), n_eligible_rules=len(set(try_)),
        scope="new episodes of the same rule families; not unseen-family transfer",
        exact_lookup_accuracy=float(np.mean(lookup_score)),
        nearest_hamming_expected_accuracy=float(np.mean(nn_score)),
        message_seen_in_fit_fraction=float(np.mean(seen)),
        uniform_rule_chance=1 / len(set(try_)),
    )
    return result
