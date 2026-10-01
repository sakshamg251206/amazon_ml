"""Competition metric: F0.5 per Source-1 entity, macro-averaged.

With P = TP/|pred| and R = TP/|true|, F0.5 = 1.25*P*R / (0.25*P + R), which simplifies to
1.25*TP / (|pred| + 0.25*|true|). A singleton (no true matches) scores 1 only for an empty
prediction.
"""
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

_EMPTY: frozenset[str] = frozenset()


def entity_f05(pred: AbstractSet[str], true: AbstractSet[str]) -> float:
    if not true:
        return 0.0 if pred else 1.0
    tp = len(pred & true)
    return 1.25 * tp / (len(pred) + 0.25 * len(true)) if tp else 0.0


def macro_f05(pred: Mapping[str, AbstractSet[str]], truth: Mapping[str, AbstractSet[str]]) -> float:
    return sum(entity_f05(pred.get(s1, _EMPTY), t) for s1, t in truth.items()) / len(truth)
