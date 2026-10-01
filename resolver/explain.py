"""Human-readable feature catalogue and per-pair explanations.

Explanations use exact TreeSHAP contributions of the stage-1 LightGBM (log-odds units), grouped
into the evidence families a reviewer thinks in: name, address, house number, competition between
candidates, vocabulary, and record properties.
"""
from __future__ import annotations

import math

import numpy as np

GROUPS = {
    "name": "Name similarity",
    "address": "Address similarity",
    "number": "House number",
    "competition": "Competing candidates",
    "vocab": "Generic vs. rare words",
    "record": "Record properties",
    "collective": "Other records' evidence",
}

# feature -> (label, group, kind); kind drives value formatting: pct 0..100, cos 0..1, bool, int, num
_INFO = {
    "name_ratio": ("Name edit similarity", "name", "pct"),
    "name_tset": ("Name token-set similarity", "name", "pct"),
    "name_partial": ("Name partial match", "name", "pct"),
    "name_jw": ("Name Jaro-Winkler", "name", "cos"),
    "ns_ratio": ("Space-free name similarity", "name", "pct"),
    "namekey_eq": ("Same name key (sorted core words)", "name", "bool"),
    "ns_eq": ("Identical space-free name", "name", "bool"),
    "cos_name": ("Name TF-IDF cosine (char 3-grams)", "name", "cos"),
    "len_name_rec": ("Record name length", "name", "int"),
    "len_name_s1": ("Reference name length", "name", "int"),
    "n_tok_extra": ("Words only in the record", "name", "int"),
    "n_tok_miss": ("Reference words missing from record", "name", "int"),
    "addr_tset": ("Address token-set similarity", "address", "pct"),
    "addr_partial": ("Address partial match", "address", "pct"),
    "cos_addr": ("Address TF-IDF cosine", "address", "cos"),
    "cos_comb": ("Combined TF-IDF cosine", "address", "cos"),
    "fz": ("Combined fuzzy score", "address", "pct"),
    "hn_lev": ("House-number edit distance", "number", "int"),
    "hn_eq": ("Same house number", "number", "bool"),
    "hn_missing_rec": ("Record has no house number", "number", "bool"),
    "hn_missing_s1": ("Reference has no house number", "number", "bool"),
    "hn_prefix": ("House numbers share a prefix", "number", "bool"),
    "nums_shared": ("Shared address numbers", "number", "int"),
    "nums_jac": ("Address-number overlap (Jaccard)", "number", "cos"),
    "nums_extra_rec": ("Numbers only in the record", "number", "int"),
    "nums_missing_rec": ("Reference numbers missing from record", "number", "int"),
    "hn_delta": ("House-number difference", "number", "num"),
    "hn_absdelta": ("House-number distance", "number", "num"),
    "hn_substr": ("One house number contains the other", "number", "bool"),
    "rank_fz": ("Rank among the record's candidates", "competition", "int"),
    "gap_fz": ("Gap to the record's best candidate", "competition", "num"),
    "rev_rank_fz": ("Rank among the reference's candidates", "competition", "int"),
    "n_cand_rec": ("Candidates of this record", "competition", "int"),
    "n_cand_s1": ("Candidates of this reference", "competition", "int"),
    "margin_fz": ("Fuzzy margin over runner-up", "competition", "num"),
    "rev_is_best_cos": ("Reference is the record's best cosine", "competition", "bool"),
    "rev_margin_cos": ("Cosine margin over the record's next best", "competition", "cos"),
    "rev_rel_cos": ("Cosine relative to record's best", "competition", "cos"),
    "rev_rel_name": ("Name cosine relative to record's best", "competition", "cos"),
    "ctx_rel_cos": ("Cosine relative to reference's best", "competition", "cos"),
    "ctx_gap_cos": ("Cosine gap to reference's best", "competition", "cos"),
    "ctx_rel_name": ("Name cosine relative to reference's best", "competition", "cos"),
    "ctx_rel_addr": ("Address cosine relative to reference's best", "competition", "cos"),
    "ctx_is_best_cos": ("Record is the reference's best cosine", "competition", "bool"),
    "n_extra_generic": ("Generic words added (Holdings, Group...)", "vocab", "int"),
    "n_extra_rare": ("Rare words added", "vocab", "int"),
    "n_miss_generic": ("Generic words dropped", "vocab", "int"),
    "n_miss_rare": ("Rare words dropped", "vocab", "int"),
    "core_tset": ("Rare-word core similarity", "vocab", "pct"),
    "core_eq": ("Identical rare-word core", "vocab", "bool"),
    "src3": ("Record from Source 3", "record", "bool"),
    "nonlatin": ("Record in non-Latin script", "record", "bool"),
    "empty_addr": ("Record has empty address", "record", "bool"),
    "in_keys": ("Found only by exact keys", "record", "bool"),
    "p1": ("Stage-1 probability", "collective", "cos"),
    "p1_rank_rec": ("Stage-1 rank for the record", "collective", "int"),
    "p1_margin_rec": ("Stage-1 margin over runner-up", "collective", "cos"),
    "p1_sum_rec": ("Stage-1 mass over record's candidates", "collective", "num"),
    "claims_s1": ("Other confident records on this reference", "collective", "int"),
    "best_other_p1_s1": ("Best other record on this reference", "collective", "cos"),
    "rank_among_claimants": ("Rank among the reference's claimants", "collective", "int"),
    "n_twins": ("Exact-key twin records", "collective", "int"),
    "twins_agree": ("Twins choosing the same reference", "collective", "int"),
    "twins_best_p1_same": ("Best twin probability on this reference", "collective", "cos"),
}
FEATURE_INFO = {k: {"label": v[0], "group": v[1], "group_label": GROUPS[v[1]]} for k, v in _INFO.items()}


def fmt_value(feature: str, v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    kind = _INFO.get(feature, ("", "", "num"))[2]
    if kind == "bool":
        return "yes" if v >= 0.5 else "no"
    if kind == "pct":
        return f"{v:.0f}%"
    if kind == "cos":
        return f"{v:.2f}"
    if kind == "int":
        return f"{v:.0f}"
    return f"{v:.3g}"


def explain_rows(features: list[str], X: np.ndarray, contrib: np.ndarray, top: int = 6) -> list[dict]:
    """Per row: grouped contributions + the strongest individual reasons (both signs)."""
    out = []
    for x, c in zip(X, contrib):
        groups: dict[str, float] = {}
        for f, v in zip(features, c[:-1]):
            g = _INFO.get(f, ("", "record", ""))[1]
            groups[g] = groups.get(g, 0.0) + float(v)
        order = np.argsort(-np.abs(c[:-1]))[:top]
        out.append({
            "base": float(c[-1]),
            "groups": sorted(({"group": g, "label": GROUPS[g], "value": round(v, 3)} for g, v in groups.items()),
                             key=lambda r: -abs(r["value"])),
            "reasons": [{"feature": features[i], "label": _INFO.get(features[i], (features[i],))[0],
                         "value": fmt_value(features[i], float(x[i])), "impact": round(float(c[i]), 3)}
                        for i in order if abs(c[i]) > 1e-3],
        })
    return out
