"""Learned transliteration map for romanised non-Latin names (E-T1).

anyascii turns "राम मार्केटिंग प्राइवेट लिमिटेड" into "ram marketimg praivet limited", while the
English copy reads "ram marketing private limited". "praivet" is then not recognised as a legal
form and looks like an ADDED word, the fake-record signal. We learn token -> token maps from
our own training pairs (a record in a non-Latin script matched to a Latin S1): clean both full
names, align tokens positionally when the counts agree, and keep frequent, unambiguous mappings.
Learned only from training labels (for validation: only from S1 outside the validation sample).
"""
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from resolver.normalize import LEGAL_FORMS, STOPWORDS, clean

MIN_COUNT, MIN_SHARE = 3, 0.6


def learn_map(pairs: pl.DataFrame) -> dict[str, str]:
    """pairs: (s1_name, rec_name) raw names of true pairs whose record is non-Latin."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s_raw, r_raw in pairs.select("s1_name", "rec_name").iter_rows():
        s_tok, r_tok = clean(s_raw or "").split(), clean(r_raw or "").split()
        if len(s_tok) != len(r_tok) or not s_tok:
            continue
        for s, r in zip(s_tok, r_tok):
            if s != r:
                counts[r][s] += 1
    out = {}
    for r, c in counts.items():
        s, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            out[r] = s
    return out


def translit_tokens(raw_name: str, mapping: dict[str, str]) -> str:
    """Raw name -> mapped name tokens, legal forms / stopwords removed (same rule as name_tok)."""
    toks = [mapping.get(t, t) for t in clean(raw_name or "").split()]
    return " ".join(t for t in toks if t not in LEGAL_FORMS and t not in STOPWORDS)


if __name__ == "__main__":
    demo = pl.DataFrame({"s1_name": ["Ram Marketing Private Limited"] * 3 + ["Sunrise Technology"] * 3,
                         "rec_name": ["राम मार्केटिंग प्राइवेट लिमिटेड"] * 3 + ["sanraij teknolji"] * 3})
    m = learn_map(demo)
    assert m["praivet"] == "private" and m["teknolji"] == "technology", m
    assert translit_tokens("राम मार्केटिंग प्राइवेट लिमिटेड", m) == "ram marketing", translit_tokens("राम मार्केटिंग प्राइवेट लिमिटेड", m)
    assert fuzz.token_set_ratio(translit_tokens("sanraij teknolji", m), "sunrise technology") == 100
    print("translit self-check ok")
