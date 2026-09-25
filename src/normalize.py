"""Text normalisation shared by every stage. v1: just enough for the exact-key rule baseline."""
import re

from anyascii import anyascii

# Legal forms across US / India / France (hand-written domain knowledge, no external data).
LEGAL_FORMS = frozenset({
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "pc", "plc", "pllc", "sarl", "sas", "sasu", "eurl", "sci", "sa",
})
STOPWORDS = frozenset({"and", "the", "of"})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def clean(text: str) -> str:
    # Dots are removed (not spaced) so "L.L.C." -> "llc" and "Pvt." -> "pvt".
    return _NON_ALNUM.sub(" ", anyascii(text).lower().replace(".", "")).strip()


def name_key(name: str) -> str:
    tokens = {t for t in clean(name).split() if t not in LEGAL_FORMS and t not in STOPWORDS}
    return " ".join(sorted(tokens))


def house_no(address: str) -> str:
    m = _DIGITS.search(anyascii(address))
    return (m.group().lstrip("0") or "0") if m else ""
