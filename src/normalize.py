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


# ---- v2: richer fields for candidate generation and pair features ----
_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9-]+)\.(?:com|co\.in|in|net|org|fr|us|biz|info)\b")
_TAG = re.compile(r"#\s*\d+")


def name_tokens(name: str) -> list[str]:
    """Core name tokens in original order: domains unwrapped, '#123' tags, legal forms and
    stopwords removed."""
    t = _TAG.sub(" ", _DOMAIN.sub(r"\1", anyascii(name).lower()))
    return [w for w in clean(t).split() if w not in LEGAL_FORMS and w not in STOPWORDS]


def nums(address: str) -> list[str]:
    """Every digit run in the address, leading zeros stripped, de-duplicated, in order."""
    out: list[str] = []
    for d in _DIGITS.findall(anyascii(address)):
        d = d.lstrip("0") or "0"
        if d not in out:
            out.append(d)
    return out


def addr_parts(address: str) -> list[str]:
    """Comma-separated address components, cleaned; empty components dropped."""
    return [c for c in (clean(p) for p in address.split(",")) if c]


def is_nonlatin(text: str) -> bool:
    """True if the text contains letters outside the Latin blocks (e.g. Devanagari, Tamil)."""
    return any(ch.isalpha() and ord(ch) > 0x024F for ch in text)
