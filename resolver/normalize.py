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


# ---- frame-level normalisation (raw TSV columns -> every field the pipeline uses) ----
NORM_COLS = ["entity_id", "country", "empty_addr", "name_tok", "addr", "parts", "nums", "nonlatin",
             "name_key", "name_ns", "hn"]


def _norm_rows(rows: tuple[list[str], list[str]]) -> dict[str, list]:
    names, addrs = rows
    out: dict[str, list] = {"name_tok": [], "addr": [], "parts": [], "nums": [], "nonlatin": []}
    for n, a in zip(names, addrs):
        out["name_tok"].append(" ".join(name_tokens(n)))
        out["addr"].append(clean(a))
        out["parts"].append(addr_parts(a))
        out["nums"].append(nums(a))
        out["nonlatin"].append(is_nonlatin(n))
    return out


def normalize_frame(df, workers: int = 1, chunk: int = 125_000):
    """df: (entity_id, business_name, business_address, country) -> NORM_COLS.
    Per-row Python is the slow part; `workers` > 1 spreads chunks over processes."""
    import polars as pl
    df = df.with_columns(pl.col("business_name", "business_address", "country").fill_null(""))
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)] or [([], [])]
    if workers > 1 and len(chunks) > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            parts = pool.map(_norm_rows, chunks)
    else:
        parts = [_norm_rows(c) for c in chunks]
    cols = {k: [v for p in parts for v in p[k]] for k in parts[0]}
    return df.select("entity_id", "country", (pl.col("business_address").str.strip_chars() == "").alias("empty_addr")).with_columns(
        pl.Series("name_tok", cols["name_tok"], dtype=pl.String),
        pl.Series("addr", cols["addr"], dtype=pl.String),
        pl.Series("parts", cols["parts"], dtype=pl.List(pl.String)),
        pl.Series("nums", cols["nums"], dtype=pl.List(pl.String)),
        pl.Series("nonlatin", cols["nonlatin"], dtype=pl.Boolean),
    ).with_columns(
        pl.col("name_tok").str.split(" ").list.sort().list.unique(maintain_order=True)
        .list.join(" ").alias("name_key"),
        pl.col("name_tok").str.replace_all(" ", "").alias("name_ns"),
        pl.col("nums").list.first().fill_null("").alias("hn"),
    )
