"""Synthetic multi-source business data in the exact Amazon ML Challenge 2026 format.

Why: the competition data cannot be redistributed, yet the system must run end to end for anyone.
The generator reproduces the structure and the noise the project measured on the real data:

* Source 1 is a deduplicated reference; each S1 entity has 0..n copies spread over Source 2/3
  (about 6% singletons, ~3 copies per S1).
* Copies carry name noise (legal-form variants, abbreviations, typos, casing, word order, '&'/'and',
  DBA and domain forms, store tags, Devanagari script, phonetic romanisation) and address noise
  (abbreviations, missing / reordered components, zip/PIN codes, landmarks, house-number formats,
  shifted house numbers, empty addresses, lost accents).
* Hard negatives that are NOT copies: "neighbour" fakes that add generic words (Holdings, Group,
  Développement...) next door, co-located businesses at the same address, chain branches with the
  same brand in different places, and unrelated background businesses.
* Train covers US and India; test adds France, a country the model never sees in training
  (the competition's open-set requirement).

Output: `<out>/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `<out>/test/test_source{1,2,3}.tsv`,
and (unlike the competition, for demo evaluation only) `test_ground_truth.tsv`.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from resolver.synth import vocab as V
from resolver.synth.translit import to_devanagari, phonetic_romanise

COUNTRY_NAMES = {"US": "US", "IN": "India", "FR": "France"}


@dataclass
class Entity:
    """One real-world business. Its S1 row is the canonical rendering."""
    country: str                 # "US" | "India" | "France"
    state: str                   # state code / region key
    city: str
    name_core: list[str]         # core name words in order
    legal: str                   # canonical legal form ("" if none)
    house: str
    street: str                  # US/FR: street name; India: road name
    street_type: str = ""        # US/FR street type (canonical long form)
    unit: str = ""
    locality: str = ""           # India
    building: str = ""           # India
    postcode: str = ""
    chain_tag: str = ""          # "#1234" for chain branches
    devanagari_ok: bool = False
    s1_id: str = ""
    copies: list[str] = field(default_factory=list)


# ======================================================================== helpers
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.3:                                      # transpose
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op < 0.55:                                     # delete
        return word[:i] + word[i + 1:]
    if op < 0.8:                                      # substitute with a keyboard-ish neighbour vowel/consonant
        pool = "aeiou" if word[i].lower() in "aeiou" else "bcdfghklmnprstvz"
        return word[:i] + rng.choice(pool) + word[i + 1:]
    return word[:i] + word[i] + word[i:]               # duplicate


def _case(s: str, rng: random.Random, p_upper: float, p_lower: float) -> str:
    r = rng.random()
    if r < p_upper:
        return s.upper()
    if r < p_upper + p_lower:
        return s.lower()
    return s


def _disp(text: str) -> str:
    """Canonical French display form (with accents)."""
    return V.FR_ACCENTS.get(text, text)


# ======================================================================== entity sampling
class EntityFactory:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used_names: set[tuple[str, str]] = set()

    def _place(self, country: str):
        r = self.rng
        if country == "US":
            st = r.choice(list(V.US_STATES))
            return st, r.choice(V.US_STATES[st][1])
        if country == "India":
            st = r.choice(list(V.IN_STATES))
            return st, r.choice(V.IN_STATES[st][1])
        st = r.choice(list(V.FR_REGIONS))
        return st, r.choice(V.FR_REGIONS[st][1])

    def _name(self, country: str) -> tuple[list[str], str]:
        r = self.rng
        if country == "US":
            p = r.random()
            ind = r.choice(V.US_INDUSTRIES).split()
            if p < 0.35:
                core = [r.choice(V.US_SURNAMES)] + ind
            elif p < 0.5:
                core = [r.choice(V.US_SURNAMES), "&", "Sons"] + ind
            elif p < 0.8:
                core = [r.choice(V.US_ADJ), r.choice(V.US_NOUNS)] + ind
            elif p < 0.9:
                core = [r.choice(V.US_SURNAMES), "&", r.choice(V.US_SURNAMES)] + (ind if r.random() < 0.5 else [])
            else:
                core = [r.choice(V.US_ADJ), r.choice(V.US_SURNAMES)] + ind
            legal = r.choice(V.US_LEGAL) if r.random() < 0.75 else ""
        elif country == "India":
            p = r.random()
            ind = r.choice(V.IN_INDUSTRIES).split()
            if p < 0.35:
                core = [r.choice(V.IN_PREFIX), r.choice(V.IN_PREFIX + V.IN_WORDS)] + ind
            elif p < 0.6:
                core = [r.choice(V.IN_SURNAMES)] + ind
            elif p < 0.85:
                core = [r.choice(V.IN_WORDS)] + ind
            else:
                core = [r.choice(V.IN_SURNAMES), "&", "Sons"] + (ind if r.random() < 0.4 else [])
            legal = r.choice(V.IN_LEGAL) if r.random() < 0.7 else ""
        else:
            p = r.random()
            if p < 0.4:
                core = r.choice(V.FR_TRADES).split() + [r.choice(V.FR_SURNAMES)]
            elif p < 0.6:
                core = [r.choice(V.FR_SURNAMES)] + r.choice(V.FR_SUFFIX_WORDS).split()
            elif p < 0.75:
                core = [r.choice(V.FR_SURNAMES), "et", "Fils"]
            else:
                core = r.choice(V.FR_TRADES).split() + r.choice(V.FR_PLACES).split()
            legal = r.choice(V.FR_LEGAL) if r.random() < 0.7 else ""
        return core, legal

    def _address(self, e: Entity) -> None:
        r = self.rng
        if e.country == "US":
            e.house = str(r.choice([r.randint(1, 999), r.randint(100, 9999), r.randint(1000, 29999)]))
            e.street = r.choice(V.US_STREET_NAMES) if r.random() < 0.85 else f"{r.randint(2, 99)}th"
            e.street_type = r.choice(list(V.US_STREET_TYPES))
            if r.random() < 0.25:
                e.unit = r.choice(V.US_UNITS).format(n=r.randint(100, 950), f=r.randint(2, 12), l=r.choice("ABCDE"))
            e.postcode = V.US_STATES[e.state][2] + f"{r.randint(0, 9999):04d}"
        elif e.country == "India":
            e.house = str(r.randint(1, 450))
            e.street = r.choice(V.IN_STREETS)
            e.locality = r.choice(V.IN_LOCALITIES)
            if r.random() < 0.35:
                e.building = r.choice(V.IN_BUILDINGS)
            if r.random() < 0.3:
                e.unit = r.choice(V.IN_UNITS).format(n=r.randint(1, 60))
            e.postcode = V.IN_STATES[e.state][2] + f"{r.randint(0, 99999):05d}"
            e.devanagari_ok = e.state in V.IN_DEVANAGARI
        else:
            e.house = str(r.randint(1, 180)) + (r.choice(["", "", "", " bis", " ter"]))
            e.street_type = r.choice(list(V.FR_STREET_TYPES))
            e.street = r.choice(V.FR_STREET_NAMES)
            if r.random() < 0.15:
                e.unit = r.choice(V.FR_UNITS).format(n=r.randint(1, 90), f=r.randint(1, 6), l=r.choice("ABCD"))
            e.postcode = V.FR_REGIONS[e.state][2] + f"{r.randint(0, 999):03d}"

    def entity(self, country: str) -> Entity:
        st, city = self._place(country)
        for _ in range(20):  # S1 is deduplicated: avoid an identical (name, city)
            core, legal = self._name(country)
            if (" ".join(core), city) not in self.used_names:
                break
        self.used_names.add((" ".join(core), city))
        e = Entity(country=country, state=st, city=city, name_core=core, legal=legal, house="", street="")
        self._address(e)
        return e

    def branch(self, brand: str, country: str, store_no: int) -> Entity:
        e = self.entity(country)
        e.name_core, e.legal = brand.split(), ""
        if self.rng.random() < 0.5:
            e.chain_tag = f"#{store_no}"
        return e


# ======================================================================== rendering
def state_name(e: Entity) -> str:
    return {"US": lambda: V.US_STATES[e.state][0], "India": lambda: V.IN_STATES[e.state][0],
            "France": lambda: _disp(V.FR_REGIONS[e.state][0])}[e.country]()


def render_s1(e: Entity) -> tuple[str, str]:
    """Canonical reference rendering. The address always ends with the state/region (as on S1)."""
    name = " ".join(e.name_core + ([e.chain_tag] if e.chain_tag else []) + ([e.legal] if e.legal else []))
    if e.country == "US":
        parts = [f"{e.house} {e.street} {e.street_type}"] + ([e.unit] if e.unit else []) + [e.city, e.state]
    elif e.country == "India":
        first = ", ".join(([e.unit] if e.unit else []) + ([e.building] if e.building else []))
        parts = ([first] if first else []) + [f"No {e.house}", e.street, e.locality, e.city, state_name(e)]
    else:
        parts = [f"{e.house} {e.street_type} {e.street}"] + ([e.unit] if e.unit else []) + \
                [f"{e.postcode} {_disp(e.city)}", state_name(e)]
    return name, ", ".join(parts)


class Noise:
    """Per-copy perturbations. `level` scales every probability (S3 is noisier than S2)."""

    def __init__(self, rng: random.Random, level: float = 1.0):
        self.r, self.k = rng, level

    def p(self, prob: float) -> bool:
        return self.r.random() < min(prob * self.k, 0.95)

    # ---------------------------------------------------------------- names
    def name(self, e: Entity) -> str:
        r = self.r
        words = list(e.name_core)
        if e.chain_tag and not self.p(0.3):
            words.append(e.chain_tag)
        if "&" in words and self.p(0.45):
            words[words.index("&")] = "and"
        if len(words) > 2 and self.p(0.06):                      # drop a trailing descriptor word
            words = words[:-1]
        if len(words) >= 2 and self.p(0.06):                     # word-order transposition
            i = r.randrange(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
        if self.p(0.15):
            words = [V.WORD_ABBREV.get(w, w) for w in words]
        if self.p(0.18):
            i = r.randrange(len(words))
            words[i] = _typo(words[i], r)
        if self.p(0.04):
            i = r.randrange(len(words))
            words[i] = _typo(words[i], r)
        legal = e.legal
        if legal:
            if self.p(0.25):
                legal = ""
            elif self.p(0.6):
                legal = r.choice(V.LEGAL_VARIANTS.get(legal, [legal]))
        elif self.p(0.08):
            legal = r.choice({"US": ["LLC", "Inc"], "India": ["Pvt Ltd", "Ltd"], "France": ["SARL", "SAS"]}[e.country])
        name = " ".join(words + ([legal] if legal else []))
        if e.country == "India" and e.devanagari_ok and self.p(0.12):
            return to_devanagari(name)
        if e.country == "India" and self.p(0.05):
            name = phonetic_romanise(name, r)
        if e.country == "France" and self.p(0.5):
            name = _strip_accents(name)
        if self.p(0.02):                                          # domain form
            return re.sub(r"[^a-z0-9]", "", "".join(e.name_core).lower()) + r.choice([".com", ".net", ".in" if e.country == "India" else ".fr" if e.country == "France" else ".us"])
        if self.p(0.02):                                          # DBA form
            name = f"{r.choice(V.US_SURNAMES)} Holdings DBA {name}" if e.country == "US" else f"{name} ({r.choice(V.IN_SURNAMES if e.country == 'India' else V.FR_SURNAMES)})"
        return _case(name, r, 0.2 * self.k, 0.08 * self.k)

    # ---------------------------------------------------------------- addresses
    def house(self, e: Entity) -> str:
        h = e.house
        if self.p(0.025):                                         # small shift: true copy, different number
            base = int(re.match(r"\d+", h).group())
            h = str(max(1, base + self.r.choice([-4, -2, -1, 1, 2, 4]))) + h[len(str(base)):]
        if e.country == "India" and self.p(0.5):
            h = self.r.choice(V.IN_HOUSE_FMT).format(h=h, s=self.r.randint(1, 9))
        elif e.country == "US" and self.p(0.05):
            h = self.r.choice([f"#{h}", f"{h}-A", f"{h}{self.r.choice('AB')}"])
        return h

    def address(self, e: Entity) -> str:
        r = self.r
        if self.p(0.05):
            return ""
        if e.country == "US":
            st_type = V.US_STREET_TYPES[e.street_type] if self.p(0.55) else e.street_type
            street = e.street if not self.p(0.07) else _typo(e.street, r)
            parts = [] if self.p(0.06) else [self.house(e)]
            line = " ".join(parts + [street, st_type])
            comps = [line]
            if e.unit and not self.p(0.4):
                comps.append(e.unit)
            elif not e.unit and self.p(0.05):
                comps.append(f"Suite {r.randint(100, 900)}")
            if not self.p(0.06):
                comps.append(e.city)
            state = e.state
            if self.p(0.25):
                state = V.US_STATES[e.state][0]
            if self.p(0.12):
                state = ""
            if self.p(0.4):
                state = (state + " " + e.postcode).strip()
            if state:
                comps.append(state)
        elif e.country == "India":
            comps = []
            if e.unit and not self.p(0.5):
                comps.append(e.unit)
            if e.building and not self.p(0.4):
                comps.append(e.building)
            if not self.p(0.08):
                comps.append(self.house(e))
            road = e.street
            if self.p(0.4):
                road = road.replace("Road", "Rd").replace("Street", "St")
            if self.p(0.15):
                road = road.replace("MG ", "M.G. ").replace("SV ", "S.V. ").replace("GT ", "G.T. ")
            comps.append(road if not self.p(0.06) else _typo(road, r))
            if self.p(0.25):
                comps.append(r.choice(V.IN_LANDMARKS))
            if not self.p(0.2):
                comps.append(e.locality)
            city = e.city
            if self.p(0.35):
                city = f"{city} - {e.postcode}" if self.p(0.5) else f"{city} {e.postcode}"
            if not self.p(0.05):
                comps.append(city)
            if not self.p(0.15):
                comps.append(e.state if self.p(0.12) else state_name(e))
        else:
            st_type = V.FR_STREET_TYPES[e.street_type] if self.p(0.4) else e.street_type
            comps = [" ".join(([] if self.p(0.06) else [self.house(e)]) + [st_type, e.street])]
            if e.unit and not self.p(0.5):
                comps.append(e.unit)
            city = _disp(e.city)
            comps.append(f"{e.postcode} {city}" if not self.p(0.3) else city)
            if not self.p(0.3):
                comps.append(state_name(e))
        if len(comps) > 2 and self.p(0.08):                      # component reordering
            i, j = r.sample(range(len(comps)), 2)
            comps[i], comps[j] = comps[j], comps[i]
        addr = ", ".join(c for c in comps if c)
        if e.country == "France" and self.p(0.5):
            addr = _strip_accents(addr)
        return _case(addr, r, 0.15 * self.k, 0.05 * self.k)


# ======================================================================== distractors
def neighbour_fake(e: Entity, rng: random.Random) -> Entity:
    """Same core name + a generic word, usually next door: the hardest negatives."""
    f = Entity(**{**e.__dict__, "copies": []})
    words = list(e.name_core)
    g = rng.choice(V.GENERIC_WORDS[e.country])
    if rng.random() < 0.7:
        words.append(g)
    else:
        words.insert(rng.randrange(len(words) + 1), g)
    f.name_core = words
    if rng.random() < 0.5:
        f.legal = rng.choice({"US": V.US_LEGAL, "India": V.IN_LEGAL, "France": V.FR_LEGAL}[e.country])
    m = re.match(r"\d+", e.house)
    base = int(m.group()) if m else 1
    r = rng.random()
    if r < 0.55:
        f.house = str(max(1, base + rng.choice([-30, -12, -6, -2, 2, 4, 8, 20, 50])))
    elif r < 0.75:
        f.house = e.house
    else:
        f.house = str(rng.randint(1, 4000))
    f.unit = "" if rng.random() < 0.5 else e.unit
    return f


def co_located(e: Entity, factory: EntityFactory) -> Entity:
    """A different business in the same building."""
    other = factory.entity(e.country)
    return Entity(**{**e.__dict__, "name_core": other.name_core, "legal": other.legal, "chain_tag": "",
                     "unit": other.unit, "copies": []})


# ======================================================================== dataset assembly
@dataclass
class SynthConfig:
    seed: int = 42
    train: dict[str, int] = field(default_factory=lambda: {"US": 4000, "India": 4000})
    test: dict[str, int] = field(default_factory=lambda: {"US": 1500, "India": 1500, "France": 1500})
    p_singleton: float = 0.06
    mean_copies: float = 2.4          # copies beyond the first, Poisson mean
    p_fake: float = 0.18              # share of S1 with neighbour fakes (doubled for singletons)
    p_colocated: float = 0.05
    p_chain: float = 0.06
    p_background: float = 0.08        # unrelated records, as a share of S1
    noise_s2: float = 0.85
    noise_s3: float = 1.25

    def scaled(self, scale: float) -> "SynthConfig":
        return SynthConfig(**{**self.__dict__, "train": {k: max(50, int(v * scale)) for k, v in self.train.items()},
                              "test": {k: max(50, int(v * scale)) for k, v in self.test.items()}})


def _poisson(lam: float, rng: random.Random) -> int:
    # Knuth; lam is small
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def generate_split(split: str, counts: dict[str, int], cfg: SynthConfig, rng: random.Random):
    """-> (s1_rows, s2_rows, s3_rows, truth) with rows = (entity_id, name, address, country)."""
    factory = EntityFactory(rng)
    ents: list[Entity] = []
    for country, n in counts.items():
        n_chain = int(n * cfg.p_chain)
        brands = V.CHAINS[country]
        k = 0
        while k < n_chain:                              # chain branches (2..6 per brand)
            brand = rng.choice(brands)
            for _ in range(min(rng.randint(2, 6), n_chain - k)):
                ents.append(factory.branch(brand, country, rng.randint(100, 9999)))
                k += 1
        ents += [factory.entity(country) for _ in range(n - n_chain)]
    rng.shuffle(ents)
    for i, e in enumerate(ents):
        e.s1_id = f"S1-{split[:2].upper()}{i + 1:06d}"

    recs: list[tuple[Entity, int, str]] = []            # (entity to render, source, owner s1 id or "")
    for e in ents:
        n_copies = 0 if rng.random() < cfg.p_singleton else 1 + min(_poisson(cfg.mean_copies, rng), 9)
        for _ in range(n_copies):
            recs.append((e, 2 if rng.random() < 0.5 else 3, e.s1_id))
        if rng.random() < cfg.p_fake * (2 if n_copies == 0 else 1):
            fake = neighbour_fake(e, rng)
            for _ in range(1 + min(_poisson(0.8, rng), 3)):      # phantom entities also have copies
                recs.append((fake, 2 if rng.random() < 0.5 else 3, ""))
        if rng.random() < cfg.p_colocated:
            recs.append((co_located(e, factory), 2 if rng.random() < 0.5 else 3, ""))
    for country, n in counts.items():
        for _ in range(int(n * cfg.p_background)):
            b = factory.entity(country)
            for _ in range(1 + min(_poisson(0.5, rng), 3)):
                recs.append((b, 2 if rng.random() < 0.5 else 3, ""))
    rng.shuffle(recs)

    s1_rows = [(e.s1_id, *render_s1(e), e.country) for e in ents]
    noise = {2: Noise(rng, cfg.noise_s2), 3: Noise(rng, cfg.noise_s3)}
    rows: dict[int, list[tuple]] = {2: [], 3: []}
    truth: dict[str, list[str]] = {e.s1_id: [] for e in ents}
    for e, src, owner in recs:
        rid = f"S{src}-{split[:2].upper()}{len(rows[src]) + 1:07d}"
        nz = noise[src]
        name = nz.name(e)
        rows[src].append((rid, name if name.strip() else " ".join(e.name_core), nz.address(e), e.country))
        if owner:
            truth[owner].append(rid)
    return s1_rows, rows[2], rows[3], truth


def _clean_field(s: str) -> str:
    return s.replace("\t", " ").replace("\n", " ").strip()


def write_split(out: Path, split: str, s1, s2, s3, truth, with_truth: bool) -> None:
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    for i, rows in ((1, s1), (2, s2), (3, s3)):
        with open(d / f"{split}_source{i}.tsv", "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for rid, name, addr, country in rows:
                f.write(f"{rid}\t{_clean_field(name)}\t{_clean_field(addr)}\t{country}\n")
    if with_truth:
        with open(d / f"{split}_ground_truth.tsv", "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, recs in truth.items():
                f.write(f"{s1_id}\t{','.join(sorted(recs))}\n")


def generate(out: Path | str, cfg: SynthConfig | None = None, test_truth: bool = True) -> dict:
    """Write a complete train + test dataset. Returns row counts."""
    cfg = cfg or SynthConfig()
    out = Path(out)
    stats = {}
    for split, counts, seed_off in (("train", cfg.train, 0), ("test", cfg.test, 1)):
        rng = random.Random(cfg.seed * 1000 + seed_off)
        s1, s2, s3, truth = generate_split(split, counts, cfg, rng)
        write_split(out, split, s1, s2, s3, truth, with_truth=(split == "train" or test_truth))
        n_true = sum(len(v) for v in truth.values())
        stats[split] = {"s1": len(s1), "s2": len(s2), "s3": len(s3), "true_pairs": n_true,
                        "singletons": sum(1 for v in truth.values() if not v)}
    return stats
