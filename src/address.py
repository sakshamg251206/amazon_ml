"""Canonical address strings (E-A1): same place, same words.

Raw `addr` is only lower-cased / de-punctuated, so "18 R. Jean Bouin, Nantes, Loire-Atlantique" and
"18 Rue Jean Bouin, Nantes, Pays de la Loire" differ in 3 tokens, and "KS" vs "Kansas" in US.
1) whole address components that are a state / department name or code become one canonical
   state (Gironde -> nouvelle aquitaine, kansas -> ks, up -> uttar pradesh). Cities and streets are
   never touched (the learned partition aliases also map cities, so they are NOT used here);
2) street words go through a small country-aware abbreviation map (general language knowledge,
   not entity data); number prefixes (no, ndeg, hno) are dropped.
"""
import polars as pl
from rapidfuzz import fuzz

from src.features import _fuzz

_US = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
       "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace", "trl": "trail",
       "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor", "n": "north", "s": "south", "e": "east",
       "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort"}
_IN = {"rd": "road", "st": "street", "nr": "near", "opp": "opposite", "bldg": "building", "flr": "floor", "fl": "floor",
       "apt": "apartment", "sec": "sector", "ph": "phase", "ngr": "nagar", "extn": "extension", "ext": "extension",
       "dist": "district", "no": "", "hno": "", "h": ""}
_FR = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
       "all": "allee", "imp": "impasse", "rte": "route", "pl": "place", "q": "quai", "fbg": "faubourg", "ch": "chemin",
       "chem": "chemin", "sq": "square", "pass": "passage", "crs": "cours", "res": "residence", "st": "saint",
       "ste": "sainte", "ndeg": "", "no": "", "n": ""}
STREET = pl.DataFrame([(c, k, v) for c, m in (("US", _US), ("India", _IN), ("France", _FR)) for k, v in m.items()],
                      schema=["country", "tok", "canon"], orient="row")
_IN_ST = {"up": "uttar pradesh", "mh": "maharashtra", "ka": "karnataka", "tn": "tamil nadu", "gj": "gujarat",
          "wb": "west bengal", "ts": "telangana", "tg": "telangana", "hr": "haryana", "kl": "kerala", "rj": "rajasthan",
          "br": "bihar", "mp": "madhya pradesh", "ap": "andhra pradesh", "od": "orissa", "or": "orissa", "odisha": "orissa",
          "pb": "punjab", "dl": "delhi", "jh": "jharkhand", "cg": "chhattisgarh", "uk": "uttarakhand", "hp": "himachal pradesh",
          "jk": "jammu and kashmir", "as": "assam", "ga": "goa"}
_FR_ST = {"nord": "hauts de france", "pas de calais": "hauts de france", "somme": "hauts de france", "aisne": "hauts de france",
          "oise": "hauts de france", "hdf": "hauts de france", "gironde": "nouvelle aquitaine", "landes": "nouvelle aquitaine",
          "dordogne": "nouvelle aquitaine", "lot et garonne": "nouvelle aquitaine", "pyrenees atlantiques": "nouvelle aquitaine",
          "charente": "nouvelle aquitaine", "charente maritime": "nouvelle aquitaine", "naq": "nouvelle aquitaine",
          "loire atlantique": "pays de la loire", "vendee": "pays de la loire", "maine et loire": "pays de la loire",
          "sarthe": "pays de la loire", "mayenne": "pays de la loire", "pdl": "pays de la loire"}


def _states() -> pl.DataFrame:
    from src.partition import US_STATES   # full name -> abbreviation
    rows = [("US", k, v) for k, v in US_STATES.items()] + [("India", k, v) for k, v in _IN_ST.items()] \
         + [("France", k, v) for k, v in _FR_ST.items()]
    return pl.DataFrame(rows, schema=["country", "part", "state"], orient="row")


STATES = _states()
CA = ["ca_tset", "ca_sort", "ca_jacc"]


def canon_addr(df: pl.DataFrame, alias: pl.DataFrame = STATES) -> pl.DataFrame:
    """df: (entity_id, country, parts) -> (entity_id, canon)."""
    p = df.select("entity_id", "country", "parts").with_row_index("ri").explode("parts", empty_as_null=True) \
          .with_row_index("pi").rename({"parts": "part"})
    p = p.join(alias.rename({"state": "st_canon"}), on=["country", "part"], how="left", maintain_order="left") \
         .with_columns(pl.coalesce("st_canon", "part").alias("comp"))
    t = p.select("ri", "pi", "entity_id", "country", pl.col("comp").str.split(" ").alias("tok")).explode("tok", empty_as_null=True) \
         .with_row_index("ti")
    t = t.join(STREET, on=["country", "tok"], how="left", maintain_order="left") \
         .with_columns(pl.coalesce("canon", "tok").alias("w")).filter(pl.col("w").is_not_null() & (pl.col("w") != ""))
    out = t.sort("ti").group_by("ri", maintain_order=True).agg(pl.col("entity_id").first(), pl.col("w").str.join(" ").alias("canon"))
    return df.select("entity_id").join(out.select("entity_id", "canon"), on="entity_id", how="left").with_columns(pl.col("canon").fill_null(""))


def canon_features(pairs: pl.DataFrame, c1: pl.DataFrame, c2: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, rec); c1/c2: (entity_id, canon) for S1 / records -> (s1, rec, *CA)."""
    d = pairs.select("s1", "rec").join(c1.rename({"entity_id": "s1", "canon": "a1"}), on="s1", how="left") \
             .join(c2.rename({"entity_id": "rec", "canon": "a2"}), on="rec", how="left").with_columns(pl.col("a1", "a2").fill_null(""))
    t1, t2 = pl.col("a1").str.split(" ").list.unique(), pl.col("a2").str.split(" ").list.unique()
    return d.select("s1", "rec",
                    _fuzz(d["a1"], d["a2"], fuzz.token_set_ratio).alias("ca_tset"),
                    _fuzz(d["a1"], d["a2"], fuzz.token_sort_ratio).alias("ca_sort"),
                    (t1.list.set_intersection(t2).list.len() / t1.list.set_union(t2).list.len().clip(1)).alias("ca_jacc"),
                    ).with_columns(pl.col(x).cast(pl.Float32) for x in CA)


if __name__ == "__main__":
    df = pl.DataFrame({"entity_id": ["a", "b", "c", "d"], "country": ["France", "France", "US", "US"],
                       "parts": [["18 rue jean bouin", "nantes", "pays de la loire"], ["18 r jean bouin", "nantes", "loire atlantique"],
                                 ["3906 33rd ter", "topeka", "ks"], ["3906 33rd terrace", "topeka", "kansas"]]})
    c = canon_addr(df)
    assert c["canon"][0] == c["canon"][1] == "18 rue jean bouin nantes pays de la loire", c
    assert c["canon"][2] == "3906 33rd terrace topeka ks" and c["canon"][3] == c["canon"][2], c
    f = canon_features(pl.DataFrame({"s1": ["a"], "rec": ["b"]}), c.filter(pl.col("entity_id") == "a"), c.filter(pl.col("entity_id") == "b"))
    assert f["ca_tset"][0] == 100 and f["ca_jacc"][0] == 1.0
    print("address self-check ok")
