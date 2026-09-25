"""Label-free state/region inference used to partition blocking by (country, state).

S1 addresses always end with their state. That lets us learn, from S1 alone:
  * which address components (states, cities) point to exactly one state (alias map), then
  * from S2/S3 co-occurrence, further aliases such as full state names, department names,
    region codes and transliterated native-script names.
No labels are used, so the same code runs on train and on test (including France).
"""
import polars as pl

MIN_COUNT = 5
MIN_PURITY = 0.98
EXPAND_COUNT = 20
EXPAND_PURITY = 0.95

# Generic domain knowledge (no lookup): US state full names -> postal codes. Needed because
# towns share state names (Indiana PA, Maryland NY) and S1 only ever writes the codes.
_US_NAMES = (
    "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia "
    "hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts "
    "michigan minnesota mississippi missouri montana nebraska nevada new_hampshire new_jersey "
    "new_mexico new_york north_carolina north_dakota ohio oklahoma oregon pennsylvania "
    "rhode_island south_carolina south_dakota tennessee texas utah vermont virginia washington "
    "west_virginia wisconsin wyoming district_of_columbia").split()
_US_CODES = ("al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne "
             "nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc").split()
US_STATES = {n.replace("_", " "): c for n, c in zip(_US_NAMES, _US_CODES, strict=True)}


def _aliases_from(pairs: pl.DataFrame, min_count: int, min_purity: float) -> pl.DataFrame:
    """pairs: (country, part, state) rows -> aliases (country, part, state) with high purity."""
    c = pairs.group_by("country", "part", "state").len("n")
    tot = c.group_by("country", "part").agg(pl.col("n").sum().alias("tot"))
    return (c.join(tot, on=["country", "part"])
             .filter((pl.col("n") >= min_count) & (pl.col("n") / pl.col("tot") >= min_purity))
             .select("country", "part", "state"))


def s1_state(s1: pl.DataFrame) -> pl.Series:
    """An S1's state = its digit-free address part that is most frequent across all S1 of the
    country (states outnumber any city inside them). Robust to component reordering.
    Ties go to the later part (states usually come last)."""
    p = (s1.select(pl.int_range(pl.len()).alias("rid"), "country", "parts")
           .explode("parts", empty_as_null=True).rename({"parts": "part"}).drop_nulls()
           .with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))
           .filter(~pl.col("part").str.contains(r"\d")))
    p = p.with_columns(pl.len().over("country", "part").alias("freq"))
    best = (p.sort(["rid", "freq", "pos"], descending=[False, True, True])
              .group_by("rid", maintain_order=True).first())
    return (pl.DataFrame({"rid": pl.int_range(s1.height, eager=True)})
              .join(best.select("rid", pl.col("part").alias("state")), on="rid", how="left")["state"])


def learn_aliases(s1: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    s1p = (s1.select("country", "parts", s1_state(s1).alias("state"))
             .explode("parts", empty_as_null=True).rename({"parts": "part"}).drop_nulls())
    alias = _aliases_from(s1p, MIN_COUNT, MIN_PURITY)
    # Expansion: parts of S2/S3 records that co-occur with a single known state.
    rp = recs.select(pl.int_range(pl.len()).alias("rid"), "country", "parts").explode("parts", empty_as_null=True) \
             .rename({"parts": "part"}).drop_nulls()
    known = rp.join(alias, on=["country", "part"])
    one_state = known.group_by("rid").agg(pl.col("state").unique()).filter(pl.col("state").list.len() == 1) \
                     .with_columns(pl.col("state").list.first())
    cooc = rp.join(one_state, on="rid").join(alias.select("country", "part"), on=["country", "part"], how="anti")
    extra = _aliases_from(cooc.select("country", "part", "state"), EXPAND_COUNT, EXPAND_PURITY)
    override = pl.DataFrame({"country": "US", "part": list(US_STATES), "state": list(US_STATES.values())})
    learned = pl.concat([alias, extra]).join(override.select("country", "part"), on=["country", "part"], how="anti")
    return pl.concat([learned, override]).unique(["country", "part"])


def infer_state(df: pl.DataFrame, alias: pl.DataFrame) -> pl.Series:
    """Majority state over a record's alias-matching address parts; null if none match."""
    rp = df.select(pl.int_range(pl.len()).alias("rid"), "country", "parts").explode("parts", empty_as_null=True) \
           .rename({"parts": "part"}).drop_nulls()
    votes = (rp.join(alias, on=["country", "part"]).group_by("rid", "state").len("n")
               .sort(["rid", "n", "state"], descending=[False, True, False])
               .group_by("rid", maintain_order=True).first())
    return (pl.DataFrame({"rid": pl.int_range(df.height, eager=True)})
              .join(votes, on="rid", how="left")["state"])
