import polars as pl
from src.baseline_rule import add_keys, rule_matches

COLS = ["entity_id", "business_name", "business_address", "country"]
S1 = pl.DataFrame([
    ["S1-1", "Yazzie Empire Enterprises LLC", "24800 Euclid Avenue, Euclid, OH", "US"],
    ["S1-2", "Primary Care Group", "10 Main St, Austin, TX", "US"],     # chain: same key as S1-3
    ["S1-3", "Primary Care Group", "10 Main Street, Austin, Texas", "US"],
    ["S1-4", "Lonely Shop", "5 Rue de Valmy, Lille", "France"],         # no records
], schema=COLS, orient="row")


def test_add_keys_drops_empty_keys():
    df = pl.DataFrame([["S2-9", "LLC", "1 X St", "US"], ["S2-8", "Acme", "", "US"]],
                      schema=COLS, orient="row")
    assert add_keys(df).height == 0


def test_rule_matches_exact_key_and_rejects_number_change():
    recs = pl.DataFrame([
        ["S2-1", "YAZZIE EMPIRE  ENTERPRISES", "24800 EUCLID AVE, EUCLID, OH", "US"],
        ["S3-1", "Yazzie Empire Enterprises", "24801 Euclid Ave, Euclid, Ohio", "US"],  # neighbour
    ], schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {"S1-1": ["S2-1"]}


def test_rule_skips_ambiguous_s1_keys():
    recs = pl.DataFrame([["S2-5", "PRIMARY CARE GROUP", "10 MAIN ST, AUSTIN, TX", "US"]],
                        schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {}


def test_rule_never_crosses_country():
    recs = pl.DataFrame([["S2-7", "Lonely Shop", "5 Main St", "US"]], schema=COLS, orient="row")
    assert rule_matches(S1, recs) == {}
