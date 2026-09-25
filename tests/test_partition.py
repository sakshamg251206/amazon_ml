import polars as pl

import src.partition as P


def _s1(rows):
    return pl.DataFrame({"country": [r[0] for r in rows], "parts": [r[1] for r in rows]})


def test_learns_city_alias_from_s1_and_full_name_from_records(monkeypatch):
    monkeypatch.setattr(P, "MIN_COUNT", 2)
    monkeypatch.setattr(P, "EXPAND_COUNT", 2)
    s1 = _s1([("US", ["1 main st", "euclid", "oh"])] * 3 + [("US", ["5 elm st", "austin", "tx"])] * 3)
    recs = _s1([("US", ["ohio", "euclid", "1 main st"])] * 3 + [("US", ["austin", "texas"])] * 3)
    alias = P.learn_aliases(s1, recs)
    got = {(r["part"], r["state"]) for r in alias.iter_rows(named=True)}
    assert ("euclid", "oh") in got and ("ohio", "oh") in got and ("texas", "tx") in got
    # street components shared across states never become aliases
    s1_mixed = _s1([("US", ["main st", "a", "oh"])] * 3 + [("US", ["x st", "c", "oh"])] * 3
                   + [("US", ["y st", "e", "oh"])] * 3 + [("US", ["main st", "b", "tx"])] * 3
                   + [("US", ["z st", "d", "tx"])] * 3 + [("US", ["w st", "f", "tx"])] * 3)
    assert "main st" not in set(P.learn_aliases(s1_mixed, s1_mixed)["part"])


def test_infer_state_votes_and_returns_null_when_unknown():
    alias = pl.DataFrame({"country": ["US", "US"], "part": ["euclid", "ohio"], "state": ["oh", "oh"]})
    df = _s1([("US", ["x", "euclid", "ohio"]), ("US", ["nowhere"]), ("France", ["euclid"])])
    assert P.infer_state(df, alias).to_list() == ["oh", None, None]


def test_s1_state_is_most_frequent_part_even_when_reordered():
    s1 = _s1([("US", ["fostoria", "oh", "534 stinchomb dr"]),
              ("US", ["1 main st", "euclid", "oh"]),
              ("US", ["5 elm st", "toledo", "oh"])])
    assert P.s1_state(s1).to_list() == ["oh", "oh", "oh"]


def test_us_state_full_names_override_same_named_towns(monkeypatch):
    monkeypatch.setattr(P, "MIN_COUNT", 2)
    s1 = _s1([("US", ["1 main st", "indiana", "pa"])] * 3 + [("US", ["2 oak st", "gary", "in"])] * 5)
    alias = P.learn_aliases(s1, s1)
    got = dict(zip(alias["part"], alias["state"]))
    assert got["indiana"] == "in" and got["gary"] == "in"
