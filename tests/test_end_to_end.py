"""End to end on a small synthetic dataset: generate -> train -> resolve -> validate -> index -> API.

One module-scoped fixture trains once (~15 s); every test reuses it.
"""
import time

import polars as pl
import pytest

from resolver.engine import Matcher, resolve
from resolver.evaluate import summary
from resolver.index import ResolverIndex
from resolver.io import Dataset, submission_tables, validate_submission
from resolver.synth.generator import SynthConfig, generate
from resolver.train import fit


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate(root, SynthConfig(seed=7).scaled(0.12))
    train, test = Dataset.load(root / "train"), Dataset.load(root / "test")
    matcher, report = fit(train)
    path = root / "model.joblib"
    matcher.save(path)
    return train, test, Matcher.load(path), report


def test_generator_matches_challenge_layout(tmp_path):
    stats = generate(tmp_path, SynthConfig(seed=1).scaled(0.03))
    ds = Dataset.load(tmp_path / "test")
    assert set(ds.countries) == {"US", "India", "France"}
    assert set(Dataset.load(tmp_path / "train").countries) == {"US", "India"}   # France is unseen in training
    assert ds.s1["entity_id"].str.starts_with("S1-").all()
    assert stats["test"]["singletons"] > 0 and stats["test"]["true_pairs"] > stats["test"]["s1"]


def test_generator_is_deterministic(tmp_path):
    generate(tmp_path / "a", SynthConfig(seed=3).scaled(0.02))
    generate(tmp_path / "b", SynthConfig(seed=3).scaled(0.02))
    for f in ("train/train_source2.tsv", "test/test_ground_truth.tsv"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_training_report_shows_learning_beats_rules(trained):
    _, _, _, report = trained
    steps = {r["step"]: r["f05"] for r in report["ladder"]}
    assert steps["Stage 2 ensemble"] > steps["Fuzzy score"] > steps["Exact-key rule"]
    assert report["final"]["f05"] > 0.9
    assert report["blocking"]["cand_recall"] > 0.9
    assert len(report["importance"]) == 54


def test_resolve_generalises_and_writes_valid_submission(trained):
    _, test, matcher, _ = trained
    res = resolve(test, matcher)
    m = summary(res.matches, test.truth, test.s1["entity_id"])
    assert m["f05"] > 0.9 and m["precision"] > 0.95
    fr = test.s1.filter(pl.col("country") == "France")["entity_id"]
    assert summary(res.matches, test.truth, fr)["f05"] > 0.85                   # zero-shot country
    t = submission_tables(test.s1["entity_id"], res.matches, res.candidates)
    assert t["matching_results.tsv"].height == test.s1.height
    assert validate_submission(t["matching_results.tsv"], t["candidate_pairs.tsv"], test) == []
    assert res.matches["rec"].is_unique().all()                                # exclusive assignment


def test_resolution_is_deterministic(trained):
    _, test, matcher, _ = trained
    a, b = resolve(test, matcher).matches, resolve(test, matcher).matches
    assert a.sort("s1", "rec").select("s1", "rec").equals(b.sort("s1", "rec").select("s1", "rec"))


def test_online_match_agrees_with_batch(trained):
    _, test, matcher, _ = trained
    held = test.recs.sample(60, seed=2)
    rest = Dataset(s1=test.s1, recs=test.recs.join(held.select("entity_id"), on="entity_id", how="anti"),
                   truth=test.truth, name="rest")
    index = ResolverIndex.build(rest, matcher)
    batch = dict(resolve(test, matcher).matches.select("rec", "s1").iter_rows())
    agree = 0
    for r in held.iter_rows(named=True):
        out = index.match(r["business_name"], r["business_address"], r["country"])
        online = out["decision"]["s1"] if out["decision"].get("matched") else None
        agree += online == batch.get(r["entity_id"])
        assert all(c["explanation"]["groups"] for c in out["candidates"])
    assert agree / len(held) >= 0.9


def test_validator_catches_rule_breaks(trained):
    _, test, _, _ = trained
    ids = test.s1["entity_id"]
    bad = pl.DataFrame({"source1_entity_id": ids[1:], "matched_entity_ids": ["S1-X"] * (len(ids) - 1)})
    issues = validate_submission(bad, None, test)
    assert any("missing" in i for i in issues) and any("not Source-2/3" in i for i in issues)


def test_api_end_to_end(trained, tmp_path, monkeypatch):
    import joblib
    from fastapi.testclient import TestClient

    from resolver import config
    from resolver.api import app as appmod

    _, test, matcher, report = trained
    joblib.dump(ResolverIndex.build(test, matcher), tmp_path / "index.joblib")
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "index.joblib")
    monkeypatch.setattr(config, "TRAIN_REPORT", tmp_path / "none.json")
    monkeypatch.setattr(config, "TEST_REPORT", tmp_path / "none.json")
    with TestClient(appmod.app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        meta = c.get("/api/meta").json()
        assert set(meta["countries"]) == {"US", "India", "France"} and meta["examples"]
        ex = meta["examples"][0]
        r = c.post("/api/match", json={"name": ex["name"], "address": ex["address"], "country": ex["country"]}).json()
        assert r["candidates"] and "decision" in r
        assert c.post("/api/match", json={"name": "", "address": "", "country": "US"}).status_code == 422
        lst = c.get("/api/entities", params={"limit": 5}).json()
        assert lst["total"] == test.s1.height
        assert c.get(f"/api/entities/{lst['items'][0]['s1']}").json()["entity"]["s1"] == lst["items"][0]["s1"]
        assert c.get("/api/entities/S1-NOPE").status_code == 404
        job = c.post("/api/jobs/sample").json()
        for _ in range(120):
            job = c.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert job["status"] == "done" and job["result"]["validation"]["pass"]
        assert c.get(f"/api/jobs/{job['id']}/download").headers["content-type"] == "application/zip"
        bad = c.post("/api/jobs", files={"source1": ("a.tsv", b"id\tname\n1\tx\n"), "source2": ("b.tsv", b"x"),
                                         "source3": ("c.tsv", b"x")})
        assert bad.status_code == 422
