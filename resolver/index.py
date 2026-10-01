"""A resolved dataset held in memory for interactive use.

* browse: every S1 entity with its predicted cluster, candidates, probabilities and (when labels
  exist) whether each decision was right;
* explain: exact stage-1 SHAP contributions for any candidate pair;
* match: score a NEW record against the reference exactly as if it had been part of the batch:
  same partition index (TF-IDF fitted on S1 only), same exact keys, record- and S1-side
  competition features against the stored candidates, collective features against the stored
  stage-1 scores, twin links through the stored key graph, and expected-F0.5 decoding of the S1's
  existing cluster plus the new record.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from resolver import engine as E
from resolver.block import CAP, KEYS, with_keys
from resolver.collective import MAX_GROUP, TWIN_COLS, twin_keys
from resolver.decode import ef_decode, exclusive
from resolver.evaluate import per_s1, summary
from resolver.explain import explain_rows
from resolver.features2 import s1_maxima
from resolver.normalize import normalize_frame
from resolver.partition import infer_state

QUERY_ID = "S0-QUERY"


class ResolverIndex:
    def __init__(self, ds, matcher: E.Matcher, res: E.Resolution):
        prep = res.prepared
        self.ds, self.matcher, self.res = ds, matcher, res
        self.s1n, self.recn, self.gen, self.comp = prep.s1n, prep.recn, prep.gen, prep.comp
        self.partitions, self.aliases = prep.blocked.partitions, prep.blocked.aliases
        self.feats = prep.feats
        self.pairs = prep.blocked.pairs
        self.s1_best = s1_maxima(self.pairs)
        self.fz = self.feats.select("s1", "fz")
        self.scored = res.scored.select("s1", "rec", "p1", "p")
        self.p1_rows = self.scored.select("s1", "rec", pl.col("p1").alias("p"))
        self.excl = exclusive(self.scored.select("s1", "rec", "p")).select("s1", "rec", pl.col("p").cast(pl.Float64))
        self.matches = res.matches
        self.s1_keys = {k: g for k, g in self._s1_key_tables().items()}
        self.twin_keys = twin_keys(self.recn.select(TWIN_COLS))
        raw = ds.s1.select("entity_id", "business_name", "business_address", "country")
        self.raw_s1 = raw
        self.raw_rec = ds.recs.select("entity_id", "business_name", "business_address", "country", "source")
        self.entities = self._entity_table()
        self.metrics = summary(self.matches, ds.truth, ds.s1["entity_id"]) if ds.truth is not None else None

    @classmethod
    def build(cls, ds, matcher: E.Matcher, workers: int = 1) -> ResolverIndex:
        return cls(ds, matcher, E.resolve(ds, matcher, keep_index=True, workers=workers))

    # ------------------------------------------------------------------ browsing
    def _s1_key_tables(self) -> dict[str, pl.DataFrame]:
        s1k = with_keys(self.s1n)
        return {k: s1k.filter(pl.col(k).is_not_null()).select(pl.col("entity_id").alias("s1"), k)
                      .filter(pl.len().over(k) <= CAP) for k in KEYS}

    def _entity_table(self) -> pl.DataFrame:
        pred = self.matches.group_by("s1").agg(pl.len().alias("n_matched"), pl.col("p").min().alias("min_p"))
        cands = self.pairs.group_by("s1").len("n_candidates")
        t = self.raw_s1.rename({"entity_id": "s1"}).join(pred, on="s1", how="left") \
                       .join(cands, on="s1", how="left") \
                       .with_columns(pl.col("n_matched", "n_candidates").fill_null(0))
        if self.ds.truth is not None:
            ps = per_s1(self.matches, self.ds.truth, self.ds.s1["entity_id"])
            t = t.join(ps.select("s1", "n_true", "tp", "f05"), on="s1", how="left").with_columns(
                pl.when((pl.col("n_true") == 0) & (pl.col("n_matched") == 0)).then(pl.lit("correct_singleton"))
                  .when(pl.col("n_true") == 0).then(pl.lit("false_merge"))
                  .when((pl.col("tp") == pl.col("n_true")) & (pl.col("n_matched") == pl.col("n_true"))).then(pl.lit("exact"))
                  .when(pl.col("tp") < pl.col("n_matched")).then(pl.lit("has_false_match"))
                  .otherwise(pl.lit("missed_match")).alias("outcome"))
        return t

    def list_entities(self, country: str | None = None, q: str | None = None, outcome: str | None = None,
                      offset: int = 0, limit: int = 25) -> dict:
        t = self.entities
        if country:
            t = t.filter(pl.col("country") == country)
        if q:
            ql = q.strip().lower()
            t = t.filter(pl.col("business_name").str.to_lowercase().str.contains(ql, literal=True)
                         | pl.col("s1").str.to_lowercase().str.contains(ql, literal=True)
                         | pl.col("business_address").str.to_lowercase().str.contains(ql, literal=True))
        if outcome and "outcome" in t.columns:
            t = t.filter(pl.col("outcome") == outcome) if outcome != "errors" else \
                t.filter(pl.col("outcome").is_in(["false_merge", "has_false_match", "missed_match"]))
        return {"total": t.height, "items": t.slice(offset, limit).to_dicts()}

    def _record_rows(self, ids: list[str]) -> dict[str, dict]:
        return {r["entity_id"]: r for r in self.raw_rec.filter(pl.col("entity_id").is_in(ids)).to_dicts()}

    def entity(self, s1: str) -> dict | None:
        row = self.entities.filter(pl.col("s1") == s1)
        if row.height == 0:
            return None
        cands = self.feats.filter(pl.col("s1") == s1)
        sc = self.scored.filter(pl.col("s1") == s1).select("rec", "p1", "p")
        chosen = set(self.matches.filter(pl.col("s1") == s1)["rec"].to_list())
        truth = set(self.ds.truth.filter(pl.col("s1") == s1)["rec"].to_list()) if self.ds.truth is not None else None
        # where did each true record that is not in this cluster end up?
        missing = sorted((truth or set()) - chosen)
        owner = {r["rec"]: r for r in self.matches.filter(pl.col("rec").is_in(missing)).to_dicts()}
        expl = self._explain(cands)
        recs = self._record_rows(cands["rec"].to_list() + missing)
        p_of = {r["rec"]: r for r in sc.to_dicts()}
        items = []
        for i, rec in enumerate(cands["rec"].to_list()):
            items.append({"record": recs.get(rec), "rec": rec, "p": _f(p_of.get(rec, {}).get("p")),
                          "p1": _f(p_of.get(rec, {}).get("p1")), "matched": rec in chosen,
                          "truth": (rec in truth) if truth is not None else None, "explanation": expl[i]})
        for rec in missing:
            if rec not in set(cands["rec"].to_list()):
                o = owner.get(rec)
                items.append({"record": recs.get(rec), "rec": rec, "p": None, "p1": None, "matched": False, "truth": True,
                              "explanation": None, "note": "not retrieved by blocking" if not o else f"assigned to {o['s1']}"})
            elif rec in owner:
                for it in items:
                    if it["rec"] == rec:
                        it["note"] = f"assigned to {owner[rec]['s1']} instead"
        items.sort(key=lambda r: (not r["matched"], -(r["p"] or -1)))
        return {"entity": row.to_dicts()[0], "candidates": items}

    def _explain(self, feats: pl.DataFrame) -> list[dict]:
        if feats.height == 0:
            return []
        X = E.to_matrix(feats, E.STAGE1)
        return explain_rows(E.STAGE1, X, self.matcher.contributions(feats))

    # ------------------------------------------------------------------ single-record matching
    def match(self, name: str, address: str, country: str, top: int = 5) -> dict:
        t0 = time.time()
        q = pl.DataFrame({"entity_id": [QUERY_ID], "business_name": [name or ""], "business_address": [address or ""],
                          "country": [country]})
        qn = normalize_frame(q).with_columns(pl.lit(0, pl.Int8).alias("source"))
        norm = {k: qn[k][0] for k in ("name_tok", "addr", "hn", "nonlatin")}
        if country not in self.aliases:
            return {"query": norm, "error": f"no reference entities for country {country!r}", "candidates": []}
        state = infer_state(qn, self.aliases[country])[0]
        pairs, searched = self._query_pairs(qn, country, state)
        out = {"query": {**norm, "country": country, "state": state, "searched": searched}, "candidates": []}
        if pairs.height == 0:
            out["decision"] = {"matched": False, "reason": "no candidates retrieved"}
            out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            return out
        best = self.s1_best.join(pairs.select("s1").unique(), on="s1", how="right").select("s1", "s_best", "s_best_n", "s_best_a")
        upd = pl.concat([best, s1_maxima(pairs)], how="vertical_relaxed").group_by("s1").agg(
            pl.col("s_best", "s_best_n", "s_best_a").fill_null(0.0).max())
        feats = E.featurize(pairs, self.s1n, qn, self.gen, s1_best=upd)
        # S1-side ranks must count the S1's stored candidates too
        other = self.fz.join(feats.select("s1", pl.col("fz").alias("qfz")), on="s1").group_by("s1").agg(
            pl.len().alias("n_other"), (pl.col("fz") > pl.col("qfz")).sum().alias("n_above"))
        feats = feats.join(other, on="s1", how="left").with_columns(
            (pl.col("rev_rank_fz") + pl.col("n_above").fill_null(0)).cast(pl.Float32).alias("rev_rank_fz"),
            (pl.col("n_cand_s1") + pl.col("n_other").fill_null(0)).cast(pl.Float32).alias("n_cand_s1")).drop("n_other", "n_above")
        p1 = self.matcher.p1(feats)
        feats = feats.with_columns(pl.Series("p1", p1, dtype=pl.Float32))
        kept = feats.filter(pl.col("p1") >= E.PRUNE)
        p2 = {}
        if kept.height:
            col = self._collective(kept, qn)
            x2 = kept.drop("p1").join(col, on=["s1", "rec"])
            p2 = dict(zip(x2["s1"].to_list(), self.matcher.p2(x2).tolist()))
        feats = feats.with_columns(pl.col("s1").replace_strict(p2, default=None, return_dtype=pl.Float64).alias("p"))
        ranked = feats.sort(pl.col("p").fill_null(-1.0), pl.col("p1"), descending=True).head(top)
        expl = self._explain(ranked)
        decision = self._decide(ranked)
        s1_raw = {r["entity_id"]: r for r in self.raw_s1.filter(pl.col("entity_id").is_in(ranked["s1"].to_list())).to_dicts()}
        sizes = dict(self.matches.filter(pl.col("s1").is_in(ranked["s1"].to_list())).group_by("s1").len().iter_rows())
        for i, r in enumerate(ranked.select("s1", "p", "p1").to_dicts()):
            out["candidates"].append({"s1": r["s1"], "reference": s1_raw.get(r["s1"]), "p": _f(r["p"]), "p1": _f(r["p1"]),
                                      "cluster_size": sizes.get(r["s1"], 0), "explanation": expl[i],
                                      "selected": decision.get("s1") == r["s1"] and decision["matched"]})
        out["decision"] = decision
        out["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        return out

    def _query_pairs(self, qn: pl.DataFrame, country: str, state: str | None) -> tuple[pl.DataFrame, str]:
        parts = [(k, v) for k, v in self.partitions.items() if k[0] == country]
        if state is not None and (country, state) in self.partitions:
            targets, searched = [((country, state), self.partitions[(country, state)])], f"region {state}"
        else:  # unknown region: search every partition of the country (batch mode would rely on keys only)
            targets, searched = parts, "all regions (region not recognised)"
        tf = []
        for _, (idx, ids) in targets:
            c = idx.query(qn, E.K_NAME, E.K_COMB, E.K_ADDR)
            if c.height:
                tf.append(pl.DataFrame({"s1": ids[c["s"].to_numpy()]}, schema={"s1": pl.String})
                            .with_columns(c.select("cos_name", "cos_addr", "cos_comb")))
        schema = {"s1": pl.String, "rec": pl.String, "cos_name": pl.Float32, "cos_addr": pl.Float32, "cos_comb": pl.Float32}
        tfp = (pl.concat(tf).sort("cos_comb", descending=True).head(E.K_KEEP)
                 .with_columns(pl.lit(QUERY_ID).alias("rec")).select(*schema)) if tf else pl.DataFrame(schema=schema)
        qk = with_keys(qn)
        key_hits = [self.s1_keys[k].join(qk.select(k).drop_nulls(), on=k).select("s1") for k in KEYS]
        kp = pl.concat(key_hits).unique().with_columns(pl.lit(QUERY_ID).alias("rec"))
        kp = kp.join(self.raw_s1.filter(pl.col("country") == country).select(pl.col("entity_id").alias("s1")), on="s1")
        key_only = kp.join(tfp.select("s1", "rec"), on=["s1", "rec"], how="anti") \
                     .with_columns(pl.lit(None, pl.Float32).alias(c) for c in ("cos_name", "cos_addr", "cos_comb"))
        pairs = pl.concat([tfp.with_columns(pl.lit(False).alias("in_keys")),
                           key_only.select(*schema).with_columns(pl.lit(True).alias("in_keys"))], how="vertical_relaxed")
        return pairs, searched

    def _collective(self, kept: pl.DataFrame, qn: pl.DataFrame) -> pl.DataFrame:
        """Collective features of the query's rows against the stored stage-1 scores."""
        qk = twin_keys(qn.select(TWIN_COLS))
        twins = []
        for k in ("k1", "k2"):
            key = qk[k][0]
            if key is not None:
                grp = self.twin_keys.filter(pl.col(k) == key)["entity_id"].to_list()
                if 0 < len(grp) < MAX_GROUP:
                    twins += grp
        twins = sorted(set(twins))
        comp = self.comp
        if twins:
            known = comp.filter(pl.col("entity_id").is_in(twins))
            cid = known["comp"].min() if known.height else QUERY_ID
            members = known.filter(pl.col("comp") == cid)["entity_id"].to_list() if known.height else twins
            comp = pl.concat([comp.filter(~pl.col("entity_id").is_in(members + [QUERY_ID])),
                              pl.DataFrame({"entity_id": members + [QUERY_ID], "comp": cid})])
        cand_s1 = kept["s1"].unique()
        others = self.p1_rows.filter(pl.col("s1").is_in(cand_s1.implode()))["rec"].unique().to_list()
        ctx = self.p1_rows.filter(pl.col("rec").is_in(others + twins))
        rows = pl.concat([ctx, kept.select("s1", "rec", pl.col("p1").cast(pl.Float64).alias("p"))], how="vertical_relaxed")
        return E.collective(rows, comp).filter(pl.col("rec") == QUERY_ID)

    def _decide(self, ranked: pl.DataFrame) -> dict:
        """Exclusive assignment (best S1 only), then expected-F0.5 over that S1's existing cluster
        plus the query: matched iff the decoder keeps the query."""
        top = ranked.filter(pl.col("p").is_not_null()).head(1)
        if top.height == 0:
            return {"matched": False, "reason": "every candidate scored below the stage-2 pruning threshold"}
        s1, p = top["s1"][0], float(top["p"][0])
        cluster = self.excl.filter(pl.col("s1") == s1)
        rows = pl.concat([cluster, pl.DataFrame({"s1": [s1], "rec": [QUERY_ID], "p": [p]})], how="vertical_relaxed")
        kept = ef_decode(rows)
        matched = QUERY_ID in kept["rec"].to_list()
        reason = (f"best candidate kept by expected-F0.5 decoding (p = {p:.2f})" if matched
                  else f"best candidate p = {p:.2f} is not worth the precision risk under F0.5")
        return {"matched": matched, "s1": s1, "p": p, "reason": reason}

    # ------------------------------------------------------------------ examples
    def examples(self, n: int = 8, seed: int = 7) -> list[dict]:
        """Record-like queries drawn from the dataset, covering each country and the hard cases."""
        rng = np.random.default_rng(seed)
        out = []
        recs = self.raw_rec.join(self.recn.select("entity_id", "nonlatin"), on="entity_id")
        picks = []
        for country in sorted(recs["country"].unique().to_list()):
            pool = recs.filter(pl.col("country") == country)
            picks += pool.sample(min(2, pool.height), seed=int(rng.integers(1 << 30))).to_dicts()
        nl = recs.filter(pl.col("nonlatin"))
        if nl.height:
            picks += nl.sample(1, seed=int(rng.integers(1 << 30))).to_dicts()
        for r in picks[:n]:
            out.append({"name": r["business_name"], "address": r["business_address"], "country": r["country"],
                        "tag": "non-Latin script" if r.get("nonlatin") else r["country"]})
        return out


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
