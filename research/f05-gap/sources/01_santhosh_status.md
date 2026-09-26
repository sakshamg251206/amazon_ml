# 01: SanthoshReddy352 STATUS.md (primary, competition repo)

- **URL:** https://raw.githubusercontent.com/SanthoshReddy352/Amazon-ML-Challenge-2026/main/STATUS.md (fetched 2026-09-26)
- **Type:** primary / practitioner log.
- **Scores:** Credibility high (versioned, validation *and* leaderboard). Recency: this week. Bias: low (reports its own failures).

## Quotes (verbatim from the fetch)

- Version table:
  - v1 0.9706 val / **0.962** LB: "LightGBM with 200k fit S1, threshold 0.70".
  - v2 0.9710 / 0.951: "Added crowding features (France regression)".
  - v3 0.9748 / **0.957**.
  - v4c 0.9781: "Stage-3 refiner: house v2, token miss/extra, log2-bucketed frequency".
  - v5c **0.9821**: "Indian address codes, adjacent address-word pairs".
- Feature importance (gain, v1): "rev_margin 0.66, rev_is_best 0.10, a_nums_jacc 0.065, rev_score_rel 0.02, a_street_tset 0.02, a_tok_jacc 0.02"
- Stage 2: "Compare each candidate with the S1's 2 best other candidates (p1, name/skeleton/address/house agreement)." Full validation went from 0.9704 to 0.9732 (+0.0028).
- "Name-only extension for unclaimed empty-address records (v6): no gain vs. v5b"
- "False-positive recipe across countries: S1 name + extra word / swapped word / legal form change, house number shifted +1..+21 on the same street"
- "Expected-F0.5 prefix selection: best 0.9705 < threshold 0.9710"
- "Removed acronym + per-split token frequency (v4b): tree thresholds flipped, dropped 46k India test pairs"
- "Blocking/features run over ALL train S1 (label-free); the split only selects whose labels fit vs score"
- France: "house-number disagreement in only 1.5% of true pairs, vs. 9.8% US and 19.5% India"
