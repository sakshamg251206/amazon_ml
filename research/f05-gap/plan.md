# Plan: closing the F0.5 gap (0.951 OOF -> as high as the data supports)

**Date:** 2026-09-26. **Genre:** decision report (which techniques to implement in the ~1 day left).
**Decision it feeds:** the next 1–3 changes to the pipeline, each kept only if out-of-fold macro F0.5 improves.

## Hypotheses (falsifiable)

- **H1:** The gap to the best public validations (0.97–0.99) comes mainly from *features*, specifically margin / is-best / relative-score features on the blocker's TF-IDF score, not from model families. Refuted if teams attribute their gains to model choice, or if adding these features to M1 does not raise OOF F0.5.
- **H2:** Our confident false positives are mostly "neighbour" distractors (S1 name + extra word, house number shifted; or a different business in the same building) that token-set fuzz cannot see. Refuted if the FP profile does not differ from TPs on these signals.
- **H3:** Name-only retrieval for no-state (empty-address) records adds F0.5. Refuted if a team measured no gain, or our E-B5 shows none.
- **H4:** ≥ 0.98 OOF is reachable in the time left on this hardware. Refuted if the best documented paths needed many iterations, more RAM or a GPU.

## Sourcing strategy

1. Primary: competitor repos and logs for this exact competition (GitHub raw files).
2. Our own data: OOF error profile (primary, reproducible).
3. Academic: collective ER; one-to-one matching; hard negatives.
4. Opposition queries: negative results (no gain, leaderboard regressions).

## Risk register

- vishwajit's repo is gone (404). Its claims survive only in our earlier notes (AMAZON_ML_DEEP_RESEARCH.md) and cannot be re-verified.
- Small-sample validations inflate scores (Akash-bardia: 5k S1).
- Features that encode density (crowding, raw frequency) broke France for SanthoshReddy.

## Stop criteria

Stop when each hypothesis is confirmed, refuted or marked under-determined, and the top items are measured on our own OOF.

## Changelog

- 17:45 plan written; the prior report (AMAZON_ML_DEEP_RESEARCH.md) was reused rather than re-researched.
