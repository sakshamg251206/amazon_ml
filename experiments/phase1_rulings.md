Setup: Ruling: branch `phase-1-rule-baseline` in place instead of a worktree — dataset is gitignored and would be absent from a worktree — cost if wrong: none (branch is equally isolated from main)
Setup: Ruling: this run executes Tasks 0–5 (Phases 0–1) only — Phases 2–6 are roadmap-only in the plan by design — cost if wrong: none
Task 0: Ruling: polars 1.44 renamed missing_utf8_is_empty_string -> empty_string_is_null=False (same semantics, verified 168,967 empty S2 addresses as '' and 0 nulls) — cost if wrong: none, test guards it
Final: Ruling: France exact-key w/o city/postcode — match rate 79.5% in line with US 77%; precision unmeasurable without labels; stands for Sub 1 — cost if wrong: some France FPs on Sub 1 only; revisited in Phase 5 France work
Final: Ruling: experiments/log.csv stays versioned — spec §4.2 wants a run log travelling with code — cost if wrong: a noisy diff
Final: Ruling: LEGAL_FORMS/STOPWORDS lists (incl. sa, co) — cost already reflected in measured train F0.5 0.595 — cost if wrong: small recall/precision shift, re-tuned in Phase 3 normaliser v2
Final: minor (deferred): house_no takes first digit run; breaks under component reordering (Unit 11, 1064 Newton Rd) — Phase 3 nums set/number-relation features
Final: minor (deferred): name_key set() collapses repeated tokens (Bala Bala Traders)
Final: minor (deferred): no tests for non-ASCII digits in house_no or explicit record-exclusivity
Final: minor (deferred): requirements.txt lacks Python version note (3.12) — add in Phase 6 package README
Final: minor (deferred): read_ground_truth does not strip/filter empty IDs from trailing commas (not present in current data)
