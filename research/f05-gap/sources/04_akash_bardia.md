# 04: Akash-bardia README (secondary, competition repo)

- **URL:** https://raw.githubusercontent.com/Akash-bardia/amazon-ml-challenge-2026/main/README.md
- **Type:** practitioner claim.
- **Credibility:** medium-low. Validated on only 5,000 S1, and small samples inflate scores.

## Quotes

- "0.976105 ... held-out stratified set of 5,000 Source-1 records ... full-density blocking"
- "τ_S2 = 0.75, τ_S3 = 0.85" (source-specific thresholds), plus greedy one-to-one assignment.
- Ablation: name-only 0.8897; address-only 0.9191; 28 → 32 features took 0.9741 → 0.9761.
- Features include "Cross-field product (NameSim × AddrSim), sum, max, min" and "Exact core name with missing target address indicator".
