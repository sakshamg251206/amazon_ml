# 13: Our loss decomposition at 0.9614 (E-X1, primary data)

Source: `experiments/e_x1_error_audit.py` → `work/e_x1_error_audit.csv`. Each row is the F0.5 gained if exactly that category were fixed.

| Category | Pairs | F0.5 if fixed |
|---|---|---|
| FN blocking: record has no state | 28,011 | +0.0066 |
| FN model: house numbers differ | 18,653 | +0.0048 |
| FN model: record has empty address | 24,224 | +0.0048 |
| FP on a singleton S1 | 1,899 | +0.0036 |
| FN blocking: same state, not retrieved | 12,521 | +0.0033 |
| FN model: other low-confidence | 11,692 | +0.0028 |
| FP swap (record belongs to another S1) | 3,918 | +0.0027 |
| FN model: non-Latin record | 8,337 | +0.0026 |

- **Totals:** all FPs fixed +0.0132; all model FNs fixed +0.0157; all blocking FNs fixed +0.0111.
- **Decoding** (singleton prior α, sharpening γ, C): no gain over the default (0.9614).
- **House-number misses** are mostly true copies with small shifts (3273→3275), the same recipe as fakes, plus Indian names romanised from native script ("sanraij teknolji" = "sunrise technology").
