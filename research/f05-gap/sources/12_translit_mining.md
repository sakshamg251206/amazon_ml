# 12: Transliteration mining from parallel data (academic)

- Ekbal/Bandyopadhyay-style Indic transliteration from parallel corpora: https://www.researchgate.net/publication/255968530
- Transliteration equivalence (CCA): https://link.springer.com/chapter/10.1007/978-3-642-12275-0_10
- **Quote (search summary):** "the parallel corpus is word-aligned using GIZA++ ... extracting 1-to-1 alignment word pairs as potential transliteration equivalents".
- **Our use:** our true training pairs (non-Latin record ↔ Latin S1) are a free parallel corpus. We use positional 1-to-1 alignment with count ≥ 3 and share ≥ 0.6. That gave 362 mappings (validation-safe) and 515 (all of train); see `src/translit.py`.
- SanthoshReddy's `dictionaries.py` is hand-written only: "Hand-written normalisation dictionaries (general language knowledge, not entity data)".
