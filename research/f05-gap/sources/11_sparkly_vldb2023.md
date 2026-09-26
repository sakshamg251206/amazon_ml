# 11: Paulsen et al., "Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching", PVLDB 16 (2023) (academic)

- **URL:** https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf (search summary)
- **Quotes:**
  - "kNN-cosine and kNN-jaccard also use top-k, yet underperform Sparkly, which uses tf/idf"
  - Uses BM25 with Lucene inverted indexes.
  - "recall ... ranges from 86.57% to 99.96%"
- **Relevance:** supports a top-k TF-IDF blocker (ours) and suggests BM25 weighting as a possible recall lever. Untested here, because test re-blocking costs about 4.5 h.
