# 1 — Data cleaning

Checking the scraped dataset for publishers that share text (owner networks, chains, syndicated articles)
and cleaning it, before any classifier is trained. Starts from the dataset produced by `0-scraping/`
(`Dataset/CleannedDataset/`).

| Notebook | |
|---|---|
| `01_testate_parenti.ipynb` | label-blind: near-duplicate articles across publishers (MinHash, 5-word shingles, Jaccard ≥ 0.5) → groups of related publishers (≥ 2 shared articles); validated against NewsGuard owners |
| `02_pulizia.ipynb` | label-blind cleaning: one copy per near-identical article (Jaccard ≥ 0.9), no copies across groups, paragraphs shared by ≥ 2 publishers removed, articles ≤ 100 words dropped |

`comune.py` holds shared functions. Large intermediate files (near-duplicate pairs, cleaned dataset) go to
`../Dataset/derived/` (not in git).

`esplorazioni/` keeps the first exploratory notebooks (feature classifier on the raw data, error analysis,
effect of groups). They are kept with their outputs for reference and are not meant to be re-run from there.
