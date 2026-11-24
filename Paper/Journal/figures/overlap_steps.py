"""Data for the figure on shared text through the cleaning: at every step of step 2, how many articles have a copy
(Jaccard of word 5-grams >= 0.5) or a partial overlap (>= 0.3) in another outlet, of the same owner or of another one.

Runs where the pipeline data are (the server), from Code/Pipeline:
    cd ~/John/Tentativo20 && python ~/John/paper_figures/overlap_steps.py
It repeats the cleaning of step 2 (cleaning.chain: the same functions and rules). The pairs are computed once, on the
articles of the step "published from 2025" (MinHash, as in step 2), and every later step keeps the pairs whose two
articles are still there. Ownership groups: those of the outlets at "published from 2025". Output:
Dataset/tentativi/20_FromScratch/paper_figures/overlap_steps.csv (outside git).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import cleaning as CL  # noqa: E402
import common as C  # noqa: E402

N_JOBS = 16
OUT = C.OUT / "paper_figures"
OUT.mkdir(exist_ok=True)
ng = CL.read_newsguard(C.NEWSGUARD)

kept, start = {}, None
for name, art in CL.chain(pd.read_csv(C.OUT / "extracted.csv.gz"), ng, N_JOBS):
    if name == "from 2025":
        start = art[["article_id", "domain", "text"]].copy()
    if start is not None:
        kept[name] = set(art.article_id)
    print(name, len(art), flush=True)

start = start.assign(norm=start.text.map(CL.normalize))
pairs = CL.similar_pairs(start.reset_index(drop=True), 0.3, N_JOBS)
pairs = pairs[(pairs.outlet_a != pairs.outlet_b) & (pairs.jaccard >= 0.3)]
groups = CL.ownership_groups(sorted(start.domain.unique()), ng)
pairs["same_owner"] = [groups[a] == groups[b] for a, b in zip(pairs.outlet_a, pairs.outlet_b)]
print(f"pairs across outlets with Jaccard >= 0.3: {len(pairs):,}", flush=True)

rows = []
for name, ids in kept.items():
    p = pairs[pairs.article_a.isin(ids) & pairs.article_b.isin(ids)]
    row = {"step": name, "articles": len(ids)}
    for owner, label in [(True, "same owner"), (False, "other owner")]:
        q = p[p.same_owner == owner]
        for thr, kind in [(0.5, "copy"), (0.3, "overlap")]:
            arts = set(q.article_a[q.jaccard >= thr]) | set(q.article_b[q.jaccard >= thr])
            row[f"{kind}, {label}"] = len(arts) / len(ids)
    rows.append(row)
    print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT / "overlap_steps.csv", index=False)
print("saved", OUT / "overlap_steps.csv")
