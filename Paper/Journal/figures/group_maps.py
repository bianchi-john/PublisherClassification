"""Data for the maps of the ownership groups through the cleaning (Figure 2): every political outlet with extracted text
placed on a 2-D map of its texts, and the cleaning steps it survives.

Runs where the pipeline data are (the server), from Code/Pipeline:
    cd ~/John/Tentativo20 && python ~/John/paper_figures/group_maps.py
It repeats the cleaning of step 2 (cleaning.chain: the same functions and rules) and records which outlets survive each
step. The map is computed once, on the articles of the step "text extracted": the outlet's own name, links and e-mail
addresses are removed (passes 1-2 of step 3), each outlet is the mean of the TF-IDF vectors of its articles, and t-SNE
(on 50 SVD components, cosine distance, seed 42) places the outlets on the plane; outlets with similar texts end up
close. Ownership groups: those of all the political outlets sent to the scraper. Output (outlets numbered, never named):
Dataset/tentativi/20_FromScratch/paper_figures/group_maps.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import anonymization_ner as AX  # noqa: E402
import cleaning as CL  # noqa: E402
import common as C  # noqa: E402

N_JOBS = 16
OUT = C.OUT / "paper_figures"
OUT.mkdir(exist_ok=True)
ng = CL.read_newsguard(C.NEWSGUARD)

steps, start = {}, None
for name, art in CL.chain(pd.read_csv(C.OUT / "extracted.csv.gz"), ng, N_JOBS):
    if name == "text extracted":
        start = art[["domain", "title", "text"]].copy()
    if start is not None:
        steps[name] = set(art.domain)
    print(name, art.domain.nunique(), flush=True)

from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

a = start.assign(title=start.title.fillna(""), text=start.text.fillna("")).reset_index(drop=True)
ruled, _ = AX.rules(a)                                               # links, e-mails, the outlet's own name
X = normalize(TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.5, stop_words="english", max_features=200_000)
              .fit_transform(ruled.text))
domains = sorted(a.domain.unique())
rows = pd.Series(range(len(a)), index=a.domain.to_numpy())
V = np.vstack([np.asarray(X[rows[[d]].to_numpy()].mean(axis=0)) for d in domains])
V = normalize(TruncatedSVD(50, random_state=C.SEED).fit_transform(normalize(V)))
xy = TSNE(2, metric="cosine", perplexity=30, init="pca", random_state=C.SEED).fit_transform(V)

political = CL.political_outlets(CL.read_status(C.SCRAPING), ng)
groups = CL.ownership_groups(political, ng)                          # numbered from the largest group
size = pd.Series(groups).value_counts()
out = pd.DataFrame({"node": range(len(domains)), "x": xy[:, 0], "y": xy[:, 1],
                    "group": [groups[d] for d in domains], "group_size": [size[groups[d]] for d in domains]})
for name, kept in steps.items():
    out[name] = [d in kept for d in domains]
out.to_csv(OUT / "group_maps.csv", index=False)
print("saved", OUT / "group_maps.csv", len(out), "outlets")
