"""What do the sites of each content network share?

For every network of at least three publishers (explain/publisher_networks.csv):
the share of articles whose opening is shared with another site of the
network, the two most widely shared openings, and the terms whose document
frequency inside the network most exceeds that in the rest of the corpus.
Output: Results/derived/explain/network_content.csv
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import config as C
import run_tfidf_lr_baseline as B

net = pd.read_csv(C.DERIVED_DIR / "explain" / "publisher_networks.csv", index_col=0)
e = pd.read_csv(B.ENRICHED, usecols=["domain", "text", "score"])
e["comp"] = e.domain.map(net.comp)
e["size"] = e.domain.map(net.comp_size)
norm = lambda s: re.sub(r"\d+", "#", re.sub(r"\W+", " ", str(s).lower())).strip()
e["first"] = e.text.astype(str).str.split().str[:12].str.join(" ")
e["tmpl"] = e["first"].map(norm)
cv = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=10, max_df=0.5, stop_words="english")
X = cv.fit_transform(e.text.astype(str))
terms = np.array(cv.get_feature_names_out())
rows = []
for c, g in e[e["size"] >= 3].groupby("comp"):
    idx = g.index.to_numpy()
    rest = np.setdiff1d(np.arange(len(e)), idx)
    pin = np.asarray(X[idx].mean(0)).ravel()
    pout = np.asarray(X[rest].mean(0)).ravel()
    top = np.argsort(pin - pout)[::-1][:8]
    nsites = g.groupby("tmpl").domain.nunique()
    best = nsites.sort_values(ascending=False).head(2)
    rows.append({
        "n_sites": g.domain.nunique(),
        "labels": net[net.comp == c].label.map(lambda k: C.CLASS_LABELS[k]).value_counts().to_dict(),
        "scores": sorted(g.groupby("domain").score.first().unique().tolist()),
        "examples": sorted(g.domain.unique())[:3],
        "share_shared_opening": round(float((g.tmpl.map(nsites) >= 2).mean()), 3),
        "top_openings": [(g[g.tmpl == t]["first"].iloc[0], int(n)) for t, n in best.items()],
        "distinctive_terms": [(terms[j], round(float(pin[j]), 3), round(float(pout[j]), 3)) for j in top],
    })
out = pd.DataFrame(rows).sort_values("n_sites", ascending=False)
out.to_csv(C.DERIVED_DIR / "explain" / "network_content.csv", index=False)
print(out[["n_sites", "labels", "share_shared_opening"]].to_string(index=False))
