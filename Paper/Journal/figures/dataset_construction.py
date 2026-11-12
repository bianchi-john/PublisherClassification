"""Data for the figures on how the dataset is built (ownership groups, duplicates, anonymization).

Runs where the pipeline data are (the server), from Code/Pipeline:
    cd ~/John/Tentativo20 && python ~/John/paper_figures/dataset_construction.py
It repeats the cleaning of step 2 with the same functions and constants (no new rule) and records, along the way,
what the figures need. Output in Dataset/tentativi/20_FromScratch/paper_figures/ (outside git: the ownership groups
come from NewsGuard; outlets are numbered, never named):
  steps.csv          outlets, ownership groups, sister outlets and articles after every cleaning step
  nodes.csv          the outlets of the step "published from 2025": group, group size, kept in the dataset, layout x/y
  edges.csv          pairs of those outlets that share articles (exact or near duplicates), with how many
  similarity.csv     the outlets with >= 10 articles after deduplication: text similarity to the nearest outlet of their
                     own ownership group and of another group (TF-IDF centroids, links, e-mails and own name removed)
  anonymization.csv  per article of the final dataset: level, words, placeholders by kind and pass, own name left
  summary.json       counts quoted in the captions
"""
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, ".")
import anonymization_ner as AX  # noqa: E402
import cleaning as CL  # noqa: E402
import common as C  # noqa: E402

N_JOBS = 16
NEAR_DUPLICATE, CANDIDATES = 0.5, 0.3
MIN_WORDS, MAX_WORDS, MIN_ARTICLES, FROM_DATE = 100, 1500, 10, "2025-01-01"
OUT = C.OUT / "paper_figures"
OUT.mkdir(exist_ok=True)
ng = CL.read_newsguard(C.NEWSGUARD)
steps, summary = [], {}


def step(name, df):
    domains = sorted(df.domain.unique())
    groups = CL.ownership_groups(domains, ng)
    size = Counter(groups.values())
    steps.append({"step": name, "articles": len(df), "outlets": len(domains), "groups": len(size),
                  "sister_outlets": len(domains) - len(size),
                  "outlets_in_shared_groups": sum(n for n in size.values() if n > 1)})
    print(steps[-1], flush=True)


# ---------------------------------------------------------------------------------------------- step 2, as in 02
status = CL.read_status(C.SCRAPING)
political = CL.political_outlets(status, ng)
steps.append({"step": "sent to the scraper", "articles": None, "outlets": len(political),
              "groups": len(set(CL.ownership_groups(political, ng).values())), "sister_outlets": None,
              "outlets_in_shared_groups": None})
steps[-1]["sister_outlets"] = steps[-1]["outlets"] - steps[-1]["groups"]
raw = pd.read_csv(C.OUT / "extracted.csv.gz")
step("at least one article", raw)
art = raw[raw.text.notna()].reset_index(drop=True)
step("text extracted", art)
art["language"] = Parallel(n_jobs=N_JOBS, batch_size=200)(delayed(CL.language)(t) for t in art.text)
art = art[art.language == "en"]
step("English", art)
art["words"] = art.text.str.split().str.len()
art = art[art.words.between(MIN_WORDS, MAX_WORDS)]
step("100-1,500 words", art)
art["date"] = pd.to_datetime(art.date, errors="coerce")
art = art[art.date >= FROM_DATE].reset_index(drop=True).rename_axis("article_id").reset_index()
step("from 2025", art)
stage = art.copy()                                                   # the outlets of the network figure

art["norm"] = art.text.map(CL.normalize)
art["fingerprint"] = art.norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
exact = art.groupby("fingerprint").domain.apply(lambda s: sorted(set(s)))
art = art.sort_values(["date", "article_id"]).drop_duplicates("fingerprint").sort_values("article_id")
step("no exact duplicates", art)
pairs = CL.similar_pairs(art.reset_index(drop=True), CANDIDATES, N_JOBS)
pairs = pairs[pairs.jaccard >= NEAR_DUPLICATE]
art = art[~art.article_id.isin(CL.copies_to_drop(pairs, art.set_index("article_id").date))]
step("no near duplicates", art)
n = art.groupby("domain").size()
art = art[art.domain.isin(n.index[n >= MIN_ARTICLES])]
step(">= 10 articles", art)
at10 = art.copy()
art["group"] = art.domain.map(CL.ownership_groups(sorted(art.domain.unique()), ng))
kept = (art.groupby("domain").agg(group=("group", "first"), n=("article_id", "size")).reset_index()
        .sort_values(["n", "domain"], ascending=[False, True]).groupby("group").head(1).domain)
art = art[art.domain.isin(kept)]
step("one outlet per group", art)
final = pd.read_csv(C.OUT / "outlets.csv", index_col="domain")
assert set(art.domain) == set(final.index) and len(art) == final.articles.sum(), "not the paper's dataset"
pd.DataFrame(steps).to_csv(OUT / "steps.csv", index=False)

# ---------------------------------------------------------------------------------------------- network of shared text
domains = sorted(stage.domain.unique())
groups = CL.ownership_groups(domains, ng)
gsize = Counter(groups.values())
idx = {d: i for i, d in enumerate(domains)}
links = Counter()
for outs in exact[exact.map(len) > 1]:                               # the same text in several outlets
    for a, b in combinations(outs, 2):
        links[(idx[a], idx[b])] += 1
for a, b in zip(pairs.outlet_a, pairs.outlet_b):                     # near duplicates across outlets
    if a != b:
        links[tuple(sorted((idx[a], idx[b])))] += 1
edges = pd.DataFrame([(a, b, w) for (a, b), w in links.items()], columns=["a", "b", "shared"])
G = nx.Graph(); G.add_nodes_from(range(len(domains)))
G.add_weighted_edges_from(edges.itertuples(index=False, name=None))
connected = [v for v in G if G.degree(v) > 0]
pos = nx.spring_layout(G.subgraph(connected), seed=C.SEED, k=0.12, iterations=300)
nodes = pd.DataFrame({"node": range(len(domains)), "group": [groups[d] for d in domains],
                      "group_size": [gsize[groups[d]] for d in domains], "kept": [d in final.index for d in domains],
                      "articles": stage.groupby("domain").size().reindex(domains).to_numpy(),
                      "x": [pos[i][0] if i in pos else np.nan for i in range(len(domains))],
                      "y": [pos[i][1] if i in pos else np.nan for i in range(len(domains))]})
nodes.to_csv(OUT / "nodes.csv", index=False)
edges.to_csv(OUT / "edges.csv", index=False)
same = np.array([groups[domains[a]] == groups[domains[b]] for a, b in zip(edges.a, edges.b)])
level = {d: C.LEVELS[int(lv)] for d, lv in zip(domains, C.level_from_score(pd.Series(domains).map(ng.Score)))}
largest = [d for d in domains if groups[d] == 0]
summary.update({
    "network_outlets": len(domains), "network_connected": len(connected), "network_edges": len(edges),
    "network_edges_same_group": int(same.sum()), "network_shared_same_group": int(edges.shared[same].sum()),
    "network_shared_total": int(edges.shared.sum()),
    "largest_group_size": len(largest), "largest_group_levels": dict(Counter(level[d] for d in largest)),
    "near_duplicate_pairs": len(pairs), "near_duplicate_pairs_across_outlets": int((pairs.outlet_a != pairs.outlet_b).sum()),
})

# ---------------------------------------------------------------------------------------------- text similarity of sisters
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

sim_art = at10.assign(title="", text=at10.text.fillna(""))[["article_id", "domain", "title", "text"]]
ruled, _ = AX.rules(sim_art)                                         # links, e-mails, the outlet's own name
X = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.5, stop_words="english", max_features=200_000) \
    .fit_transform(ruled.text)
X = normalize(X)
d10 = sorted(at10.domain.unique())
rows = pd.Series(range(len(ruled)), index=ruled.domain.to_numpy())
centroids = normalize(np.vstack([np.asarray(X[rows[[d]].to_numpy()].mean(axis=0)) for d in d10]))
S = centroids @ centroids.T
np.fill_diagonal(S, -1)
g10 = CL.ownership_groups(d10, ng)
gid = np.array([g10[d] for d in d10])
sim = []
for i, d in enumerate(d10):
    sister = gid == gid[i]; sister[i] = False
    sim.append({"node": i, "has_sister": bool(sister.any()), "kept": d in final.index,
                "nearest_sister": float(S[i, sister].max()) if sister.any() else np.nan,
                "nearest_other": float(S[i, ~sister & (np.arange(len(d10)) != i)].max())})
pd.DataFrame(sim).to_csv(OUT / "similarity.csv", index=False)

# ---------------------------------------------------------------------------------------------- anonymization
a = pd.read_csv(C.OUT / "articles.csv.gz")
a["title"] = a.title.fillna(""); a["text"] = a.text.fillna("")
ruled, counts = AX.rules(a)
spans = AX.run(ruled)
anon = AX.apply(ruled, spans)
k = spans[spans.kept & (spans.field == "text")].groupby(["article_id", "label"]).size().unstack(fill_value=0)
count = lambda s, tok: s.str.count(tok.replace("[", r"\[").replace("]", r"\]"))
before = lambda tok: count(a.text, tok)                              # brackets already in the original text
per = pd.DataFrame({"article_id": a.article_id, "level": a.level, "words": a.text.str.split().str.len(),
                    "url": count(ruled.text, "[URL]") - before("[URL]"),
                    "email": count(ruled.text, "[EMAIL]") - before("[EMAIL]"),
                    "own_name": count(ruled.text, "[OUTLET]") - before("[OUTLET]")})
per["gliner_outlet"] = per.article_id.map(k.get("news outlet", pd.Series(dtype=int))).fillna(0).astype(int)
per["gliner_journalist"] = per.article_id.map(k.get("journalist", pd.Series(dtype=int))).fillna(0).astype(int)
def own_left(df):
    """Articles whose title or text still contains the name of their own outlet (the paper's check), by article_id."""
    return AX.own_name_left(df).set_index("article_id")["own name left"]


for name, df in [("raw", a), ("rules", ruled), ("final", anon)]:
    per[f"own_name_left_{name}"] = per.article_id.map(own_left(df)).astype(bool)
per.to_csv(OUT / "anonymization.csv", index=False)
summary.update({"rule_counts_title_and_text": counts, "spans_kept": spans.groupby("label").kept.sum().to_dict(),
                "spans_found": spans.groupby("label").size().to_dict()})
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
print(json.dumps(summary, indent=2, default=int))
print("done", OUT)
