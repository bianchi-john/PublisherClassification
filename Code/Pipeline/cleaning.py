"""Step 2: from the scraped pages to a clean dataset (the rules of attempt 16).

1. political outlets only (NewsGuard 2026: a single topic, "Political news or commentary", and a score);
2. text re-extracted from the saved HTML with trafilatura in precision mode;
3. English, 100-1,500 words, published from 2025;
4. exact and near duplicates removed across all outlets (Jaccard of word 5-grams >= 0.5), keeping the oldest copy;
5. at least 10 articles per outlet;
6. one outlet per ownership group (same NewsGuard Owner or Parent Domain), the one with most articles.
The scraping files keep their original (Italian) field names: dominio, esito, titolo, data, autore, testo.
"""
import gzip
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd

from common import POLITICS, SEED


# ------------------------------------------------------------------------- inputs
def read_newsguard(path):
    """NewsGuard 2026: one row per (lowercase) domain."""
    ng = pd.read_csv(path, low_memory=False)
    return ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")


OUTCOMES = {"completata": "complete", "esauriti gli URL o i tentativi": "out of URLs or attempts",
            "homepage irraggiungibile": "homepage unreachable", "tempo massimo raggiunto": "time limit reached",
            "processo ucciso": "process killed", "processo terminato senza stato": "process died", "errore": "error"}


def read_status(folders):
    """One row per outlet the scraper tried (the files in stato/), with the outcome translated into English."""
    status = pd.DataFrame([json.load(open(f)) for c in folders for f in Path(c, "stato").glob("*.json")])
    for it, en in OUTCOMES.items():
        status["esito"] = status.esito.str.replace(it, en, regex=False)
    return status


def political_outlets(status, ng):
    """Scraped outlets whose only NewsGuard topic is politics and that have a score."""
    topics = status.dominio.map(ng.Topics).fillna("").str.split(",").map(lambda x: [a.strip() for a in x if a.strip()])
    political = topics.map(lambda x: x == [POLITICS]) & status.dominio.map(ng.Score).notna()
    return sorted(status.dominio[political.to_numpy()])


# ------------------------------------------------------------------------- text
def extract_outlet(domain, folders):
    """Re-extract the text of every article of an outlet from its saved HTML with trafilatura in precision mode
    (no comments, tables, images, links or formatting). Title, date and author come from the scraping."""
    import trafilatura
    rows = []
    for folder in folders:
        fa, fh = Path(folder, "articoli", f"{domain}.jsonl"), Path(folder, "html", f"{domain}.jsonl.gz")
        if not fa.exists():
            continue
        html = {}
        if fh.exists():
            for line in gzip.open(fh, "rt"):
                d = json.loads(line)
                if d.get("tipo") == "articolo" and d.get("html"):
                    html[d["url"]] = d["html"]
        for line in open(fa):
            a = json.loads(line)
            h, text = html.get(a["url"]), None
            if h:
                try:
                    text = trafilatura.extract(h, url=a["url"], favor_precision=True, include_comments=False,
                                               include_tables=False, include_images=False, include_links=False,
                                               include_formatting=False)
                except Exception:
                    text = None
            rows.append({"domain": domain, "url": a["url"], "title": a.get("titolo"), "date": a.get("data"),
                         "author": a.get("autore"), "scraped_text": a.get("testo"), "text": text, "has_html": h is not None})
    return rows


def language(text):
    """Language of the first 400 words (langdetect, fixed seed)."""
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    try:
        return detect(" ".join(text.split()[:400]))
    except LangDetectException:
        return "?"


# ------------------------------------------------------------------------- duplicates
def normalize(text):
    """Lowercase, letters and digits only, single spaces: to compare texts regardless of their form."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def _minhash(text):
    from datasketch import MinHash
    w = text.split()
    m = MinHash(num_perm=128, seed=SEED)
    m.update_batch([" ".join(w[i:i + 5]).encode() for i in range(max(1, len(w) - 4))])
    return m


def similar_pairs(df, threshold, n_jobs=32):
    """Pairs of articles whose Jaccard similarity of word 5-grams (MinHash estimate) is at least `threshold`."""
    from datasketch import MinHashLSH
    from joblib import Parallel, delayed
    hashes = Parallel(n_jobs=n_jobs, batch_size=500)(delayed(_minhash)(t) for t in df.norm)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, m in enumerate(hashes):
        lsh.insert(i, m)
    c = pd.DataFrame([(i, j, m.jaccard(hashes[j])) for i, m in enumerate(hashes) for j in lsh.query(m) if j > i],
                     columns=["i", "j", "jaccard"])
    for side, col in [("a", "i"), ("b", "j")]:
        c[f"article_{side}"] = df.article_id.values[c[col]]
        c[f"outlet_{side}"] = df.domain.values[c[col]]
    return c.drop(columns=["i", "j"])


def copies_to_drop(pairs, dates):
    """Within every group of linked copies, all but the oldest (ties: the lowest article_id)."""
    G = nx.Graph(); G.add_edges_from(zip(pairs.article_a, pairs.article_b))
    return {x for group in nx.connected_components(G) for x in sorted(group, key=lambda x: (dates[x], x))[1:]}


# ------------------------------------------------------------------------- ownership groups
def normalize_owner(owner):
    """Owner name without legal forms and generic words (Inc, LLC, Media, …), to match the same owner."""
    o = re.sub(r"[^a-z0-9 ]", " ", str(owner).lower())
    o = re.sub(r"\b(inc|llc|ltd|limited|gmbh|co|kg|corp|corporation|company|the|group|media|plc|sa|ag|"
               r"holdings?|news|publishing)\b", " ", o)
    o = re.sub(r"\s+", " ", o).strip()
    return "" if o in ("", "nan", "unknown") else o


def ownership_groups(domains, ng):
    """Outlets are related if they share the NewsGuard Owner (normalized) or Parent Domain, also indirectly.
    Returns {domain: group number}, the largest group first."""
    G = nx.Graph(); G.add_nodes_from(domains)
    owner = pd.Series({d: normalize_owner(ng.Owner.get(d, "")) for d in domains})
    parent = pd.Series({d: str(ng["Parent Domain"].get(d, "")).strip().lower() for d in domains}).replace("nan", "")
    for values in [owner, parent]:
        v = values[values != ""]
        for _, same in v.groupby(v):
            G.add_edges_from(zip(same.index, same.index[1:]))
    return {d: i for i, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)) for d in c}


# ------------------------------------------------------------------------- the chain again, one step at a time
# the rules of step 2 (02_cleaning.ipynb sets the same values in its first cell)
NEAR_DUPLICATE = 0.5          # Jaccard of word 5-grams
CANDIDATES = 0.3              # MinHash looks for candidate pairs from this similarity (approximate), then >= NEAR_DUPLICATE
MIN_WORDS, MAX_WORDS = 100, 1500
MIN_ARTICLES = 10
FROM_DATE = "2025-01-01"
STEPS = ["sent to the scraper", "at least one article", "text extracted", "English", "100-1,500 words", "from 2025",
         "no exact duplicates", "no near duplicates", ">= 10 articles", "one outlet per group"]


def chain(raw, ng, n_jobs=16, info=None):
    """The cleaning of step 2 again, with its functions and rules, one step at a time: yields (step, articles after it)
    for STEPS[1:], from `raw` (the cache of the extracted texts) to one outlet per group; article_id is given at "from
    2025", as in step 2. `info` (a dict) receives, along the way, "exact" (the outlets of every exact text, before the
    deduplication) and "pairs" (the near-duplicate pairs). Used by step 14 and by the figures on the dataset; step 2
    keeps its own code (changing it would mean running it again and rewriting the dataset). The yielded tables are
    the ones the chain goes on with: copy them before changing them."""
    import hashlib

    from joblib import Parallel, delayed
    info = {} if info is None else info
    yield STEPS[1], raw
    art = raw[raw.text.notna()].reset_index(drop=True)
    yield STEPS[2], art
    art = art.assign(language=Parallel(n_jobs=n_jobs, batch_size=200)(delayed(language)(t) for t in art.text))
    art = art[art.language == "en"]
    yield STEPS[3], art
    art = art.assign(words=art.text.str.split().str.len())
    art = art[art.words.between(MIN_WORDS, MAX_WORDS)]
    yield STEPS[4], art
    art = art.assign(date=pd.to_datetime(art.date, errors="coerce"))
    art = art[art.date >= FROM_DATE].reset_index(drop=True).rename_axis("article_id").reset_index()
    yield STEPS[5], art
    art = art.assign(norm=art.text.map(normalize))
    art["fingerprint"] = art.norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
    info["exact"] = art.groupby("fingerprint").domain.apply(lambda s: sorted(set(s)))
    art = art.sort_values(["date", "article_id"]).drop_duplicates("fingerprint").sort_values("article_id")
    yield STEPS[6], art
    pairs = similar_pairs(art.reset_index(drop=True), CANDIDATES, n_jobs)
    info["pairs"] = pairs = pairs[pairs.jaccard >= NEAR_DUPLICATE]
    art = art[~art.article_id.isin(copies_to_drop(pairs, art.set_index("article_id").date))]
    yield STEPS[7], art
    n = art.groupby("domain").size()
    art = art[art.domain.isin(n.index[n >= MIN_ARTICLES])]
    yield STEPS[8], art
    group = art.domain.map(ownership_groups(sorted(art.domain.unique()), ng))
    kept = (art.assign(group=group).groupby("domain").agg(group=("group", "first"), n=("article_id", "size"))
            .reset_index().sort_values(["n", "domain"], ascending=[False, True]).groupby("group").head(1).domain)
    yield STEPS[9], art[art.domain.isin(kept)]
