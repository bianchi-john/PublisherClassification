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
