"""Tentativo 14: le funzioni del notebook chunk.ipynb.

Il notebook resta corto e leggibile; qui stanno le funzioni lunghe. Sono quelle dei tentativi precedenti, con le stesse
regole (costruzione del dataset: 0-scraping/costruisci_dataset; pulizia e reti: 09/1-data-cleaning/01-02; dataset
solo politici: 12/2-dataset e 13/1-dataset; classificatore a feature e ModernBERT: 11-13), riscritte come funzioni.
La novità è ModernBERT a pezzi (chunk): un articolo si divide in pezzi da 512 token, ogni pezzo si valuta da solo, poi
i pezzi si mettono insieme in vari modi (`aggrega`).
"""
import collections
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
CLASSI = ["p_l1", "p_l2", "p_l3", "p_l4", "p_l5"]
CONFINI = [-1, 39, 59, 74, 99, 100]          # dal punteggio NewsGuard ai livelli l1..l5
POLITICA = "Political news or commentary"


def cartella_dati():
    """La cartella unica dei dati: Dataset/ nella radice del repo (sul server: ~/John/Dataset, accanto ai tentativi).
    Si cerca salendo dalla cartella di lavoro fino a una cartella Dataset/ con dentro LEGGIMI.md."""
    for p in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (p / "Dataset" / "LEGGIMI.md").exists():
            return p / "Dataset"
    raise FileNotFoundError("non trovo la cartella Dataset/ (con LEGGIMI.md) salendo dalla cartella di lavoro")


DATI = cartella_dati()
DATASET = DATI / "tentativi" / "14_DatasetRipensatoChunk"          # il dataset di questo tentativo (articoli.csv.gz, testate.csv, cache)


# ===================================================================================== 1. dati grezzi
def leggi_scraping(cartelle, testate=None):
    """Tutti i testi estratti dallo scraping (un file articoli/<dominio>.jsonl per testata, in una o più cartelle),
    in ordine alfabetico di file come nella costruzione del dataset dei tentativi precedenti. Con `testate`, solo quelle."""
    file = sorted((f for c in cartelle for f in Path(c, "articoli").glob("*.jsonl")), key=lambda f: f.name)
    if testate is not None:
        file = [f for f in file if f.stem in set(testate)]
    righe = [json.loads(r) for f in file for r in open(f)]
    return pd.DataFrame(righe)


def leggi_stati(cartelle):
    return pd.DataFrame([json.load(open(f)) for c in cartelle for f in Path(c, "stato").glob("*.json")])


def lingua(testo):
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    try:
        return detect(" ".join(testo.split()[:400]))          # le prime 400 parole bastano
    except LangDetectException:
        return "?"


def leggi_newsguard(percorso):
    """Una riga per dominio (minuscolo)."""
    ng = pd.read_csv(percorso, low_memory=False)
    return ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")


def livello(score):
    return pd.cut(pd.Series(score), CONFINI, labels=False).to_numpy()


# ===================================================================================== 2. pulizia
def normalizza(testo):
    """Minuscole, solo lettere e cifre, spazi singoli: per confrontare testi senza badare alla forma."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(testo).lower())).strip()


def frasi(riga):
    return [f for f in re.split(r"(?<=[.!?])\s+", riga.strip()) if f]


def normalizza_owner(owner):
    o = re.sub(r"[^a-z0-9 ]", " ", str(owner).lower())
    o = re.sub(r"\b(inc|llc|ltd|limited|gmbh|co|kg|corp|corporation|company|the|group|media|plc|sa|ag|"
               r"holdings?|news|publishing)\b", " ", o)
    o = re.sub(r"\s+", " ", o).strip()
    return "" if o in ("", "nan", "unknown") else o


def _firma(testo):
    from datasketch import MinHash
    w = testo.split()
    m = MinHash(num_perm=128, seed=SEED)
    m.update_batch([" ".join(w[i:i + 5]).encode() for i in range(max(1, len(w) - 4))])
    return m


def coppie_simili(df, soglia, n_jobs=32):
    """Coppie di articoli con Jaccard dei 5-grammi di parole (stima MinHash) almeno `soglia`."""
    from datasketch import MinHashLSH
    from joblib import Parallel, delayed
    firme = Parallel(n_jobs=n_jobs, batch_size=500)(delayed(_firma)(t) for t in df.norm)
    lsh = MinHashLSH(threshold=soglia, num_perm=128)
    for i, m in enumerate(firme):
        lsh.insert(i, m)
    c = pd.DataFrame([(i, j, m.jaccard(firme[j])) for i, m in enumerate(firme) for j in lsh.query(m) if j > i],
                     columns=["i", "j", "jaccard"])
    for lato, col in [("a", "i"), ("b", "j")]:
        c[f"articolo_{lato}"] = df.article_id.values[c[col]]
        c[f"testata_{lato}"] = df.domain.values[c[col]]
    return c.drop(columns=["i", "j"])


def togli_frasi_ripetute(art, tutti, min_articoli=3):
    """Toglie le frasi che la stessa testata ripete in almeno `min_articoli` articoli (firme, inviti, piè di pagina).
    Le ripetizioni si contano su tutti gli articoli della testata (`tutti`). Ritorna i testi puliti e la quota tolta."""
    in_testata = collections.Counter()
    for d, t in zip(tutti.domain, tutti.text):
        in_testata.update((d, f) for f in {normalizza(f) for riga in t.split("\n") for f in frasi(riga)} - {""})
    puliti = []
    for d, testo in zip(art.domain, art.text):
        righe = []
        for riga in testo.split("\n"):
            tenute = [f for f in frasi(riga) if in_testata[(d, normalizza(f))] < min_articoli]
            if tenute:
                righe.append(" ".join(tenute))
        puliti.append("\n".join(righe))
    esempi = {}                                   # per gli esempi: la frase (di almeno 6 parole) più ripetuta di ogni testata
    for n, d, f in sorted(((n, d, f) for (d, f), n in in_testata.items() if n >= min_articoli and len(f.split()) >= 6), reverse=True):
        esempi.setdefault(d, (n, d, f))
    return puliti, list(esempi.values())[:10]


MESI = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
ORA = r"\d{1,2}:\d\d\s*(?:[ap]\.?m\.?)?(?:\s*[A-Z]{2,4}T)?"
TIMBRO = re.compile(rf"^\s*(?:(?:Published|Updated|Posted|Last updated)\s*(?:on|at)?\s*)?(?:{ORA}\s*)?(?:\w+day,?\s*)?"
                    rf"{MESI}\s+\d{{1,2}},?\s+20\d\d(?:\s*(?:at|,)?\s*{ORA})?[\s|·•,-]*", re.I)


def togli_cornice(testo, titolo):
    """Toglie dall'inizio del testo il titolo ripetuto e il timbro di pubblicazione (dipendono dal sito)."""
    fine_parole = [m.end() for m in re.finditer(r"\S+", testo)]
    taglio, tolto = 0, False
    candidati = {normalizza(titolo), normalizza(re.split(r"\s[-–—|]\s", str(titolo))[0])}
    for cand in sorted((c for c in candidati if len(c.split()) >= 3), key=len, reverse=True):
        n = len(cand.split())
        for k in range(max(1, n - 4), min(n + 6, len(fine_parole)) + 1):
            if normalizza(testo[:fine_parole[k - 1]]) == cand:
                taglio, tolto = fine_parole[k - 1], True; break
        if tolto:
            break
    resto = testo[taglio:].lstrip()
    senza_timbro = TIMBRO.sub("", resto, count=1)
    return senza_timbro, tolto, senza_timbro != resto


LINK = re.compile(r"(?:https?://|www\.)\S+|pic\.twitter\.com/\S+", re.I)
DATA_TWEET = rf"{MESI}\s+\d{{1,2}},?\s+\d{{4}}"
FIRMA_IN_FONDO = re.compile(rf"(?:[—–]{{1,2}}|\s-{{1,2}})\s*[^—–\n]{{0,80}}\(@\w{{1,30}}\)\s*(?:{DATA_TWEET})?\s*$", re.I)
FIRMA_SOLA = re.compile(rf"^\s*[^\n]{{0,80}}\(@\w{{1,30}}\)\s*{DATA_TWEET}\s*$", re.I)
SCRITTE_INCORPORATI = re.compile(r"^\s*(?:View this post on Instagram|A post shared by [^\n]{0,80})\s*$", re.I)


def togli_incorporati(testo, min_parole_riga=3):
    """Toglie link, firme di tweet incorporati e le scritte dei post incorporati; le righe rimaste quasi vuote spariscono."""
    righe = []
    for riga in testo.split("\n"):
        if FIRMA_SOLA.match(riga) or SCRITTE_INCORPORATI.match(riga):
            continue
        nuova, f = FIRMA_IN_FONDO.subn("", riga)
        nuova, k = LINK.subn("", nuova)
        if (f or k) and len(re.findall(r"[^\W\d_]{2,}", nuova)) < min_parole_riga:
            continue
        righe.append(re.sub(r"[ \t]{2,}", " ", nuova).strip() if (f or k) else riga)
    return "\n".join(righe)


def togli_doppioni_in_testata(art, soglia=0.5, n_jobs=32):
    """Tra articoli quasi uguali della stessa testata si tiene il più vecchio."""
    dentro = coppie_simili(art, soglia, n_jobs)
    dentro = dentro[(dentro.testata_a == dentro.testata_b) & (dentro.jaccard >= soglia)]
    G = nx.Graph(); G.add_edges_from(zip(dentro.articolo_a, dentro.articolo_b))
    data = art.set_index("article_id").data
    via = {x for fam in nx.connected_components(G) for x in sorted(fam, key=lambda x: (data[x], x))[1:]}
    return art[~art.article_id.isin(via)], len(via)


# ===================================================================================== 3. reti di testate
CODICI = {
    "adsense": r"\bca-pub-\d{10,20}\b", "analytics": r"\bUA-\d{4,10}-\d{1,3}\b|\bG-[A-Z0-9]{8,12}\b",
    "tag_manager": r"\bGTM-[A-Z0-9]{4,9}\b", "facebook_pixel": r"fbq\(\s*['\"]init['\"]\s*,\s*['\"](\d{8,20})",
    "facebook_app": r"fb:app_id['\"]?\s+content=['\"](\d+)", "chartbeat": r"_sf_async_config\.uid\s*=\s*(\d+)",
    "quantcast": r"qacct\s*[:=]\s*['\"](p-[A-Za-z0-9_-]+)", "comscore": r"\bc2[\"']?\s*[:=]\s*[\"'](\d{6,9})",
}
LEGALE = re.compile(r"privacy|terms of (use|service)|do not sell|cookie policy|terms (&|and) conditions", re.I)
PIATTAFORME = {"substack.com", "medium.com", "wordpress.com", "blogspot.com", "blogger.com", "ghost.io", "squarespace.com", "wix.com",
               "weebly.com", "google.com", "tumblr.com", "safesubscribe.org", "safesubscribe.com", "constantcontact.com", "email-comply.com"}
GENERICHE = re.compile(r"staff|editor|admin|desk|team|report|news|press|wire|contributor|guest|associated|reuters|^by$")


def _dominio_registrato(u):
    import tldextract
    return tldextract.extract(u).top_domain_under_public_suffix


def _impronte_html(d, cartelle, pagine_max=30, quota_codice=0.5, quota_legale=0.8):
    """Codici di account (pubblicità, statistiche) e link legali esterni nelle pagine di articolo della testata."""
    from urllib.parse import urljoin, urlparse
    from lxml import html as LH
    f = next((Path(c, "html", f"{d}.jsonl.gz") for c in cartelle if Path(c, "html", f"{d}.jsonl.gz").exists()), None)
    codici, legali, n = collections.Counter(), collections.Counter(), 0
    if f is not None:
        for riga in gzip.open(f, "rt"):
            r = json.loads(riga)
            if r.get("tipo") != "articolo" or not r.get("html"):
                continue
            n += 1
            h = r["html"]
            codici.update({f"{k}:{m}" for k, p in CODICI.items() for m in set(re.findall(p, h))})
            try:
                albero = LH.fromstring(h)
            except Exception:
                albero = None
            if albero is not None:
                casa, visti = _dominio_registrato(r["url"]), set()
                for a in albero.xpath("//a[@href]"):
                    if LEGALE.search(a.text_content() or ""):
                        u = urljoin(r["url"], a.get("href").strip())
                        if _dominio_registrato(u) and _dominio_registrato(u) != casa:
                            p = urlparse(u)
                            visti.add((p.netloc.lower().removeprefix("www.") + p.path.rstrip("/")).lower())
                legali.update(visti)
            if n >= pagine_max:
                break
    return {"testata": d, "pagine": n, "codici": [k for k, v in codici.items() if n >= 2 and v >= quota_codice * n],
            "legali": [k for k, v in legali.items() if n >= 5 and v >= quota_legale * n]}


def reti_di_testate(art, copie, cartelle, disconnect, n_jobs=32, min_copie=2, quota_relazione=0.10, min_articoli_autore=2):
    """Le reti di siti, senza guardare le etichette: due testate sono collegate se condividono articoli (copie), codici di
    account, il titolare legale (privacy policy esterna), il dominio o il redirect, o gli stessi autori.
    Ritorna (rete per testata, collegamenti con i segnali, conteggio delle coppie per segnale)."""
    from joblib import Parallel, delayed
    testate = sorted(art.domain.unique())
    collegamenti, per_segnale = collections.defaultdict(set), collections.Counter()

    def collega(coppie, segnale):
        for a, b in coppie:
            if a != b:
                collegamenti[frozenset((a, b))].add(segnale); per_segnale[segnale] += 1

    n_articoli = art.groupby("domain").size()
    sistematica = lambda p, n: n >= quota_relazione * min(n_articoli[d] for d in p)
    n_copie = collections.Counter(frozenset(p) for p in zip(copie.testata_a, copie.testata_b))
    collega([tuple(p) for p, n in n_copie.items() if n >= min_copie and sistematica(p, n)], "copie")
    impronte = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_impronte_html)(d, cartelle) for d in testate)).set_index("testata")

    def per_valore(colonna):
        v = collections.defaultdict(set)
        for d, lista in impronte[colonna].items():
            for x in lista:
                v[x].add(d)
        return {k: s for k, s in v.items() if len(s) >= 2}

    condivisi = per_valore("codici")
    for tipo in CODICI:
        collega([p for k, s in condivisi.items() if k.startswith(tipo + ":") for p in itertools.combinations(sorted(s), 2)], f"codice {tipo}")
    servizi = set()
    for lista in json.load(open(disconnect))["categories"].values():
        for ente in lista:
            for sotto in ente.values():
                for url, domini in sotto.items():
                    if isinstance(domini, list):
                        servizi |= {_dominio_registrato("http://" + x) for x in domini} | {_dominio_registrato(url)}
    esclusi = (servizi | PIATTAFORME) - {""}
    legali = {k: s for k, s in per_valore("legali").items() if _dominio_registrato("http://" + k) not in esclusi}
    collega([p for s in legali.values() for p in itertools.combinations(sorted(s), 2)], "titolare legale")
    registrato = {d: _dominio_registrato(d) for d in testate}
    casa = {}
    for d in testate:
        f = next((Path(c, "stato", f"{d}.json") for c in cartelle if Path(c, "stato", f"{d}.json").exists()), None)
        casa[d] = (_dominio_registrato(json.load(open(f)).get("homepage", "")) if f else "") or registrato[d]
    for nome, chiave in [("stesso dominio registrato", registrato), ("redirect allo stesso dominio", casa)]:
        collega([p for _, g in itertools.groupby(sorted(testate, key=chiave.get), key=chiave.get) for p in itertools.combinations(list(g), 2)],
                f"dominio: {nome}")
    autore = art.autore.fillna("").str.lower().str.strip()
    buoni = autore.str.split().str.len().between(2, 4) & ~autore.str.contains(GENERICHE)
    firme = art[buoni].assign(autore=autore[buoni]).groupby(["autore", "domain"]).size()
    firme = firme[firme >= min_articoli_autore].reset_index()
    in_comune = collections.Counter()
    for _, g in firme.groupby("autore"):
        quanti = dict(zip(g.domain, g[0]))
        for a, b in itertools.combinations(sorted(quanti), 2):
            in_comune[frozenset((a, b))] += min(quanti[a], quanti[b])
    collega([tuple(p) for p, n in in_comune.items() if sistematica(p, n)], "autori")
    G = nx.Graph(); G.add_nodes_from(testate); G.add_edges_from(tuple(p) for p in collegamenti)
    rete = {d: i for i, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)) for d in c}
    return rete, collegamenti, per_segnale


# ===================================================================================== 4. il dataset
def gruppi(testate, rete, newsguard_tutti):
    """Gruppi = reti ∪ stesso owner ∪ stesso Parent Domain (da tutti i feed NewsGuard passati)."""
    nodi = sorted(set(testate) | set(rete))
    G = nx.Graph(); G.add_nodes_from(nodi)
    per_rete = collections.defaultdict(list)
    for d, r in rete.items():
        per_rete[r].append(d)
    for g in per_rete.values():
        G.add_edges_from(zip(g, g[1:]))
    for ng in newsguard_tutti:
        for campo, norma in [("Owner", normalizza_owner), ("Parent Domain", lambda x: "" if str(x) == "nan" else str(x).strip().lower())]:
            s = pd.Series({d: norma(ng[campo].get(d, "")) if d in ng.index else "" for d in nodi})
            s = s[s != ""]
            for _, g in s.groupby(s):
                G.add_edges_from(zip(g.index, g.index[1:]))
    return {d: i for i, c in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)) for d in c}


def togli_copie_tra_gruppi(art, copie):
    """Di un testo pubblicato da testate di gruppi diversi si tiene la copia più vecchia: così lo stesso testo non sta
    mai nel training e nel test."""
    presenti = set(art.article_id)
    copie = copie[copie.articolo_a.isin(presenti) & copie.articolo_b.isin(presenti)]
    C = nx.Graph(); C.add_edges_from(zip(copie.articolo_a, copie.articolo_b))
    info = art.set_index("article_id")
    via = set()
    for fam in nx.connected_components(C):
        fam = sorted(fam, key=lambda x: (info.data[x], x))
        if len({info.gruppo[x] for x in fam}) > 1:
            primo = info.gruppo[fam[0]]
            via |= {x for x in fam if info.gruppo[x] != primo}
    return art[~art.article_id.isin(via)], len(via)


def testate_e_fold(art, min_valutate=20, per_gruppo=5, min_training=5):
    """Testate valutate (almeno `min_valutate` articoli, al più `per_gruppo` per gruppo, a caso senza guardare le
    etichette) e di solo training (almeno `min_training` articoli, peso min(1, per_gruppo/n)); 5 fold per gruppo,
    stratificati per livello, fatti sulle valutate; le altre testate seguono il loro gruppo."""
    from sklearn.model_selection import StratifiedGroupKFold
    t = art.groupby("domain").agg(gruppo=("gruppo", "first"), label=("label", "first"), articoli=("article_id", "size"))
    t = t[t.articoli >= min_training]
    idonee = t[t.articoli >= min_valutate].reset_index()
    valutate = set(idonee.sample(frac=1, random_state=SEED).groupby("gruppo").head(per_gruppo).domain)
    t["valutazione"] = t.index.isin(valutate)
    t = t.sort_index()
    val = t[t.valutazione].reset_index()
    val["fold"] = 0
    for i, (_, idx) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(val, val.label, val.gruppo), 1):
        val.loc[idx, "fold"] = i
    fold_del_gruppo = val.groupby("gruppo").fold.first()
    rng = np.random.default_rng(SEED)
    senza = sorted(set(t.gruppo) - set(fold_del_gruppo.index))
    fold_del_gruppo = pd.concat([fold_del_gruppo, pd.Series(rng.integers(1, 6, len(senza)), index=senza)])
    t["fold"] = t.gruppo.map(fold_del_gruppo).astype(int)
    return t


def ricalcola_peso(t, per_gruppo=5):
    t = t.copy()
    t["testate_nel_gruppo"] = t.groupby("gruppo").label.transform("size")
    t["peso"] = np.minimum(1.0, per_gruppo / t.testate_nel_gruppo)
    return t


# ===================================================================================== 5. metriche
def pesi_bilanciati(label, peso):
    """Peso di ogni riga: il peso della sua testata, con le classi bilanciate sui pesi."""
    label, peso = np.asarray(label), np.asarray(peso, dtype=float)
    per_classe = pd.Series(peso).groupby(label).sum()
    return peso * peso.sum() / (len(per_classe) * per_classe.reindex(label).to_numpy())


def voto(valori):
    """Voto di maggioranza; a parità vince il livello più basso."""
    c = pd.Series(valori).value_counts()
    return int(c[c == c.max()].index.min())


def metriche(y, p):
    """F1 macro sui 5 livelli, accuratezza, tolerant (±1 livello: pieno credito o metà), binario (affidabile = l3-l5)."""
    from sklearn.metrics import accuracy_score, f1_score
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    d = np.abs(y - p)
    return {"F1 macro": f1_score(y, p, average="macro", labels=range(5), zero_division=0), "accuratezza": accuracy_score(y, p),
            "tolerant ±1": float((d <= 1).mean()), "tolerant ½": float(np.where(d == 0, 1, np.where(d == 1, 0.5, 0)).mean()),
            "binario": float(((y >= 2) == (p >= 2)).mean()),
            "binario F1 macro": f1_score(y >= 2, p >= 2, average="macro", zero_division=0)}


def _veloci(y, p):
    M = np.bincount(y * 5 + p, minlength=25).reshape(5, 5)
    tp = np.diag(M); den = M.sum(0) + M.sum(1)
    d = np.abs(y - p)
    return np.array([np.divide(2 * tp, den, out=np.zeros(5), where=den > 0).mean(), (d <= 1).mean(), ((y >= 2) == (p >= 2)).mean()])


def confronto(y, gruppo, previsioni, riferimenti, B=2000):
    """Metriche di ogni modello, con intervalli al 95% da un bootstrap per gruppo (F1, tolerant ±1, binario), e le
    differenze con i modelli di riferimento sugli stessi ricampionamenti."""
    y = np.asarray(y).astype(int); g = np.asarray(gruppo)
    u = np.unique(g); pos = {x: np.where(g == x)[0] for x in u}
    rng = np.random.default_rng(0)
    camp = [np.concatenate([pos[x] for x in rng.choice(u, len(u))]) for _ in range(B)]
    BS = {k: np.array([_veloci(y[i], np.asarray(p).astype(int)[i]) for i in camp]) for k, p in previsioni.items()}
    righe = []
    for k, p in previsioni.items():
        r = {"modello": k, **metriche(y, p)}
        for j, m in enumerate(["F1 macro", "tolerant ±1", "binario"]):
            r[f"{m} IC"] = f"[{np.percentile(BS[k][:, j], 2.5):.3f}; {np.percentile(BS[k][:, j], 97.5):.3f}]"
        for rif in riferimenti:
            if rif != k:
                d = BS[k] - BS[rif]
                r[f"ΔF1 vs {rif}"] = f"{metriche(y, p)['F1 macro'] - metriche(y, previsioni[rif])['F1 macro']:+.3f} " \
                                     f"[{np.percentile(d[:, 0], 2.5):+.3f}; {np.percentile(d[:, 0], 97.5):+.3f}]"
        righe.append(r)
    return pd.DataFrame(righe).set_index("modello")


# ===================================================================================== 6. classificatore a feature
def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, max_features=100_000, strip_accents="unicode")


def _fold_feature(art, fold, griglia):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    tr, te = art[art.fold != fold], art[(art.fold == fold) & art.valutazione]
    medie = {}
    if len(griglia) > 1:                                   # C con la F1 per testata più alta in 3 fold interni per gruppi
        testate = tr.drop_duplicates("domain").reset_index(drop=True)
        punti = {c: [] for c in griglia}
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(testate, testate.label, testate.gruppo):
            a, b = tr[tr.domain.isin(testate.domain[itr])], tr[tr.domain.isin(testate.domain[iva]) & tr.valutazione]
            v = _tfidf(); Xa, Xb = v.fit_transform(a.text), v.transform(b.text)
            for c in griglia:
                p = LogisticRegression(C=c, max_iter=1000, random_state=SEED).fit(Xa, a.label, sample_weight=pesi_bilanciati(a.label, a.peso)).predict(Xb)
                t = pd.DataFrame({"d": b.domain.values, "y": b.label.values, "p": p}).groupby("d").agg(y=("y", "first"), p=("p", voto))
                punti[c].append(f1_score(t.y, t.p, average="macro", labels=range(5), zero_division=0))
        medie = {c: float(np.mean(s)) for c, s in punti.items()}
    C = max(medie, key=medie.get) if medie else griglia[0]
    v = _tfidf()
    clf = LogisticRegression(C=C, max_iter=1000, random_state=SEED).fit(v.fit_transform(tr.text), tr.label, sample_weight=pesi_bilanciati(tr.label, tr.peso))
    P = np.zeros((len(te), 5)); P[:, clf.classes_] = clf.predict_proba(v.transform(te.text))
    out = pd.DataFrame(P, columns=CLASSI).assign(article_id=te.article_id.values, domain=te.domain.values, fold=fold)
    return out, C


def classificatore_feature(art, griglia=(0.1, 1, 10, 100), n_jobs=5):
    """TF-IDF (parole e coppie di parole) + regressione logistica, un modello per fold; probabilità di ogni articolo di test."""
    from joblib import Parallel, delayed
    ris = Parallel(n_jobs=n_jobs)(delayed(_fold_feature)(art, f, list(griglia)) for f in sorted(art.fold.unique()))
    return pd.concat([r[0] for r in ris], ignore_index=True), [r[1] for r in ris]


# ===================================================================================== 7. ModernBERT
def unita(token_ids, lunghezza, passo=None):
    """Le unità che legge ModernBERT: con passo=None solo l'inizio (`lunghezza` token), altrimenti pezzi consecutivi di
    `lunghezza` token che partono ogni `passo` token (sovrapposizione = lunghezza − passo)."""
    if passo is None or len(token_ids) <= lunghezza:
        return [token_ids[:lunghezza]]
    inizi = list(range(0, len(token_ids) - lunghezza + passo, passo))
    return [token_ids[s:s + lunghezza] for s in inizi if s < len(token_ids)]


def addestra_e_predici(U, t, fold, cartella, registra, max_token, per_testata_epoca, epoche=2, lr=2e-5, batch=4, batch_effettivo=32,
                       batch_eval=16, per_gruppo=5, gb_gpu=6.0, modello="answerdotai/ModernBERT-base", seme=SEED):
    """Un fold: ModernBERT-base addestrato sulle unità (articoli troncati o pezzi) delle testate di training, poi le
    probabilità di tutte le unità delle testate valutate del fold. A ogni epoca al più `per_gruppo` testate per gruppo
    (diverse a ogni epoca) e `per_testata_epoca` unità per testata. Si salva `fold<k>.csv` (una riga per unità di test):
    se c'è già, il fold si salta."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    file = Path(cartella) / f"fold{fold}.csv"
    if file.exists():
        registra(f"fold {fold}: già fatto, salto"); return pd.read_csv(file)
    while torch.cuda.is_available() and (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30 < gb_gpu:
        registra("GPU occupata da altri: riprovo tra 5 minuti"); time.sleep(300)
    s = int(seme + fold); random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(modello)
    testata_di = U.domain.to_numpy(); ids = U.ids.tolist(); lab = t.label.reindex(U.domain).to_numpy().astype(int)
    tr_testate = t.index[t.fold != fold]; te_testate = t.index[(t.fold == fold) & t.valutazione]
    unita_di = {d: np.where(testata_di == d)[0] for d in tr_testate}
    testate_di = {g: sorted(x.index) for g, x in t.loc[tr_testate].groupby("gruppo")}
    conta = t.loc[tr_testate].groupby("label").peso.sum().reindex(range(5), fill_value=0).to_numpy()
    pesi = torch.tensor(conta.sum() / (5 * np.maximum(conta, 1e-9)), dtype=torch.float32, device=device)

    def campione(epoca):
        rng = np.random.default_rng(seme * 1000 + fold * 100 + epoca)
        pezzi = []
        for g in sorted(testate_di):
            ds = testate_di[g]
            for d in (rng.choice(ds, per_gruppo, replace=False) if len(ds) > per_gruppo else ds):
                u = unita_di[d]
                pezzi.append(rng.choice(u, min(len(u), per_testata_epoca), replace=False))
        c = np.concatenate(pezzi); rng.shuffle(c)
        return c

    def lotto(indici):
        seq = [[tok.cls_token_id] + list(ids[i]) + [tok.sep_token_id] for i in indici]
        L = max(len(x) for x in seq)
        x = np.full((len(seq), L), tok.pad_token_id, dtype=np.int64); m = np.zeros_like(x)
        for r, q in enumerate(seq):
            x[r, :len(q)] = q; m[r, :len(q)] = 1
        return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)

    campioni = [campione(e) for e in range(1, epoche + 1)]
    model = AutoModelForSequenceClassification.from_pretrained(modello, num_labels=5, reference_compile=False).to(device)
    nodecay = ["bias", "LayerNorm.weight", "norm.weight"]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nodecay)], "weight_decay": 0.01},
                             {"params": [p for n, p in model.named_parameters() if any(x in n for x in nodecay)], "weight_decay": 0.0}], lr=lr)
    while True:                                            # se la memoria GPU non basta si riparte con batch dimezzato
        try:
            accumulo = max(1, batch_effettivo // batch)
            passi = sum(math.ceil(len(c) / (batch * accumulo)) for c in campioni)
            sched = get_linear_schedule_with_warmup(opt, int(0.1 * passi), passi)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            loss_fn = torch.nn.CrossEntropyLoss(weight=pesi)
            t0 = time.time()
            for e, camp in enumerate(campioni, 1):
                model.train(); tot = 0.0; opt.zero_grad()
                for k in range(0, len(camp), batch):
                    pezzo = camp[k:k + batch]
                    x, m = lotto(pezzo)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        loss = loss_fn(model(input_ids=x, attention_mask=m).logits.float(), torch.tensor(lab[pezzo], device=device))
                    scaler.scale(loss / accumulo).backward(); tot += loss.item() * len(pezzo)
                    if (k // batch + 1) % accumulo == 0 or k + batch >= len(camp):
                        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
                registra(f"fold {fold} epoca {e}/{epoche}: {len(camp)} unità, loss {tot / len(camp):.4f} ({time.time() - t0:.0f} s)")
            break
        except RuntimeError as err:
            if not any(x in str(err) for x in ["out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUDA error", "CUDACachingAllocator", "NVML"]) or batch == 1:
                raise
            batch, batch_eval = batch // 2, max(1, batch_eval // 2)
            registra(f"fold {fold}: memoria GPU insufficiente, riparto con batch {batch}")
            del opt; torch.cuda.empty_cache()
            model = AutoModelForSequenceClassification.from_pretrained(modello, num_labels=5, reference_compile=False).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.eval()
    te = np.where(np.isin(testata_di, te_testate))[0]
    ordine = te[np.argsort([len(ids[i]) for i in te])]
    P = np.zeros((len(ordine), 5), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(ordine), batch_eval):
            x, m = lotto(ordine[k:k + batch_eval])
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                P[k:k + batch_eval] = torch.softmax(model(input_ids=x, attention_mask=m).logits.float(), -1).cpu().numpy()
    out = U.iloc[ordine][["article_id", "domain", "pezzo", "token"]].reset_index(drop=True)
    out[CLASSI] = P
    out["fold"] = fold
    out.to_csv(file, index=False)
    registra(f"fold {fold}: finito, {len(out)} unità di test, memoria GPU massima "
             f"{torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.1f} GB")
    del model, opt; torch.cuda.empty_cache()
    return out


# ===================================================================================== 8. mettere insieme i pezzi
def aggrega(pezzi):
    """Dalle probabilità dei pezzi al livello di ogni testata, in sette modi. Ritorna {nome: Series testata -> livello}."""
    P = pezzi[CLASSI].to_numpy()
    pezzi = pezzi.assign(voto=P.argmax(1), sicurezza=P.max(1))
    logp = pd.DataFrame(np.log(np.clip(P, 1e-9, 1)), columns=CLASSI, index=pezzi.index)
    da_prob = lambda df: pd.Series(df[CLASSI].to_numpy().argmax(1), index=df.index)
    per_art = {
        "media": pezzi.groupby("article_id")[CLASSI].mean(),
        "media geometrica": logp.groupby(pezzi.article_id).mean().pipe(lambda x: np.exp(x)),
        "media pesata per lunghezza": pezzi[CLASSI].mul(pezzi.token, axis=0).groupby(pezzi.article_id).sum().div(
            pezzi.groupby("article_id").token.sum(), axis=0),
        "pezzo più sicuro": pezzi.loc[pezzi.groupby("article_id").sicurezza.idxmax()].set_index("article_id")[CLASSI],
        "solo il primo pezzo": pezzi[pezzi.pezzo == 0].set_index("article_id")[CLASSI],
    }
    testata_di = pezzi.groupby("article_id").domain.first()
    out = {}
    voto_art = pezzi.groupby("article_id").voto.agg(voto)
    out["maggioranza (pezzi → articolo → testata)"] = voto_art.groupby(testata_di).agg(voto)
    for nome, pa in per_art.items():
        out[f"{nome} (pezzi → articolo), media (→ testata)"] = da_prob(pa.groupby(testata_di.reindex(pa.index)).mean())
    out["tutti i pezzi della testata insieme"] = da_prob(pezzi.groupby("domain")[CLASSI].mean())
    return out
