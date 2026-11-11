"""Tentativi 14 e 16: le funzioni dei notebook (il 16 aggiunge la riestrazione dall HTML, sezione 9).

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
DATASET = DATI / "tentativi" / "16_DataPreparation"          # il dataset di questo tentativo (articoli.csv.gz, testate.csv, cache)


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


# ===================================================================================== 9. riestrazione dall'HTML (tentativo 16)
def estrai_testata(dominio, cartelle):
    """Riestrae dall'HTML salvato il testo di ogni articolo di una testata con trafilatura in modalità precisione
    (senza commenti, tabelle, immagini, link e formattazione). Titolo, data e autore vengono dallo scraping."""
    import trafilatura
    righe = []
    for c in cartelle:
        fa, fh = Path(c, "articoli", f"{dominio}.jsonl"), Path(c, "html", f"{dominio}.jsonl.gz")
        if not fa.exists():
            continue
        html = {}
        if fh.exists():
            for r in gzip.open(fh, "rt"):
                d = json.loads(r)
                if d.get("tipo") == "articolo" and d.get("html"):
                    html[d["url"]] = d["html"]
        for r in open(fa):
            a = json.loads(r)
            h, nuovo = html.get(a["url"]), None
            if h:
                try:
                    nuovo = trafilatura.extract(h, url=a["url"], favor_precision=True, include_comments=False, include_tables=False,
                                                include_images=False, include_links=False, include_formatting=False)
                except Exception:
                    nuovo = None
            righe.append({"domain": dominio, "url": a["url"], "titolo": a.get("titolo"), "data": a.get("data"), "autore": a.get("autore"),
                          "testo_scraping": a.get("testo"), "text": nuovo, "ha_html": h is not None})
    return righe


# ===================================================================================== 10. dagli articoli alle testate, misure, grafico (tentativo 16)
# Le sei configurazioni del notebook 3, con il nome della colonna (o delle colonne) che ogni modello scrive per ogni articolo.
CONFIGURAZIONI = {
    "5 classi": ("1. 5 classi", "impara il livello l1…l5\nsu tutti gli articoli · testata = voto", "classificazione a 5 classi"),
    "binario": ("2. binario", "impara affidabile sì/no\nsu tutti gli articoli · testata = voto", "classificazione binaria"),
    "binario_estremi": ("3. binario sugli estremi", "impara affidabile sì/no\nsolo su l1, l2, l4, l5 · testata = voto", "classificazione binaria"),
    "regressione": ("4. regressione", "impara il punteggio 0–100\nsu tutti gli articoli · testata = media", "regressione sul punteggio"),
    "regressione_bilanciata": ("5. regressione bilanciata", "impara il punteggio, ogni livello pesa uguale\nsu tutti gli articoli · testata = media",
                               "regressione sul punteggio"),
    "regressione_estremi": ("6. regressione sugli estremi", "impara il punteggio 0–100\nsolo su l1, l2, l4, l5 · testata = media", "regressione sul punteggio"),
}
SOGLIE_NEWSGUARD = [39, 59, 74, 99]


def colonna(config):
    """Le colonne che una configurazione scrive per ogni articolo."""
    return CLASSI if config == "5 classi" else f"p_{config}" if config.startswith("binario") else f"s_{config}"


def testate_da_articoli(pa, t):
    """Una riga per testata: classificatori = voto degli articoli (e media delle probabilità), regressioni = media dei punteggi.
    Per le regressioni i 5 livelli si ricavano con le soglie NewsGuard e con le quote del training del fold."""
    g = pa.groupby("domain")
    T = pd.DataFrame({"livello vero": t.label, "punteggio vero": t.score, "fold": t.fold})
    for config in CONFIGURAZIONI:
        c = colonna(config)
        if config == "5 classi" and set(CLASSI) <= set(pa.columns):
            media = g[CLASSI].mean()
            T["5 classi"] = pa.assign(v=pa[CLASSI].to_numpy().argmax(1)).groupby("domain").v.agg(voto)
            T["5 classi: livello atteso"] = pd.Series(media.to_numpy() @ np.arange(5), index=media.index)
        elif config.startswith("binario") and c in pa:
            T[config] = g[c].agg(lambda p: voto((p >= 0.5).astype(int)))
            T[f"{config}: probabilità"] = g[c].mean()
        elif config.startswith("regressione") and c in pa:
            T[config] = g[c].mean()
    T = T.reindex(t.index)
    for config in [c for c in CONFIGURAZIONI if c.startswith("regressione") and c in T]:
        T[f"{config} → soglie NewsGuard"] = np.digitize(T[config], SOGLIE_NEWSGUARD, right=True)
        livelli = pd.Series(0, index=T.index)
        for k in sorted(T.fold.unique()):
            te = T.fold == k
            quote = np.cumsum(np.bincount(T["livello vero"][~te], minlength=5))[:-1] / (~te).sum()
            livelli[te] = np.searchsorted(np.quantile(T[config][te], quote), T[config][te])
        T[f"{config} → quote del training"] = livelli
    return T


def misure_testate(T):
    """Le due tabelle del notebook 3: i 5 livelli (F1, tolerant, con intervalli da bootstrap sulle testate) e il binario
    (accuratezza, F1, AUC su tutte le testate e solo sulle estreme)."""
    from sklearn.metrics import f1_score, roc_auc_score
    cinque = {}
    if "5 classi" in T:
        cinque["1. 5 classi"] = T["5 classi"]
    for config in [c for c in CONFIGURAZIONI if c.startswith("regressione") and c in T]:
        for modo in ["soglie NewsGuard", "quote del training"]:
            cinque[f"{CONFIGURAZIONI[config][0]} → {modo}"] = T[f"{config} → {modo}"]
    tab5 = confronto(T["livello vero"], T.index, {k: v.to_numpy() for k, v in cinque.items()}, riferimenti=list(cinque)[:1]) if cinque else None
    vero, estreme = (T["livello vero"] >= 2).to_numpy(), (T["livello vero"] != 2).to_numpy()
    binari = {}
    if "5 classi" in T:
        binari["1. 5 classi"] = ((T["5 classi"] >= 2).to_numpy(), T["5 classi: livello atteso"].to_numpy())
    for config in [c for c in CONFIGURAZIONI if c.startswith("binario") and c in T]:
        binari[CONFIGURAZIONI[config][0]] = (T[config].to_numpy().astype(bool), T[f"{config}: probabilità"].to_numpy())
    for config in [c for c in CONFIGURAZIONI if c.startswith("regressione") and c in T]:
        binari[f"{CONFIGURAZIONI[config][0]} (≥ 60)"] = ((T[config] >= 60).to_numpy(), T[config].to_numpy())
    tab2 = pd.DataFrame({nome: {f"{dove}: {mis}": f(vero[m], p[m], c[m]) for dove, m in [("tutte", np.ones(len(T), bool)), ("solo estreme", estreme)]
                                for mis, f in [("accuratezza", lambda v, p, c: (p == v).mean()), ("F1 macro", lambda v, p, c: f1_score(v, p, average="macro")),
                                               ("AUC", lambda v, p, c: roc_auc_score(v, c))]}
                         for nome, (p, c) in binari.items()}).T
    return tab5, tab2


STILE_GRAFICI = {"figure.dpi": 110, "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                 "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.grid": True,
                 "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 10,
                 "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False}


def grafico_riassunto(T, tab5, tab2, titolo, file):
    """Il grafico finale del notebook 3: una riga per configurazione presente, quattro misure con i valori sulle barre."""
    import matplotlib.pyplot as plt
    with plt.rc_context(STILE_GRAFICI):
        _grafico_riassunto(T, tab5, tab2, titolo, file)


def _grafico_riassunto(T, tab5, tab2, titolo, file):
    import matplotlib.pyplot as plt
    GRIGIO, CHIARO = "#52514e", "#b5b3ab"
    FAMIGLIA = {"classificazione a 5 classi": "#2a78d6", "classificazione binaria": "#1baf7a", "regressione sul punteggio": "#eb6834"}
    vero, estreme = (T["livello vero"] >= 2).to_numpy(), (T["livello vero"] != 2).to_numpy()
    modelli = []
    for config, (nome, descrizione, famiglia) in CONFIGURAZIONI.items():
        if config not in T:
            continue
        riga5 = "1. 5 classi" if config == "5 classi" else f"{nome} → quote del training" if config.startswith("regressione") else None
        riga2 = f"{nome} (≥ 60)" if config.startswith("regressione") else nome
        modelli.append((nome, descrizione, famiglia, riga5, riga2))
    ic = lambda testo: [float(x) for x in testo.strip("[]").split(";")]
    metriche = [("F1 macro, 5 livelli", lambda m: (tab5.loc[m[3], "F1 macro"], ic(tab5.loc[m[3], "F1 macro IC"])) if m[3] else None, 0.2, "a caso"),
                ("tolerant ±1", lambda m: (tab5.loc[m[3], "tolerant ±1"], ic(tab5.loc[m[3], "tolerant ±1 IC"])) if m[3] else None, None, None),
                ("binario, tutte le testate", lambda m: (tab2.loc[m[4], "tutte: accuratezza"], None), vero.mean(), "sempre «affidabile»"),
                ("binario, solo l1 l2 l4 l5", lambda m: (tab2.loc[m[4], "solo estreme: accuratezza"], None), vero[estreme].mean(), "sempre «affidabile»")]
    fig = plt.figure(figsize=(18, 1.05 * len(modelli) + 0.8))
    griglia = fig.add_gridspec(1, 5, width_ratios=[1.6, 1, 1, 1, 1], wspace=0.14)
    y = np.arange(len(modelli))[::-1]
    testo = fig.add_subplot(griglia[0]); testo.axis("off"); testo.set_ylim(-0.6, len(modelli) - 0.4)
    for yy, (nome, descrizione, famiglia, _, _) in zip(y, modelli):
        testo.text(0, yy + 0.12, nome, fontsize=10.5, fontweight="bold", va="center", color=FAMIGLIA[famiglia])
        testo.text(0, yy - 0.2, descrizione, fontsize=8.5, va="center", color=GRIGIO, linespacing=1.3)
    testo.set_title("configurazione", loc="left")
    for j, (tit, valore, riferimento, nome_rif) in enumerate(metriche, 1):
        ax = fig.add_subplot(griglia[j], sharey=testo)
        for yy, m in zip(y, modelli):
            v = valore(m)
            if v is None:
                ax.text(0.42, yy, "non si applica", va="center", fontsize=8, color=CHIARO); continue
            ax.barh(yy, v[0], height=0.6, color=FAMIGLIA[m[2]], alpha=0.9)
            if v[1]:
                ax.errorbar(v[0], yy, xerr=[[v[0] - v[1][0]], [v[1][1] - v[0]]], color=GRIGIO, lw=1, capsize=3)
            ax.text((v[1][1] if v[1] else v[0]) + 0.01, yy, f"{v[0]:.2f}", va="center", fontsize=9, fontweight="bold", color=GRIGIO)
        if riferimento is not None:
            ax.axvline(riferimento, color=GRIGIO, ls=":", lw=1)
            tit += f"\n··· {nome_rif}: {riferimento:.2f}"
        ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "", "0,5", "", "1"])
        ax.set_title(tit, loc="left", fontsize=9.5); ax.tick_params(left=False, labelleft=False)
        ax.grid(False); ax.grid(axis="x")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMIGLIA.values()], FAMIGLIA.keys(), ncol=3, loc="lower center",
               bbox_to_anchor=(0.55, -0.04 if len(modelli) > 3 else -0.1), fontsize=9)
    fig.suptitle(titolo, fontsize=12, fontweight="bold", x=0.01, ha="left")
    plt.savefig(file, dpi=180, bbox_inches="tight"); plt.show()


def modelli_lineari(Xtr, Xte, y, s, configs=tuple(CONFIGURAZIONI), C=10, alpha=1.0):
    """I modelli lineari del notebook 3 (regressione logistica, ridge) per le configurazioni richieste. Ritorna un dict
    colonna → previsioni per gli articoli di test."""
    from sklearn.linear_model import LogisticRegression, Ridge
    affidabile, estremi = y >= 2, y != 2
    bil = lambda e: pesi_bilanciati(e, np.ones(len(e)))
    out = {}
    if "5 classi" in configs:
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y, sample_weight=bil(y))
        P = np.zeros((Xte.shape[0], 5)); P[:, m.classes_] = m.predict_proba(Xte)
        out.update({c: P[:, i] for i, c in enumerate(CLASSI)})
    if "binario" in configs:
        out["p_binario"] = LogisticRegression(C=C, max_iter=2000).fit(Xtr, affidabile, sample_weight=bil(affidabile)).predict_proba(Xte)[:, 1]
    if "binario_estremi" in configs:
        out["p_binario_estremi"] = LogisticRegression(C=C, max_iter=2000).fit(Xtr[estremi], affidabile[estremi],
                                                                             sample_weight=bil(affidabile[estremi])).predict_proba(Xte)[:, 1]
    if "regressione" in configs:
        out["s_regressione"] = Ridge(alpha=alpha).fit(Xtr, s).predict(Xte)
    if "regressione_bilanciata" in configs:
        out["s_regressione_bilanciata"] = Ridge(alpha=alpha).fit(Xtr, s, sample_weight=bil(y)).predict(Xte)
    if "regressione_estremi" in configs:
        out["s_regressione_estremi"] = Ridge(alpha=alpha).fit(Xtr[estremi], s[estremi]).predict(Xte)
    return out


def aspetta_gpu(gb, registra):
    """Aspetta che sulla GPU ci siano almeno `gb` GB liberi (la GPU è condivisa)."""
    import torch
    while torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 < gb:
        registra(f"GPU occupata da altri ({torch.cuda.mem_get_info()[0] / 2**30:.1f} GB liberi): riprovo tra 5 minuti"); time.sleep(300)


# ===================================================================================== 11. embedding (tentativo 16)
def embedding_articoli(testi, modello, max_token, batch, istruzione, registra):
    """Embedding di ogni articolo con un modello di embedding (pooling sull'ultimo token, come Qwen3-Embedding), senza
    addestrarlo. Gli articoli si elaborano dal più corto al più lungo; i vettori escono normalizzati."""
    import torch
    import torch.nn.functional as Fn
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(modello, padding_side="left")
    m = AutoModel.from_pretrained(modello, torch_dtype=torch.float16).cuda().eval()
    testi = [f"Instruct: {istruzione}\nQuery:{x}" for x in testi]
    lunghezze = [len(x) for x in testi]
    ordine = np.argsort(lunghezze)
    E = None
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(ordine), batch):
            idx = ordine[k:k + batch]
            enc = tok([testi[i] for i in idx], padding=True, truncation=True, max_length=max_token, return_tensors="pt").to("cuda")
            h = m(**enc).last_hidden_state[:, -1]                                  # ultimo token (con il padding a sinistra)
            h = Fn.normalize(h.float(), p=2, dim=1).cpu().numpy()
            if E is None:
                E = np.zeros((len(testi), h.shape[1]), dtype=np.float32)
            E[idx] = h
            if (k // batch) % 500 == 0:
                registra(f"{k + len(idx):,}/{len(testi):,} articoli ({time.time() - t0:.0f} s)")
    del m; torch.cuda.empty_cache()
    return E


# ===================================================================================== 12. ModernBERT a 2048 token, classificatore o regressore (tentativo 16)
def modernbert_fold(art, k, config, cartella, registra, max_token=2048, epoche=5, lr=3e-5, batch=4, batch_effettivo=32,
                    quota_validazione=0.15, modello="answerdotai/ModernBERT-base", seme=SEED, gb_gpu=14, misura_test=False):
    """Un fold e una configurazione. ModernBERT legge ogni articolo intero (fino a `max_token` token). Il training usa
    tutti gli articoli delle testate degli altri fold (per le configurazioni "_estremi" solo l1, l2, l4, l5); una parte
    delle testate di training (`quota_validazione`, stratificata per livello) fa da validazione: dopo ogni epoca si misura
    sulla validazione e si tiene l'epoca migliore. Classificatori: entropia incrociata con classi bilanciate e label
    smoothing 0,1; regressori: errore quadratico sul punteggio/100. Salva le previsioni del fold di test e la curva.
    Con `misura_test` (solo per la diagnostica) misura anche il fold di test dopo ogni epoca: serve a vedere la curva,
    l'epoca si sceglie sempre sulla validazione. Precisione fp16: la V100 non supporta bf16."""
    import copy

    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    file = Path(cartella) / f"fold{k}.csv"
    if file.exists():
        fatto = pd.read_csv(file)
        if set(fatto.article_id) == set(art.article_id[art.fold == k]):     # stesso dataset: si riusa
            registra(f"{config}, fold {k}: già fatto, salto"); return fatto
        registra(f"{config}, fold {k}: il dataset è cambiato, rifaccio")
    aspetta_gpu(gb_gpu, registra)
    random.seed(seme + k); np.random.seed(seme + k); torch.manual_seed(seme + k); torch.cuda.manual_seed_all(seme + k)
    classificatore = not config.startswith("regressione")
    n_uscite = 5 if config == "5 classi" else 2 if classificatore else 1
    tok = AutoTokenizer.from_pretrained(modello)
    ids = tok(art.text.tolist(), add_special_tokens=True, truncation=True, max_length=max_token)["input_ids"]
    lab, score, fold, dom = art.label.to_numpy(), art.score.to_numpy() / 100, art.fold.to_numpy(), art.domain.to_numpy()
    obiettivo = lab if config == "5 classi" else (lab >= 2).astype(int) if classificatore else score
    testate = pd.DataFrame({"domain": dom, "label": lab, "fold": fold}).drop_duplicates("domain")
    tr_testate = testate[testate.fold != k]
    if config.endswith("estremi"):
        tr_testate = tr_testate[tr_testate.label != 2]
    if quota_validazione > 0:                      # una parte delle testate di training fa da validazione
        fit_t, val_t = train_test_split(tr_testate.domain, test_size=quota_validazione, stratify=tr_testate.label, random_state=seme + k)
    else:                                         # numero di epoche fisso: tutte le testate di training si usano per addestrare
        fit_t, val_t = tr_testate.domain, []
    fit = np.where(np.isin(dom, fit_t))[0]; val = np.where(np.isin(dom, val_t))[0]; test = np.where(fold == k)[0]
    pesi_righe = pesi_bilanciati(lab[fit], np.ones(len(fit)))                      # ogni livello pesa uguale (classi bilanciate)
    peso_di = dict(zip(fit, pesi_righe))
    device = torch.device("cuda")

    def lotti(indici, mescola):
        ordine = np.array(sorted(indici, key=lambda i: len(ids[i])))
        gruppi = [ordine[i:i + batch] for i in range(0, len(ordine), batch)]
        if mescola:
            np.random.shuffle(gruppi)
        return gruppi

    def tensori(gruppo):
        L = max(len(ids[i]) for i in gruppo)
        x = np.full((len(gruppo), L), tok.pad_token_id, dtype=np.int64); msk = np.zeros_like(x)
        for r, i in enumerate(gruppo):
            x[r, :len(ids[i])] = ids[i]; msk[r, :len(ids[i])] = 1
        return torch.from_numpy(x).to(device), torch.from_numpy(msk).to(device)

    def prevedi(model, indici):
        model.eval(); out = np.zeros((len(indici), n_uscite), dtype=np.float32); pos = {i: j for j, i in enumerate(indici)}
        with torch.no_grad():
            for g in lotti(indici, False):
                x, msk = tensori(g)
                with torch.autocast("cuda", dtype=torch.float16):
                    z = model(input_ids=x, attention_mask=msk).logits.float()
                z = torch.softmax(z, -1) if classificatore else z
                out[[pos[i] for i in g]] = z.cpu().numpy()
        return out

    def misura(P, indici):
        d = pd.DataFrame({"domain": dom[indici], "vero": obiettivo[indici] if classificatore else score[indici]})
        if classificatore:
            d["p"] = P.argmax(1); t_ = d.groupby("domain").agg(vero=("vero", "first"), p=("p", voto))
            return f1_score(t_.vero, t_.p, average="macro")
        d["p"] = P[:, 0]; t_ = d.groupby("domain").mean()
        return spearmanr(t_.vero, t_.p).correlation

    model = AutoModelForSequenceClassification.from_pretrained(modello, num_labels=n_uscite, reference_compile=False).to(device)
    if classificatore:
        conta = np.bincount(obiettivo[fit], minlength=n_uscite).astype(float)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(conta.sum() / (n_uscite * np.maximum(conta, 1)), dtype=torch.float32, device=device),
                                            label_smoothing=0.1)
    nodecay = ["bias", "LayerNorm.weight", "norm.weight"]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nodecay)], "weight_decay": 0.01},
                             {"params": [p for n, p in model.named_parameters() if any(x in n for x in nodecay)], "weight_decay": 0.0}], lr=lr)
    accumulo = max(1, batch_effettivo // batch)
    passi = epoche * int(np.ceil(len(fit) / (batch * accumulo)))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * passi), passi)
    scaler = torch.amp.GradScaler("cuda")
    curva, migliore, stato = [], -np.inf, None
    t0 = time.time()
    for e in range(1, epoche + 1):
        model.train(); opt.zero_grad(); tot, n = 0.0, 0
        gruppi = lotti(fit, True)
        for b, g in enumerate(gruppi, 1):
            x, msk = tensori(g)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(input_ids=x, attention_mask=msk).logits.float()
            if classificatore:
                loss = loss_fn(z, torch.tensor(obiettivo[g], device=device))
            else:
                w = torch.tensor([peso_di[i] for i in g] if config == "regressione_bilanciata" else [1.0] * len(g), dtype=torch.float32, device=device)
                loss = (w * (z[:, 0] - torch.tensor(score[g], dtype=torch.float32, device=device)) ** 2).sum() / w.sum()
            scaler.scale(loss / accumulo).backward(); tot += loss.item() * len(g); n += len(g)
            if b % accumulo == 0 or b == len(gruppi):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        valore = misura(prevedi(model, val), val) if len(val) else e      # senza validazione si tiene l'ultima epoca
        curva.append({"epoca": e, "loss": tot / n, "validazione": valore if len(val) else np.nan})
        if misura_test:
            curva[-1]["test"] = misura(prevedi(model, test), test)
        registra(f"{config}, fold {k}, epoca {e}/{epoche}: loss {tot / n:.4f}" + (f", validazione {valore:.3f}" if len(val) else "")
                 + (f", test {curva[-1]['test']:.3f}" if misura_test else "") + f" ({time.time() - t0:.0f} s)")
        if valore > migliore:
            migliore, stato = valore, copy.deepcopy({kk: v.detach().cpu() for kk, v in model.state_dict().items()})
    model.load_state_dict(stato)
    P = prevedi(model, test)
    out = pd.DataFrame({"article_id": art.article_id.to_numpy()[test], "domain": dom[test]})
    if config == "5 classi":
        out[CLASSI] = P
    elif classificatore:
        out[colonna(config)] = P[:, 1]
    else:
        out[colonna(config)] = P[:, 0] * 100
    out.to_csv(file, index=False)
    pd.DataFrame(curva).to_csv(Path(cartella) / f"curva_fold{k}.csv", index=False)
    registra(f"{config}, fold {k}: finito; epoca usata {int(np.nanargmax([c['validazione'] for c in curva])) + 1 if len(val) else epoche}, "
             f"memoria GPU massima {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    del model, opt; torch.cuda.empty_cache()
    return out


# ===================================================================================== 13. il confronto finale tra modelli (tentativo 16)
def misure_modelli(T, B=2000):
    """Per ogni modello (dict nome → previsioni per testata) e impostazione (5 classi, regressione): F1 sui 5 livelli,
    tolerant ±1, binario su tutte e sulle estreme, con intervalli al 95% (bootstrap sulle testate); e la differenza dal
    TF-IDF con la stessa impostazione (F1 e binario), sugli stessi ricampionamenti."""
    vero = next(iter(T.values()))["livello vero"]
    y = vero.to_numpy(); estreme = y != 2
    rng = np.random.default_rng(0)
    campioni = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    previsioni = {}
    for nome, t_ in T.items():
        t_ = t_.reindex(vero.index)
        if "5 classi" in t_:                    # affidabile: dal livello della testata, o dal voto degli articoli se c'è
            aff = t_["5 classi: affidabile"] if "5 classi: affidabile" in t_ else t_["5 classi"] >= 2
            previsioni[(nome, "5 classi")] = (t_["5 classi"].to_numpy().astype(int), aff.to_numpy().astype(bool))
        if "regressione" in t_:
            aff = t_["regressione: affidabile"] if "regressione: affidabile" in t_ else t_["regressione"] >= 60
            previsioni[(nome, "regressione")] = (t_["regressione → quote del training"].to_numpy().astype(int), aff.to_numpy().astype(bool))
    misure = {"F1 5 livelli": lambda l, a, i: metriche(y[i], l[i])["F1 macro"],
              "tolerant ±1": lambda l, a, i: (np.abs(y[i] - l[i]) <= 1).mean(),
              "binario, tutte": lambda l, a, i: ((y[i] >= 2) == a[i]).mean(),
              "binario, estreme": lambda l, a, i: ((y[i] >= 2) == a[i])[estreme[i]].mean()}
    tutte = np.arange(len(y))
    valori, intervalli, boot = {}, {}, {}
    for k, (l, a) in previsioni.items():
        valori[k] = {m: f(l, a, tutte) for m, f in misure.items()}
        boot[k] = {m: np.array([f(l, a, i) for i in campioni]) for m, f in misure.items()}
        intervalli[k] = {m: np.percentile(boot[k][m], [2.5, 97.5]) for m in misure}
    differenze = {}
    for (nome, imp) in previsioni:
        if nome != "TF-IDF" and ("TF-IDF", imp) in previsioni:
            differenze[(nome, imp)] = {f"Δ {m} dal TF-IDF": f"{valori[(nome, imp)][m] - valori[('TF-IDF', imp)][m]:+.3f} "
                                       f"[{np.percentile(boot[(nome, imp)][m] - boot[('TF-IDF', imp)][m], 2.5):+.3f}; "
                                       f"{np.percentile(boot[(nome, imp)][m] - boot[('TF-IDF', imp)][m], 97.5):+.3f}]"
                                       for m in ["F1 5 livelli", "binario, tutte"]}
    return pd.DataFrame(valori).T, intervalli, pd.DataFrame(differenze).T, vero


# Per il grafico con i dettagli: che cosa è ogni modello e che cosa misura ogni pannello.
DESCRIZIONI = {"TF-IDF": "TF-IDF\nparole e coppie di parole (1–2 grammi)\n+ logistica / ridge",
               "Embedding Qwen3 0.6B": "Embedding Qwen3 0,6B\ncongelato (non addestrato)\n+ logistica / ridge",
               "Embedding Qwen3 8B": "Embedding Qwen3 8B\ncongelato (non addestrato)\n+ logistica / ridge",
               "ModernBERT 2048": "ModernBERT-base, addestrato\n5 epoche · lr 3e-5 · batch 32\narticoli interi (2.048 token)"}


def grafico_confronto(valori, intervalli, vero, file, dettagli=False, descrizioni=None, nota=None):
    """Un grafico per tutto il confronto: una riga per modello, in blu la classificazione a 5 classi e in arancio la
    regressione; quattro misure, con i valori sulle barre e gli intervalli al 95%. Con `dettagli` accanto ai nomi c'è che
    cosa è ogni modello (`descrizioni`, altrimenti DESCRIZIONI), i titoli dicono su quali testate si misura, e sotto c'è
    come si addestra e come si testa (`nota`, altrimenti quella del notebook 6)."""
    import matplotlib.pyplot as plt
    BLU, ARANCIO, GRIGIO = "#2a78d6", "#eb6834", "#52514e"
    modelli = list(dict.fromkeys(m for m, _ in valori.index))
    y = np.arange(len(modelli))[::-1]
    v_ = np.asarray(vero); estreme = v_ != 2
    affidabili, affidabili_estreme = (v_ >= 2).mean(), (v_[estreme] >= 2).mean()
    if dettagli:
        riferimenti = {"F1 5 livelli": (0.2, "a caso"), "tolerant ±1": ((np.abs(v_ - 3) <= 1).mean(), "sempre l4, il livello più frequente"),
                       "binario, tutte": (affidabili, "baseline: sempre «affidabile»"),
                       "binario, estreme": (affidabili_estreme, "baseline: sempre «affidabile»")}
        titoli = {"F1 5 livelli": f"F1 macro sui 5 livelli (l1…l5)\ntest su tutte le {len(v_)} testate",
                  "tolerant ±1": f"tolerant ±1: livello giusto o sbagliato di uno\ntest su tutte le {len(v_)} testate",
                  "binario, tutte": f"affidabile sì/no (punteggio ≥ 60, cioè l3–l5)\ntest su tutte le {len(v_)} testate",
                  "binario, estreme": f"affidabile sì/no, test solo su l1–l2 contro l4–l5\n"
                                      f"({estreme.sum()} testate; le l3 escluse solo dal test)"}
    else:
        riferimenti = {"F1 5 livelli": (0.2, "a caso"), "binario, tutte": (affidabili, "sempre «affidabile»"),
                       "binario, estreme": (affidabili_estreme, "sempre «affidabile»")}
        titoli = {}
    with plt.rc_context(STILE_GRAFICI):
        h = (1.45 if dettagli else 1.2) * len(modelli) + 1.6                    # altezza in pollici; le distanze sotto sono fisse
        fig, assi = plt.subplots(1, 4, figsize=(18, h), sharey=True)
        for ax, metrica in zip(assi, ["F1 5 livelli", "tolerant ±1", "binario, tutte", "binario, estreme"]):
            for dy, imp, colore in [(0.19, "5 classi", BLU), (-0.19, "regressione", ARANCIO)]:
                for yy, m in zip(y, modelli):
                    if (m, imp) not in valori.index:
                        continue
                    v = valori.loc[(m, imp), metrica]; lo, hi = intervalli[(m, imp)][metrica]
                    ax.barh(yy + dy, v, height=0.36, color=colore)
                    ax.errorbar(v, yy + dy, xerr=[[v - lo], [hi - v]], color=GRIGIO, lw=1, capsize=2.5)
                    ax.text(hi + 0.01, yy + dy, f"{v:.2f}", va="center", fontsize=8.5, fontweight="bold", color=GRIGIO)
            titolo = titoli.get(metrica, metrica)
            if metrica in riferimenti:
                r, nome_r = riferimenti[metrica]; ax.axvline(r, color=GRIGIO, ls=":", lw=1); titolo += f"\n··· {nome_r}: {r:.2f}"
            ax.set_title(titolo, fontsize=9.5 if dettagli else 10); ax.set_xlim(0, 1.08); ax.grid(False); ax.grid(axis="x")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "", "0,5", "", "1"])
        descrizioni = descrizioni or DESCRIZIONI
        assi[0].set_yticks(y, [descrizioni.get(m, m) for m in modelli] if dettagli else modelli, fontsize=9.5 if dettagli else 10.5)
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in [BLU, ARANCIO]],
                   ["classificazione a 5 classi", "regressione sul punteggio (livelli con le quote del training; affidabile = ≥ 60)"],
                   ncol=2, loc="lower center", bbox_to_anchor=(0.5, -(0.37 if dettagli else 0.384) / h), fontsize=9.5)
        if dettagli:
            fig.text(0.01, -0.555 / h, nota or
                     "Addestramento: sempre su tutti gli articoli di tutte le testate dei 4 fold di training (l1…l5, anche le l3). "
                     "Test: le testate del fold escluso, una previsione per testata\n(classificazione: voto dei suoi articoli; "
                     "regressione: media dei punteggi previsti). Ogni testata è nel test una volta sola. "
                     "Baseline: il valore di chi risponde sempre allo stesso modo, senza leggere i testi.",
                     fontsize=8.5, color=GRIGIO, va="top")
        fig.suptitle(f"TF-IDF contro transformer · {len(vero)} testate politiche · 5 fold · linee grigie: intervalli al 95%",
                     fontsize=12.5, fontweight="bold", x=0.01, ha="left")
        plt.tight_layout(rect=(0, 0, 1, 1 - 0.75 / h) if h > 8 else None)    # figure alte: spazio fisso per il titolo
        plt.savefig(file, dpi=180, bbox_inches="tight"); plt.show()


# ===================================================================================== 14. la loss di ModernBERT epoca per epoca (tentativo 16)
def _media_lotti(perdite, pesi, rng, batch=4):
    """La loss come la scrive l'addestramento: per ogni lotto da `batch` articoli la media pesata (classi bilanciate),
    poi la media dei lotti pesata per la loro grandezza."""
    ordine = rng.permutation(len(perdite)); lotti = [ordine[i:i + batch] for i in range(0, len(ordine), batch)]
    return float(np.average([perdite[g].sum() / pesi[g].sum() for g in lotti], weights=[len(g) for g in lotti]))


def loss_epoche(cartella, t, eps=0.1, seme=0):
    """Per le due configurazioni di ModernBERT e i 5 fold: la loss di training dopo ogni epoca (dalle curve salvate),
    la stessa loss sugli articoli del fold di test dopo l'ultima epoca (dalle previsioni salvate) e i riferimenti.
    5 classi: entropia incrociata con classi bilanciate e label smoothing `eps`, come nell'addestramento; il minimo è la
    loss di un modello che riproduce esattamente il bersaglio smussato su ogni articolo (training imparato a memoria), e
    non è zero per via della label smoothing. Regressione: errore quadratico medio sul punteggio/100, mostrato come
    radice in punti NewsGuard (RMSE); il riferimento è prevedere sempre la media del training."""
    rng = np.random.default_rng(seme); righe = []
    for config in ["5 classi", "regressione"]:
        for k in range(1, 6):
            curva = pd.read_csv(Path(cartella) / config / f"curva_fold{k}.csv")
            te = pd.read_csv(Path(cartella) / config / f"fold{k}.csv")
            tr = t[t.fold != k]
            y_tr = np.repeat(tr.label.to_numpy(), tr.articoli.to_numpy())
            y_te, s_te = t.label.reindex(te.domain).to_numpy(), t.score.reindex(te.domain).to_numpy()
            riga = {"configurazione": config, "fold": k}
            if config == "5 classi":
                w = len(y_tr) / (5 * np.bincount(y_tr, minlength=5))
                a = eps * w / 5 + np.diag((1 - eps) * w)          # riga c: il peso di −log p_j per un articolo di livello c
                minimo = -(a * np.log(a / a.sum(1, keepdims=True))).sum(1)
                riga.update({f"epoca {e}": l for e, l in zip(curva.epoca, curva.loss)})
                P = np.clip(te[CLASSI].to_numpy(), 1e-7, 1)
                riga["test, dopo l'ultima epoca"] = _media_lotti((a[y_te] * -np.log(P)).sum(1), w[y_te], rng)
                riga["minimo possibile (training a memoria)"] = _media_lotti(minimo[y_tr], w[y_tr], rng)
                riga["a caso (1/5 a ogni livello)"] = _media_lotti(a.sum(1)[y_tr] * np.log(5), w[y_tr], rng)
            else:
                riga.update({f"epoca {e}": np.sqrt(l) * 100 for e, l in zip(curva.epoca, curva.loss)})
                riga["test, dopo l'ultima epoca"] = float(np.sqrt(((te.s_regressione - s_te) ** 2).mean()))
                riga["sempre la media del training"] = float(np.sqrt(((s_te - np.repeat(tr.score.to_numpy(), tr.articoli.to_numpy()).mean()) ** 2).mean()))
            righe.append(riga)
    return pd.DataFrame(righe).set_index(["configurazione", "fold"])


def misure_per_fold(T):
    """F1 sui 5 livelli e accuratezza affidabile sì/no per fold (dict nome → previsioni per testata)."""
    from sklearn.metrics import f1_score
    righe = []
    for nome, t_ in T.items():
        for imp, livelli, affidabile in [("5 classi", "5 classi", None), ("regressione", "regressione → quote del training", "regressione")]:
            for k, g in t_.groupby("fold"):
                y = g["livello vero"].to_numpy(); l = g[livelli].to_numpy().astype(int)
                a = (g[affidabile] >= 60).to_numpy() if affidabile else l >= 2
                righe.append({"modello": nome, "configurazione": imp, "fold": k,
                              "F1 5 livelli": f1_score(y, l, average="macro", labels=range(5), zero_division=0),
                              "affidabile sì/no": ((y >= 2) == a).mean()})
    return pd.DataFrame(righe)


def grafico_loss_epoche(L, per_fold, diagnostica, file):
    """Due righe (5 classi, regressione). A sinistra la loss di training epoca per epoca (una linea per fold) e, dopo
    l'ultima epoca, la stessa loss sul fold di test; al centro e a destra il test per fold, ModernBERT contro TF-IDF;
    in alto a destra la diagnostica sul fold 2 (il test misurato dopo ogni epoca)."""
    import matplotlib.pyplot as plt
    BLU, ARANCIO, GRIGIO, CHIARO = "#2a78d6", "#eb6834", "#52514e", "#b5b3ab"
    with plt.rc_context(STILE_GRAFICI):
        fig, assi = plt.subplots(2, 3, figsize=(17, 8.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
        for r, config in enumerate(["5 classi", "regressione"]):
            ax = assi[r, 0]; d = L.loc[config]
            epoche = [c for c in d.columns if c.startswith("epoca")]; x = np.arange(1, len(epoche) + 1)
            for k, riga in d.iterrows():
                ax.plot(x, riga[epoche].to_numpy(float), color=BLU, alpha=0.55, lw=1.3, marker="o", ms=3.5)
            ax.plot([], [], color=BLU, marker="o", label="training, un fold per linea")
            xt = len(epoche) + 0.8
            ax.scatter(np.full(len(d), xt) + np.linspace(-0.12, 0.12, len(d)), d["test, dopo l'ultima epoca"], color=ARANCIO, s=34,
                       zorder=3, label="test (articoli delle testate mai viste)")
            test = d["test, dopo l'ultima epoca"]
            ax.text(xt, test.max() * 1.03, f"media\n{test.mean():.2f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=8, color=ARANCIO, fontweight="bold")
            for nome in [c for c in d.columns if not c.startswith(("epoca", "test"))]:
                v = d[nome].mean(); ax.axhline(v, color=GRIGIO, ls=":", lw=1)
                ax.text(0.85, v, f"{nome}: {v:.2f}".replace(".", ","), va="bottom", fontsize=8, color=GRIGIO)
            ax.set_xticks(list(x) + [xt], [str(e) for e in x] + ["test"]); ax.set_xlabel("epoca di addestramento")
            ax.set_xlim(0.7, xt + 0.4); ax.legend(loc="center right", fontsize=8)
            if config == "5 classi":
                ax.set_title("5 classi · loss dopo ogni epoca (entropia incrociata)"); ax.set_ylim(0, d.max().max() * 1.2)
            else:
                ax.set_title("regressione · errore medio dopo ogni epoca (punti NewsGuard, RMSE)"); ax.set_ylim(0, d.max().max() * 1.2)
            for c, metrica in enumerate(["F1 5 livelli", "affidabile sì/no"], start=1):
                if config == "5 classi" and c == 2:
                    continue
                ax = assi[r, c]; p = per_fold[per_fold.configurazione == config]
                for dx, nome, colore in [(-0.12, "TF-IDF", GRIGIO), (0.12, "ModernBERT 2048", ARANCIO)]:
                    q = p[p.modello == nome]
                    ax.scatter(q.fold + dx, q[metrica], color=colore, s=40, zorder=3, label=nome.replace(" 2048", ""))
                    ax.axhline(q[metrica].mean(), color=colore, lw=1, ls="--", alpha=0.7)
                ax.set_xticks(range(1, 6), [f"fold {k}" for k in range(1, 6)]); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="lower right")
                ax.set_title(f"{config} · test per fold: {metrica}\n(-- media dei 5 fold)")
        ax = assi[0, 2]
        if diagnostica is not None:
            ax.plot(diagnostica.epoca, diagnostica.test, marker="o", color=ARANCIO, label="test (fold 2)")
            ax.plot(diagnostica.epoca, diagnostica.validazione, marker="o", color=CHIARO, ls="--", label="validazione (15% del training)")
            tf = per_fold[(per_fold.modello == "TF-IDF") & (per_fold.configurazione == "5 classi") & (per_fold.fold == 2)]["F1 5 livelli"]
            if len(tf):
                ax.axhline(tf.iloc[0], color=GRIGIO, ls=":", lw=1); ax.text(1.6, tf.iloc[0], f"TF-IDF sul fold 2: {tf.iloc[0]:.2f}".replace(".", ","),
                                                                      va="bottom", fontsize=8, color=GRIGIO)
            ax.set_ylim(0, 0.6); ax.set_xlabel("epoca di addestramento"); ax.legend(fontsize=8, loc="lower right")
            ax.set_title("5 classi · diagnostica: F1 sul test dopo ogni epoca\n(fold 2, 6 epoche, prima dei 5 fold)")
        else:
            ax.axis("off")
        fig.suptitle("ModernBERT epoca per epoca: impara il training quasi a memoria, ma sulle testate nuove non va oltre il TF-IDF",
                     fontsize=12.5, fontweight="bold", x=0.01, ha="left")
        plt.tight_layout(); plt.savefig(file, dpi=180, bbox_inches="tight"); plt.show()


# ===================================================================================== 15. analisi degli embedding (tentativo 16, notebook 7)
GRIGLIE = {"5 classi": (0.001, 0.01, 0.1, 1, 10), "ordinale": (0.001, 0.01, 0.1, 1, 10),   # C della logistica
           "regressione": (1, 10, 100, 1_000, 10_000)}                                      # alpha della ridge


def media_per_testata(E, art):
    """Un vettore per testata: la media dei vettori dei suoi articoli. Ritorna i vettori e una riga per testata."""
    righe = art.groupby("domain", sort=True).agg(label=("label", "first"), score=("score", "first"), fold=("fold", "first")).reset_index()
    pos = pd.Series(np.arange(len(righe)), index=righe.domain)
    M = np.zeros((len(righe), E.shape[1]), dtype=np.float32)
    np.add.at(M, pos[art.domain].to_numpy(), E)
    return M / art.domain.value_counts().reindex(righe.domain).to_numpy()[:, None], righe


def _adatta(tipo, valore, A, y, s):
    """Un modello lineare su righe già standardizzate; ritorna la funzione che prevede: probabilità dei 5 livelli
    (5 classi; ordinale, con 4 logistiche «livello ≥ k» alla Frank e Hall 2001) o il punteggio (regressione)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if tipo == "regressione":
        m = Ridge(alpha=valore).fit(A, s)
        return lambda B: m.predict(B)
    if tipo == "5 classi":
        m = LogisticRegression(C=valore, max_iter=3000).fit(A, y, sample_weight=pesi_bilanciati(y, np.ones(len(y))))

        def prob(B):
            P = np.zeros((B.shape[0], 5)); P[:, m.classes_] = m.predict_proba(B); return P
        return prob
    soglie = [LogisticRegression(C=valore, max_iter=3000).fit(A, y >= k, sample_weight=pesi_bilanciati(y >= k, np.ones(len(y))))
              for k in range(1, 5)]

    def prob_ordinale(B):
        sopra = np.column_stack([np.ones(B.shape[0])] + [m.predict_proba(B)[:, 1] for m in soglie] + [np.zeros(B.shape[0])])
        sopra = np.minimum.accumulate(sopra, axis=1)                       # P(livello ≥ k) non può crescere con k
        return sopra[:, :-1] - sopra[:, 1:]
    return prob_ordinale


def _misura_interna(tipo, previsione, dom, y, s):
    """Il criterio della scelta, sulle testate di validazione: F1 sui 5 livelli (voto degli articoli) o correlazione di
    Spearman tra punteggio vero e medio previsto."""
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    if tipo == "regressione":
        g = pd.DataFrame({"d": dom, "s": s, "p": previsione}).groupby("d").mean()
        return spearmanr(g.s, g.p).correlation
    g = pd.DataFrame({"d": dom, "y": y, "p": previsione.argmax(1)}).groupby("d").agg(y=("y", "first"), p=("p", voto))
    return f1_score(g.y, g.p, average="macro", labels=range(5), zero_division=0)


def _lavoro(X, y, s, dom, tr, te, tipo, valore, misura):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X[tr])
    prevedi = _adatta(tipo, valore, sc.transform(X[tr]), y[tr], s[tr])
    p = prevedi(sc.transform(X[te]))
    return _misura_interna(tipo, p, dom[te], y[te], s[te]) if misura else p


def modelli_scelti(X, righe, tipi=("5 classi", "regressione"), interni=3, n_jobs=32):
    """Modelli lineari con il parametro scelto dentro il training. `righe` (una per riga di X: articoli o testate) ha
    domain, label, score, fold. Per ogni fold esterno: una cross-validation interna sulle testate di training (`interni`
    fold stratificati per livello, le testate mai divise) prova ogni valore della griglia; si tiene il valore con la
    misura media migliore e si riaddestra su tutto il training. Il fold di test non entra mai nella scelta.
    Ritorna le previsioni per le righe di test (CLASSI o CLASSI dell'ordinale, s_regressione) e la tabella della scelta."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedGroupKFold
    y, s, dom, fold = righe.label.to_numpy(), righe.score.to_numpy(), righe.domain.to_numpy(), righe.fold.to_numpy()
    lavori = []
    for k in range(1, 6):
        tr = np.where(fold != k)[0]
        for j, (a, b) in enumerate(StratifiedGroupKFold(interni, shuffle=True, random_state=SEED).split(tr, y[tr], dom[tr])):
            lavori += [(k, j, tipo, v, tr[a], tr[b]) for tipo in tipi for v in GRIGLIE[tipo]]
    misure = Parallel(n_jobs=n_jobs)(delayed(_lavoro)(X, y, s, dom, a, b, tipo, v, True) for k, j, tipo, v, a, b in lavori)
    griglia = pd.DataFrame([(k, j, tipo, v, m) for (k, j, tipo, v, _, _), m in zip(lavori, misure)],
                           columns=["fold", "interno", "tipo", "valore", "misura"])
    media = griglia.groupby(["fold", "tipo", "valore"]).misura.mean().reset_index()
    scelte = media.loc[media.groupby(["fold", "tipo"]).misura.idxmax()].set_index(["fold", "tipo"])
    finali = [(k, tipo, scelte.loc[(k, tipo), "valore"]) for k in range(1, 6) for tipo in tipi]
    prev = Parallel(n_jobs=n_jobs)(delayed(_lavoro)(X, y, s, dom, np.where(fold != k)[0], np.where(fold == k)[0], tipo, v, False)
                                   for k, tipo, v in finali)
    pezzi = []
    for k in range(1, 6):
        te = np.where(fold == k)[0]
        out = righe.iloc[te][["domain"] + (["article_id"] if "article_id" in righe else [])].reset_index(drop=True)
        for (kk, tipo, _), p in zip(finali, prev):
            if kk != k:
                continue
            if tipo == "regressione":
                out["s_regressione"] = p
            else:
                out[[c if tipo == "5 classi" else f"ordinale_{c}" for c in CLASSI]] = p
        pezzi.append(out)
    return pd.concat(pezzi, ignore_index=True), media


def testate_da_previsioni(pa, t, ordinale=False):
    """Come `testate_da_articoli`, anche per previsioni già per testata (una riga per testata) e per l'ordinale."""
    if ordinale:
        pa = pa.drop(columns=[c for c in CLASSI + ["s_regressione"] if c in pa]).rename(columns={f"ordinale_{c}": c for c in CLASSI})
    return testate_da_articoli(pa, t)


def grafico_griglia(medie, titolo, file):
    """La scelta del parametro: per ogni tipo di modello la misura media della cross-validation interna per ogni valore
    della griglia, una linea per fold esterno."""
    import matplotlib.pyplot as plt
    tipi = list(dict.fromkeys(medie.tipo))
    with plt.rc_context(STILE_GRAFICI):
        fig, assi = plt.subplots(1, len(tipi), figsize=(4.6 * len(tipi), 3.2), squeeze=False)
        for ax, tipo in zip(assi[0], tipi):
            m = medie[medie.tipo == tipo]
            for k, g in m.groupby("fold"):
                ax.plot(g.valore, g.misura, marker="o", lw=1.2, color="#2a78d6", alpha=0.6)
                b = g.loc[g.misura.idxmax()]; ax.scatter(b.valore, b.misura, s=60, color="#eb6834", zorder=3)
            ax.set_xscale("log"); ax.set_xlabel("alpha della ridge" if tipo == "regressione" else "C della logistica")
            ax.set_title(f"{tipo}: " + ("Spearman" if tipo == "regressione" else "F1 5 livelli") + " sulla validazione interna")
        fig.suptitle(titolo + " · una linea per fold · ● il valore scelto", fontsize=11, fontweight="bold", x=0.01, ha="left")
        plt.tight_layout(); plt.savefig(file, dpi=160, bbox_inches="tight"); plt.show()


# ===================================================================================== 16. un LLM grande come lettore (tentativo 16, notebook 8)
def embedding_llm(testi, modello, max_token, batch, strati, registra, file_parziale=None, ogni=200):
    """Un vettore per articolo da un LLM generativo grande, caricato in 4 bit (NF4, calcoli in fp16: la V100 non ha
    bf16): la media degli stati nascosti dei token dell'articolo (senza padding), per ognuno degli `strati` (indici
    di hidden_states: 0 = embedding dei token, n = ultimo strato). Nessuna istruzione: il modello legge solo il testo.
    Gli articoli vanno dal più corto al più lungo; con `file_parziale` ogni `ogni` lotti si salva il lavoro fatto e,
    se il calcolo si interrompe, si riparte da lì. Ritorna {strato: matrice articoli × dimensione}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(modello)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(modello, quantization_config=quant, torch_dtype=torch.float16, device_map={"": 0}).eval()
    ordine = np.argsort([len(x) for x in testi])
    E, fatti = None, np.zeros(len(testi), bool)
    if file_parziale is not None and Path(file_parziale).exists():
        p = np.load(file_parziale, allow_pickle=True).item()
        if p["n"] == len(testi):
            E, fatti = p["E"], p["fatti"]; registra(f"riprendo: {fatti.sum():,} articoli già fatti")
    t0 = time.time()
    with torch.no_grad():
        for b, k in enumerate(range(0, len(ordine), batch)):
            idx = ordine[k:k + batch]
            if fatti[idx].all():
                continue
            enc = tok([testi[i] for i in idx], padding=True, truncation=True, max_length=max_token, return_tensors="pt").to("cuda")
            out = m.model(**enc, output_hidden_states=True)                  # solo il trasformatore, senza la testa
            maschera = enc["attention_mask"].unsqueeze(-1).float()
            if E is None:
                E = {s: np.zeros((len(testi), out.hidden_states[s].shape[-1]), dtype=np.float32) for s in strati}
            for s in strati:
                h = out.hidden_states[s].float()
                E[s][idx] = ((h * maschera).sum(1) / maschera.sum(1)).cpu().numpy()
            fatti[idx] = True
            del out
            if b % ogni == 0:
                registra(f"{fatti.sum():,}/{len(testi):,} articoli ({time.time() - t0:.0f} s)")
                if file_parziale is not None:
                    np.save(file_parziale, {"n": len(testi), "E": E, "fatti": fatti}, allow_pickle=True)
    for s in strati:
        if not np.isfinite(E[s]).all():
            registra(f"ATTENZIONE: strato {s}, {(~np.isfinite(E[s]).all(1)).sum()} articoli con valori non finiti")
    del m; torch.cuda.empty_cache()
    return E


# ===================================================================================== 17. la testata decisa dal voto dei suoi articoli (tentativo 16, notebook 7)
def testate_voto(pa, t):
    """Una riga per testata, come `testate_da_articoli`, ma ogni articolo dà un livello e la testata prende quello votato
    dalla maggioranza dei suoi articoli (a parità il più basso), anche per la regressione; "affidabile" è il voto degli
    articoli con un livello da l3 in su. Classificatore: il livello più probabile di ogni articolo. Regressione: in ogni
    fold i punteggi previsti degli articoli di test si ordinano e si tagliano con le quote di livelli degli articoli di
    training (come le quote del training per testata, ma articolo per articolo)."""
    pa = pa.assign(fold=t.fold.reindex(pa.domain).to_numpy())
    T = pd.DataFrame({"livello vero": t.label, "punteggio vero": t.score, "fold": t.fold})
    vota = lambda livelli: pd.Series(livelli, index=pa.index).groupby(pa.domain).agg(voto)
    if set(CLASSI) <= set(pa.columns):
        lv = pa[CLASSI].to_numpy().argmax(1)
        T["5 classi"] = vota(lv); T["5 classi: affidabile"] = vota((lv >= 2).astype(int)).astype(bool)
    if "s_regressione" in pa:
        etichetta = t.label.reindex(pa.domain).to_numpy()
        lv = np.zeros(len(pa), dtype=int)
        for k in sorted(pa.fold.unique()):
            te = (pa.fold == k).to_numpy()
            quote = np.cumsum(np.bincount(etichetta[~te], minlength=5))[:-1] / (~te).sum()
            sc = pa.s_regressione.to_numpy()[te]
            lv[te] = np.searchsorted(np.quantile(sc, quote), sc)
        T["regressione"] = pa.groupby("domain").s_regressione.mean()
        T["regressione → quote del training"] = vota(lv); T["regressione: affidabile"] = vota((lv >= 2).astype(int)).astype(bool)
    return T.reindex(t.index)
