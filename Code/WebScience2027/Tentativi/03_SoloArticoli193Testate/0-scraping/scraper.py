"""Nuovo scraping per Web Science 2027: più testate, più articoli.

Testate: tutte quelle di NewsGuard con argomento "Political news or commentary", lingua inglese e un
punteggio, in ordine casuale (seed 42). Nessuna scelta dipende dai livelli NewsGuard.

Per ogni testata (dominio):
  1. scarica la homepage;
  2. trova gli articoli da tre fonti: feed RSS/Atom, sitemap, link della homepage;
  3. li ordina dal più recente (prima i feed, poi la sitemap per data, poi la homepage);
  4. scarica gli articoli finché ne ha MAX_ARTICOLI con più di MIN_PAROLE parole (testo, titolo, data,
     autore estratti con trafilatura); salva anche quelli più corti, il filtro si applica dopo;
  5. archivia l'HTML grezzo di ogni articolo, per poter rifare l'estrazione senza rifare lo scraping.
Rispetta robots.txt e fa al massimo una richiesta al secondo per sito.

Robustezza:
  - ogni testata gira in un processo separato; un processo che supera il tempo massimo viene ucciso;
  - ogni testata finita lascia un file in stato/: se lo script viene rilanciato, le testate finite si saltano;
  - nessuna eccezione di una testata ferma le altre;
  - il file avvia.sh rilancia lo script se il processo principale dovesse morire.

Output (cartella --out, default ../../../../../Dataset/scraping/NuovoScraping):
  articoli/<dominio>.jsonl     un articolo per riga
  html/<dominio>.jsonl.gz      l'HTML grezzo degli articoli salvati
  config.json                  parametri e versioni delle librerie
  stato/<dominio>.json         esito della testata (numeri, tempi, errori)
  elenco_testate.csv           le testate scelte e l'ordine
  stato_generale.json          riassunto aggiornato a ogni testata finita
  scraping.log                 il log
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import signal
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import feedparser
import pandas as pd
import requests
import trafilatura
import urllib3
from lxml import html as lxml_html

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

QUI = Path(__file__).resolve().parent
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
MAX_ARTICOLI = 200            # articoli salvati per testata (come nello scraping originale)
MAX_TENTATIVI = 500           # URL provati per testata
MIN_PAROLE = 100              # come nello scraping originale
PAUSA = 1.0                   # secondi tra due richieste alla stessa testata
TIMEOUT = (10, 20)            # connessione, lettura
MAX_BYTE = 5_000_000          # pagine più grandi vengono scartate
TEMPO_TESTATA = 40 * 60       # dopo 40 minuti la testata si chiude con quello che ha
TEMPO_KILL = TEMPO_TESTATA + 5 * 60
MAX_SITEMAP = 25              # sitemap lette al massimo per testata

NON_ARTICOLO = re.compile(
    r"/(tag|tags|category|categories|author|authors|page|feed|wp-content|wp-json|search|login|subscribe|"
    r"newsletter|account|cart|shop|events?|video|videos|gallery|galleries|podcasts?|topics?)(/|$)|"
    r"\.(jpg|jpeg|png|gif|webp|svg|pdf|mp3|mp4|zip|xml|css|js)(\?|$)", re.I)


# --------------------------------------------------------------------------- utilità

def ora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scrivi_json(percorso, dati):
    """Scrittura atomica: prima un file temporaneo, poi rinomina."""
    tmp = Path(str(percorso) + ".tmp")
    tmp.write_text(json.dumps(dati, ensure_ascii=False, indent=1))
    os.replace(tmp, percorso)


class Tempo(Exception):
    pass


def _allarme(signum, frame):
    raise Tempo()


class Client:
    """Richieste HTTP verso una sola testata, con pausa, timeout e limite di dimensione."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
        self.ultima = 0.0
        self.richieste = 0
        self.ultimo_codice = None

    def get(self, url, riprova_429=True):
        attesa = PAUSA - (time.time() - self.ultima)
        if attesa > 0:
            time.sleep(attesa)
        self.ultima = time.time()
        self.richieste += 1
        for verifica in (True, False):
            try:
                r = self.s.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True, verify=verifica)
                contenuto, inizio = b"", time.time()
                for pezzo in r.iter_content(65536):
                    contenuto += pezzo
                    if len(contenuto) > MAX_BYTE or time.time() - inizio > 60:
                        r.close()
                        return None
                self.ultimo_codice = r.status_code
                if r.status_code == 429 and riprova_429:   # troppe richieste: una sola attesa e un nuovo tentativo
                    time.sleep(30)
                    return self.get(url, riprova_429=False)
                if r.status_code != 200:
                    return None
                r._content = contenuto
                return r
            except requests.exceptions.SSLError:
                self.ultimo_codice = "SSL"
                continue                       # riprova senza verificare il certificato
            except Exception as e:
                self.ultimo_codice = type(e).__name__
                return None
        return None


def stesso_sito(url, dominio):
    host = urlparse(url).netloc.lower().split(":")[0]
    d = dominio.lower()
    return host == d or host.endswith("." + d) or ("www." + d) == host


# --------------------------------------------------------------------------- scoperta degli URL

def link_homepage(html_text, base):
    try:
        albero = lxml_html.fromstring(html_text)
    except Exception:
        return [], []
    link, feed = [], []
    for a in albero.xpath("//a[@href]"):
        link.append((urljoin(base, a.get("href").strip()), (a.text_content() or "").strip().lower()))
    for l in albero.xpath("//link[@rel='alternate'][@href]"):
        if any(t in (l.get("type") or "") for t in ("rss", "atom", "xml")):
            feed.append(urljoin(base, l.get("href")))
    return link, feed


def sembra_articolo(url, dominio):
    if not stesso_sito(url, dominio):
        return False
    p = urlparse(url)
    if p.query and "p=" not in p.query:
        return False
    percorso = p.path.rstrip("/")
    if not percorso or NON_ARTICOLO.search(percorso):
        return False
    pezzi = [x for x in percorso.split("/") if x]
    ultimo = pezzi[-1] if pezzi else ""
    return ultimo.count("-") >= 3 or bool(re.search(r"/20\d\d/", percorso)) or (len(pezzi) >= 2 and len(ultimo) > 25)


def da_feed(client, url_feed):
    r = client.get(url_feed)
    if r is None:
        return []
    try:
        f = feedparser.parse(r.content)
    except Exception:
        return []
    out = []
    for e in f.entries:
        if e.get("link"):
            data = e.get("published") or e.get("updated") or ""
            out.append((e.link, data))
    return out


def leggi_sitemap(client, url):
    r = client.get(url)
    if r is None:
        return [], []
    contenuto = r.content
    if url.endswith(".gz") or contenuto[:2] == b"\x1f\x8b":
        try:
            contenuto = gzip.decompress(contenuto)
        except Exception:
            return [], []
    try:
        radice = ET.fromstring(contenuto)
    except Exception:
        return [], []
    ns = lambda tag: tag.split("}")[-1]
    figlie, url_pagine = [], []
    for el in radice:
        loc = lastmod = None
        for c in el:
            if ns(c.tag) == "loc":
                loc = (c.text or "").strip()
            elif ns(c.tag) == "lastmod":
                lastmod = (c.text or "").strip()
            elif ns(c.tag) == "news":
                for n in c:
                    if ns(n.tag) == "publication_date":
                        lastmod = lastmod or (n.text or "").strip()
        if not loc:
            continue
        (figlie if ns(radice.tag) == "sitemapindex" else url_pagine).append((loc, lastmod or ""))
    return figlie, url_pagine


def da_sitemap(client, home, robots_txt, scadenza):
    candidate = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots_txt or "")
    for percorso in ("/sitemap.xml", "/sitemap_index.xml", "/news-sitemap.xml", "/sitemap-news.xml",
                     "/post-sitemap.xml", "/wp-sitemap.xml", "/sitemap.xml.gz"):
        candidate.append(urljoin(home, percorso))
    visti, pagine, lette = set(), [], 0
    coda = list(dict.fromkeys(candidate))
    while coda and lette < MAX_SITEMAP and time.time() < scadenza:
        url = coda.pop(0)
        if url in visti:
            continue
        visti.add(url)
        figlie, url_pagine = leggi_sitemap(client, url)
        lette += 1
        pagine.extend(url_pagine)
        # prima le sitemap di articoli/notizie, poi le più recenti
        figlie.sort(key=lambda x: (any(k in x[0].lower() for k in ("post", "news", "article")), x[1]), reverse=True)
        coda = [f[0] for f in figlie[:MAX_SITEMAP]] + coda
        if len(pagine) > 5000:
            break
    return pagine


# --------------------------------------------------------------------------- estrazione

def estrai(html_text, url, recall=False):
    try:
        j = trafilatura.extract(html_text, url=url, output_format="json", with_metadata=True,
                                include_comments=False, include_tables=False, favor_recall=recall)
        return json.loads(j) if j else None
    except Exception:
        return None


# --------------------------------------------------------------------------- una testata

def scarica_testata(dominio, cartella):
    """Gira in un processo separato. Scrive gli articoli e lo stato della testata."""
    signal.signal(signal.SIGALRM, _allarme)
    signal.alarm(TEMPO_TESTATA)
    inizio = time.time()
    scadenza = inizio + TEMPO_TESTATA - 60
    stato = {"dominio": dominio, "esito": "in corso", "articoli": 0, "url_provati": 0,
             "fonti": {}, "inizio": ora()}
    file_art = cartella / "articoli" / f"{dominio}.jsonl"
    archivio = gzip.open(cartella / "html" / f"{dominio}.jsonl.gz", "at")
    archivia = lambda tipo, url, testo_html: archivio.write(
        json.dumps({"tipo": tipo, "url": url, "html": testo_html, "scaricato": ora()}, ensure_ascii=False) + "\n")
    client = Client()
    try:
        # 1. homepage
        home, r = None, None
        for prova in (f"https://{dominio}/", f"https://www.{dominio}/", f"http://{dominio}/"):
            r = client.get(prova)
            if r is not None:
                home = r.url
                break
        if r is None:
            stato["esito"] = f"homepage irraggiungibile ({client.ultimo_codice})"
            return stato
        stato["homepage"] = home
        host_home = urlparse(home).netloc.lower().split(":")[0]
        if host_home.startswith("www."):
            host_home = host_home[4:]
        domini = {dominio, host_home}                  # il sito può reindirizzare a un altro dominio
        link, feed = link_homepage(r.text, home)

        # robots.txt
        rp = robotparser.RobotFileParser()
        rr = client.get(urljoin(home, "/robots.txt"))
        robots_txt = rr.text if rr is not None else ""
        rp.parse(robots_txt.splitlines())

        # 2. URL degli articoli: feed, sitemap, homepage
        for f in ("/feed", "/rss", "/feed/", "/rss.xml", "/index.xml", "/atom.xml"):
            feed.append(urljoin(home, f))
        candidati = []
        for f in list(dict.fromkeys(feed))[:6]:
            if time.time() > scadenza:
                break
            candidati += [(u, d, "feed") for u, d in da_feed(client, f)]
        stato["fonti"]["feed"] = len(candidati)
        dalla_sitemap = sorted(da_sitemap(client, home, robots_txt, scadenza), key=lambda x: x[1], reverse=True)
        candidati += [(u, d, "sitemap") for u, d in dalla_sitemap]
        stato["fonti"]["sitemap"] = len(dalla_sitemap)
        candidati += [(u, "", "homepage") for u, _ in link]
        visti, ordinati = set(), []
        for u, d, fonte in candidati:
            u = u.split("#")[0]
            if u not in visti and any(sembra_articolo(u, x) for x in domini):
                visti.add(u)
                ordinati.append((u, d, fonte))
        stato["fonti"]["candidati"] = len(ordinati)

        # 3. articoli
        hash_visti = set()
        with open(file_art, "w") as fa:
            for url, data_lista, fonte in ordinati:
                if stato["articoli"] >= MAX_ARTICOLI or stato["url_provati"] >= MAX_TENTATIVI or time.time() > scadenza:
                    break
                try:
                    if not rp.can_fetch(UA, url):
                        continue
                except Exception:
                    pass
                stato["url_provati"] += 1
                p = client.get(url)
                if p is None:
                    continue
                dati = estrai(p.text, p.url)
                if not dati or not (dati.get("text") or "").strip():
                    continue
                h = hashlib.md5(dati["text"].encode()).hexdigest()
                if h in hash_visti:
                    continue
                hash_visti.add(h)
                parole = len(dati["text"].split())
                archivia("articolo", p.url, p.text)
                fa.write(json.dumps({"dominio": dominio, "url": p.url, "fonte": fonte, "data_lista": data_lista,
                                     "titolo": dati.get("title"), "data": dati.get("date"), "autore": dati.get("author"),
                                     "parole": parole, "testo": dati["text"], "scaricato": ora()}, ensure_ascii=False) + "\n")
                fa.flush()
                stato["testi_salvati"] = stato.get("testi_salvati", 0) + 1
                if parole > MIN_PAROLE:
                    stato["articoli"] += 1
        stato["esito"] = "completata" if stato["articoli"] >= MAX_ARTICOLI else "esauriti gli URL o i tentativi"
    except Tempo:
        stato["esito"] = "tempo massimo raggiunto"
    except Exception as e:
        stato["esito"] = f"errore: {type(e).__name__}"
        stato["traccia"] = traceback.format_exc()[-2000:]
    finally:
        signal.alarm(0)
        try:
            archivio.close()
        except Exception:
            pass
        stato["richieste"] = client.richieste
        stato["minuti"] = round((time.time() - inizio) / 60, 1)
        stato["fine"] = ora()
        scrivi_json(cartella / "stato" / f"{dominio}.json", stato)
    return stato


def processo_testata(dominio, cartella):
    try:
        scarica_testata(dominio, cartella)
    except BaseException:
        pass


# --------------------------------------------------------------------------- elenco delle testate

def scegli_testate(newsguard, seed):
    ng = pd.read_csv(newsguard, low_memory=False)
    ng = ng[(ng.Topics == "Political news or commentary") & (ng.Language == "en")].drop_duplicates("Domain")
    ng = ng[~ng.Domain.str.contains("/", na=True) & ng.Score.notna()]          # solo testate con un punteggio
    ng = ng.sample(frac=1, random_state=seed)                          # ordine casuale, riproducibile
    return ng[["Domain"]].rename(columns={"Domain": "dominio"})


# --------------------------------------------------------------------------- principale

def main():
    global MAX_ARTICOLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--newsguard", default=str(QUI.parents[4] / "Dataset" / "etichette/newsguard_2022.csv"))
    ap.add_argument("--out", default=str(QUI.parents[4] / "Dataset" / "scraping/NuovoScraping"))
    ap.add_argument("--processi", type=int, default=32)
    ap.add_argument("--solo", type=int, default=0, help="solo le prime N testate (per le prove)")
    ap.add_argument("--domini", nargs="*", default=None, help="solo questi domini (per le prove)")
    ap.add_argument("--max-articoli", type=int, default=MAX_ARTICOLI)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cartella = Path(args.out)
    for sotto in ("articoli", "stato", "html"):
        (cartella / sotto).mkdir(parents=True, exist_ok=True)
    log = open(cartella / "scraping.log", "a", buffering=1)
    scrivi = lambda m: (log.write(f"{ora()}  {m}\n"), print(f"{ora()}  {m}", flush=True))

    MAX_ARTICOLI = args.max_articoli
    elenco = scegli_testate(args.newsguard, args.seed)
    if args.domini:
        elenco = elenco[elenco.dominio.isin(args.domini)]
    if args.solo:
        elenco = elenco.head(args.solo)
    elenco.to_csv(cartella / "elenco_testate.csv", index=False)
    if not (cartella / "config.json").exists():
        import platform, importlib.metadata as im
        scrivi_json(cartella / "config.json", {
            "avvio": ora(), "testate": len(elenco), "seed": args.seed, "processi": args.processi,
            "parametri": {k: globals()[k] for k in ("MAX_ARTICOLI", "MAX_TENTATIVI", "MIN_PAROLE", "PAUSA", "TIMEOUT",
                                                     "MAX_BYTE", "TEMPO_TESTATA", "MAX_SITEMAP", "UA")},
            "versioni": {"python": platform.python_version(),
                         **{m: im.version(m) for m in ("trafilatura", "requests", "feedparser", "lxml", "pandas")}}})

    fatte = {p.stem for p in (cartella / "stato").glob("*.json")}
    da_fare = [d for d in elenco.dominio if d not in fatte]
    totale = len(elenco)
    scrivi(f"=== avvio: {totale} testate, {len(fatte)} già fatte, {len(da_fare)} da fare, {args.processi} processi ===")

    ctx = mp.get_context("fork")
    in_corso = {}
    inizio = time.time()
    finite_ora = 0
    while da_fare or in_corso:
        while da_fare and len(in_corso) < args.processi:
            d = da_fare.pop(0)
            p = ctx.Process(target=processo_testata, args=(d, cartella), daemon=True)
            p.start()
            in_corso[d] = (p, time.time())
        for d, (p, t0) in list(in_corso.items()):
            if not p.is_alive():
                p.join()
            elif time.time() - t0 > TEMPO_KILL:
                p.terminate()
                p.join(10)
                if p.is_alive():
                    p.kill()
                    p.join(5)
                file_stato = cartella / "stato" / f"{d}.json"
                if not file_stato.exists():
                    righe = sum(1 for _ in open(cartella / "articoli" / f"{d}.jsonl")) if (cartella / "articoli" / f"{d}.jsonl").exists() else 0
                    scrivi_json(file_stato, {"dominio": d, "esito": "processo ucciso", "articoli": righe, "fine": ora()})
            else:
                continue
            del in_corso[d]
            finite_ora += 1
            file_stato = cartella / "stato" / f"{d}.json"
            if not file_stato.exists():                     # processo morto senza scrivere lo stato
                scrivi_json(file_stato, {"dominio": d, "esito": "processo terminato senza stato", "articoli": 0, "fine": ora()})
            s = json.loads(file_stato.read_text())
            n_fatte = len(list((cartella / "stato").glob("*.json")))
            stati = [json.loads(x.read_text()) for x in (cartella / "stato").glob("*.json")]
            tot_art = sum(x.get("articoli", 0) for x in stati)
            ritmo = (time.time() - inizio) / finite_ora
            mancano = (totale - n_fatte) * ritmo / 3600
            scrivi(f"[{n_fatte}/{totale}] {d}: {s.get('esito')}, {s.get('articoli', 0)} articoli, "
                   f"{s.get('minuti', '?')} min | totale articoli {tot_art} | "
                   f"fine stimata tra {mancano:.1f} ore")
            scrivi_json(cartella / "stato_generale.json", {
                "aggiornato": ora(), "testate_finite": n_fatte, "testate_totali": totale,
                "articoli_totali": tot_art, "in_corso": sorted(in_corso),
                "esiti": pd.Series([x.get("esito") for x in stati]).value_counts().to_dict(),
                "ore_mancanti_stimate": round(mancano, 1)})
        time.sleep(2)
    scrivi("=== FINITO ===")


if __name__ == "__main__":
    main()
