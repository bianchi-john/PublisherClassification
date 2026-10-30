"""Step 1: the scraper (October 2026). Not re-run in this attempt: its output is already in Dataset/scraping/.

Outlets: every NewsGuard outlet whose Topics include "Political news or commentary", in English and with a score, in a
random order (seed 42). No choice depends on the NewsGuard levels.

For each outlet (domain):
  1. download the homepage;
  2. find article URLs from three sources: RSS/Atom feeds, sitemaps, homepage links;
  3. sort them newest first (feeds, then sitemaps by date, then homepage);
  4. download articles until MAX_ARTICLES have more than MIN_WORDS words (text, title, date and author extracted with
     trafilatura); shorter ones are saved too, the length filter comes later;
  5. archive the raw HTML of every article, so that the text can be re-extracted without scraping again.
It respects robots.txt and makes at most one request per second per site.

Robustness: one process per outlet, killed after a time limit; every finished outlet leaves a file in status/, so a
restart skips it; run_scraper.sh restarts the script if the main process dies.

Output (folder --out):
  articoli/<domain>.jsonl     one article per line
  html/<domain>.jsonl.gz      the raw HTML of the saved articles
  stato/<domain>.json         outcome of the outlet (counts, times, errors)
  config.json                 parameters and library versions
  elenco_testate.csv          the chosen outlets, in order
  scraping.log                the log
(The folder and field names are the ones of the original October 2026 scraping, which is in Italian.)

Usage: python scraper.py --newsguard <newsguard.csv> --out <folder> [--processes 64]
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
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

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
MAX_ARTICLES = 200            # articles saved per outlet
MAX_ATTEMPTS = 500            # URLs tried per outlet
MIN_WORDS = 100
PAUSE = 1.0                   # seconds between two requests to the same outlet
TIMEOUT = (10, 20)            # connect, read
MAX_BYTES = 5_000_000         # larger pages are discarded
OUTLET_TIME = 40 * 60         # after 40 minutes the outlet closes with what it has
KILL_TIME = OUTLET_TIME + 5 * 60
MAX_SITEMAPS = 25

NOT_ARTICLE = re.compile(
    r"/(tag|tags|category|categories|author|authors|page|feed|wp-content|wp-json|search|login|subscribe|"
    r"newsletter|account|cart|shop|events?|video|videos|gallery|galleries|podcasts?|topics?)(/|$)|"
    r"\.(jpg|jpeg|png|gif|webp|svg|pdf|mp3|mp4|zip|xml|css|js)(\?|$)", re.I)


# --------------------------------------------------------------------------- utilities
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path, data):
    """Atomic write: a temporary file first, then rename."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


class Client:
    """HTTP requests to a single outlet, with pause, timeout and size limit."""

    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                               "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
        self.last = 0.0
        self.requests = 0
        self.last_code = None

    def get(self, url, retry_429=True):
        wait = PAUSE - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        self.requests += 1
        for verify in (True, False):
            try:
                r = self.s.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True, verify=verify)
                content, start = b"", time.time()
                for chunk in r.iter_content(65536):
                    content += chunk
                    if len(content) > MAX_BYTES or time.time() - start > 60:
                        r.close()
                        return None
                self.last_code = r.status_code
                if r.status_code == 429 and retry_429:       # too many requests: wait once and retry
                    time.sleep(30)
                    return self.get(url, retry_429=False)
                if r.status_code != 200:
                    return None
                r._content = content
                return r
            except requests.exceptions.SSLError:
                self.last_code = "SSL"
                continue                                      # retry without verifying the certificate
            except Exception as e:
                self.last_code = type(e).__name__
                return None
        return None


def same_site(url, domain):
    host = urlparse(url).netloc.lower().split(":")[0]
    d = domain.lower()
    return host == d or host.endswith("." + d) or ("www." + d) == host


# --------------------------------------------------------------------------- finding article URLs
def homepage_links(html_text, base):
    try:
        tree = lxml_html.fromstring(html_text)
    except Exception:
        return [], []
    links, feeds = [], []
    for a in tree.xpath("//a[@href]"):
        links.append((urljoin(base, a.get("href").strip()), (a.text_content() or "").strip().lower()))
    for link in tree.xpath("//link[@rel='alternate'][@href]"):
        if any(t in (link.get("type") or "") for t in ("rss", "atom", "xml")):
            feeds.append(urljoin(base, link.get("href")))
    return links, feeds


def looks_like_article(url, domain):
    if not same_site(url, domain):
        return False
    p = urlparse(url)
    if p.query and "p=" not in p.query:
        return False
    path = p.path.rstrip("/")
    if not path or NOT_ARTICLE.search(path):
        return False
    parts = [x for x in path.split("/") if x]
    last = parts[-1] if parts else ""
    return last.count("-") >= 3 or bool(re.search(r"/20\d\d/", path)) or (len(parts) >= 2 and len(last) > 25)


def from_feed(client, feed_url):
    r = client.get(feed_url)
    if r is None:
        return []
    try:
        f = feedparser.parse(r.content)
    except Exception:
        return []
    return [(e.link, e.get("published") or e.get("updated") or "") for e in f.entries if e.get("link")]


def read_sitemap(client, url):
    r = client.get(url)
    if r is None:
        return [], []
    content = r.content
    if url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except Exception:
            return [], []
    try:
        root = ET.fromstring(content)
    except Exception:
        return [], []
    ns = lambda tag: tag.split("}")[-1]
    children, pages = [], []
    for el in root:
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
        if loc:
            (children if ns(root.tag) == "sitemapindex" else pages).append((loc, lastmod or ""))
    return children, pages


def from_sitemaps(client, home, robots_txt, deadline):
    candidates = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots_txt or "")
    for path in ("/sitemap.xml", "/sitemap_index.xml", "/news-sitemap.xml", "/sitemap-news.xml",
                 "/post-sitemap.xml", "/wp-sitemap.xml", "/sitemap.xml.gz"):
        candidates.append(urljoin(home, path))
    seen, pages, read = set(), [], 0
    queue = list(dict.fromkeys(candidates))
    while queue and read < MAX_SITEMAPS and time.time() < deadline:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        children, found = read_sitemap(client, url)
        read += 1
        pages.extend(found)
        # article/news sitemaps first, then the most recent
        children.sort(key=lambda x: (any(k in x[0].lower() for k in ("post", "news", "article")), x[1]), reverse=True)
        queue = [c[0] for c in children[:MAX_SITEMAPS]] + queue
        if len(pages) > 5000:
            break
    return pages


def extract(html_text, url):
    try:
        j = trafilatura.extract(html_text, url=url, output_format="json", with_metadata=True,
                                include_comments=False, include_tables=False)
        return json.loads(j) if j else None
    except Exception:
        return None


# --------------------------------------------------------------------------- one outlet
def scrape_outlet(domain, folder):
    """Runs in its own process. Writes the articles, the raw HTML and the status of the outlet."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(OUTLET_TIME)
    start = time.time()
    deadline = start + OUTLET_TIME - 60
    status = {"dominio": domain, "esito": "running", "articoli": 0, "url_provati": 0, "fonti": {}, "inizio": now()}
    archive = gzip.open(folder / "html" / f"{domain}.jsonl.gz", "at")
    client = Client()
    try:
        # 1. homepage
        home, r = None, None
        for attempt in (f"https://{domain}/", f"https://www.{domain}/", f"http://{domain}/"):
            r = client.get(attempt)
            if r is not None:
                home = r.url
                break
        if r is None:
            status["esito"] = f"homepage unreachable ({client.last_code})"
            return status
        host = urlparse(home).netloc.lower().split(":")[0].removeprefix("www.")
        domains = {domain, host}                          # the site may redirect to another domain
        links, feeds = homepage_links(r.text, home)
        rp = robotparser.RobotFileParser()
        rr = client.get(urljoin(home, "/robots.txt"))
        robots_txt = rr.text if rr is not None else ""
        rp.parse(robots_txt.splitlines())

        # 2. article URLs: feeds, sitemaps, homepage
        feeds += [urljoin(home, f) for f in ("/feed", "/rss", "/feed/", "/rss.xml", "/index.xml", "/atom.xml")]
        candidates = []
        for f in list(dict.fromkeys(feeds))[:6]:
            if time.time() > deadline:
                break
            candidates += [(u, d, "feed") for u, d in from_feed(client, f)]
        status["fonti"]["feed"] = len(candidates)
        sitemap = sorted(from_sitemaps(client, home, robots_txt, deadline), key=lambda x: x[1], reverse=True)
        candidates += [(u, d, "sitemap") for u, d in sitemap]
        status["fonti"]["sitemap"] = len(sitemap)
        candidates += [(u, "", "homepage") for u, _ in links]
        seen, ordered = set(), []
        for u, d, source in candidates:
            u = u.split("#")[0]
            if u not in seen and any(looks_like_article(u, x) for x in domains):
                seen.add(u)
                ordered.append((u, d, source))
        status["fonti"]["candidati"] = len(ordered)

        # 3. articles
        hashes = set()
        with open(folder / "articoli" / f"{domain}.jsonl", "w") as fa:
            for url, list_date, source in ordered:
                if status["articoli"] >= MAX_ARTICLES or status["url_provati"] >= MAX_ATTEMPTS or time.time() > deadline:
                    break
                try:
                    if not rp.can_fetch(UA, url):
                        continue
                except Exception:
                    pass
                status["url_provati"] += 1
                page = client.get(url)
                if page is None:
                    continue
                data = extract(page.text, page.url)
                if not data or not (data.get("text") or "").strip():
                    continue
                h = hashlib.md5(data["text"].encode()).hexdigest()
                if h in hashes:
                    continue
                hashes.add(h)
                words = len(data["text"].split())
                archive.write(json.dumps({"tipo": "articolo", "url": page.url, "html": page.text, "scaricato": now()},
                                         ensure_ascii=False) + "\n")
                fa.write(json.dumps({"dominio": domain, "url": page.url, "fonte": source, "data_lista": list_date,
                                     "titolo": data.get("title"), "data": data.get("date"), "autore": data.get("author"),
                                     "parole": words, "testo": data["text"], "scaricato": now()}, ensure_ascii=False) + "\n")
                fa.flush()
                if words > MIN_WORDS:
                    status["articoli"] += 1
        status["esito"] = "complete" if status["articoli"] >= MAX_ARTICLES else "out of URLs or attempts"
    except Timeout:
        status["esito"] = "time limit reached"
    except Exception as e:
        status["esito"] = f"error: {type(e).__name__}"
        status["traccia"] = traceback.format_exc()[-2000:]
    finally:
        signal.alarm(0)
        archive.close()
        status["richieste"] = client.requests
        status["minuti"] = round((time.time() - start) / 60, 1)
        status["fine"] = now()
        write_json(folder / "stato" / f"{domain}.json", status)
    return status


def outlet_process(domain, folder):
    try:
        scrape_outlet(domain, folder)
    except BaseException:
        pass


# --------------------------------------------------------------------------- which outlets
def choose_outlets(newsguard, seed):
    """Political, English, scored NewsGuard outlets, in a random but reproducible order."""
    ng = pd.read_csv(newsguard, low_memory=False)
    politics = ng.Topics.fillna("").str.contains("Political news or commentary", regex=False)
    ng = ng[politics & (ng.Language == "en")].drop_duplicates("Domain")
    ng = ng[~ng.Domain.str.contains("/", na=True) & ng.Score.notna()]
    return ng.sample(frac=1, random_state=seed)[["Domain"]].rename(columns={"Domain": "dominio"})


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--newsguard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    folder = Path(args.out)
    for sub in ("articoli", "stato", "html"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    logfile = open(folder / "scraping.log", "a", buffering=1)
    say = lambda m: (logfile.write(f"{now()}  {m}\n"), print(f"{now()}  {m}", flush=True))

    outlets = choose_outlets(args.newsguard, args.seed)
    outlets.to_csv(folder / "elenco_testate.csv", index=False)
    if not (folder / "config.json").exists():
        import importlib.metadata as im
        import platform
        write_json(folder / "config.json", {
            "start": now(), "outlets": len(outlets), "seed": args.seed, "processes": args.processes,
            "parameters": {k: globals()[k] for k in ("MAX_ARTICLES", "MAX_ATTEMPTS", "MIN_WORDS", "PAUSE", "TIMEOUT",
                                                     "MAX_BYTES", "OUTLET_TIME", "MAX_SITEMAPS", "UA")},
            "versions": {"python": platform.python_version(),
                         **{m: im.version(m) for m in ("trafilatura", "requests", "feedparser", "lxml", "pandas")}}})

    done = {p.stem for p in (folder / "stato").glob("*.json")}
    todo = [d for d in outlets.dominio if d not in done]
    say(f"=== start: {len(outlets)} outlets, {len(done)} already done, {len(todo)} to do, {args.processes} processes ===")

    ctx = mp.get_context("fork")
    running = {}
    while todo or running:
        while todo and len(running) < args.processes:
            d = todo.pop(0)
            p = ctx.Process(target=outlet_process, args=(d, folder), daemon=True)
            p.start()
            running[d] = (p, time.time())
        for d, (p, t0) in list(running.items()):
            if p.is_alive() and time.time() - t0 <= KILL_TIME:
                continue
            if p.is_alive():                                   # over the time limit: kill it
                p.terminate(); p.join(10)
                if p.is_alive():
                    p.kill(); p.join(5)
            p.join()
            del running[d]
            status_file = folder / "stato" / f"{d}.json"
            if not status_file.exists():                       # died without writing its status
                write_json(status_file, {"dominio": d, "esito": "process died", "articoli": 0, "fine": now()})
            s = json.loads(status_file.read_text())
            say(f"[{len(list((folder / 'stato').glob('*.json')))}/{len(outlets)}] {d}: {s.get('esito')}, "
                f"{s.get('articoli', 0)} articles")
        time.sleep(2)
    say("=== FINISHED ===")


if __name__ == "__main__":
    main()
