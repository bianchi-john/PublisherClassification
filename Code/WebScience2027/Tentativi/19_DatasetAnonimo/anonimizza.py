"""Tentativo 19: il dataset del tentativo 16 con i testi anonimizzati, perché il modello non possa riconoscere la testata
da quello che c'è scritto (il suo nome, i suoi autori, i suoi link) e usare quello che già sa di lei.

Solo regole semplici, una per tipo di identificativo, nell'ordine:
1. firma nei titoli (la coda ripetuta, «… | The New York Sun»)  → [TESTATA]
2. email                                                     → [EMAIL]
3. link (http, www, domini)                                  → [URL]
4. account (@nome)                                           → [ACCOUNT]
5. autori della testata                                      → [AUTORE]
6. nome della testata, dal dominio (senza un «online» in coda) → [TESTATA]
Le stesse regole valgono per il titolo. Nient'altro si tocca: persone, luoghi e fatti di cui parla l'articolo restano.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def cartella_dati():
    """La cartella unica dei dati: Dataset/ nella radice del repo (sul server: ~/John/Dataset, accanto ai tentativi).
    Si cerca salendo dalla cartella di lavoro fino a una cartella Dataset/ con dentro LEGGIMI.md."""
    for p in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (p / "Dataset" / "LEGGIMI.md").exists():
            return p / "Dataset"
    raise FileNotFoundError("non trovo la cartella Dataset/ (con LEGGIMI.md) salendo dalla cartella di lavoro")


DATI = cartella_dati()
ORIGINE = DATI / "tentativi" / "16_DataPreparation"          # il dataset di partenza
DATASET = DATI / "tentativi" / "19_DatasetAnonimo"           # il dataset di questo tentativo

PIATTAFORME = {"substack"}            # siti ospitati: il nome della testata è il sottodominio (chriscillizza.substack.com)
SOGLIA_PAROLA_COMUNE = 20             # un nome è una parola comune se compare minuscolo, come parola, più di 20 volte
                                      # negli articoli delle altre testate (politics, time, reason, …)
QUOTA_MINUSCOLO = 0.8                 # ...ma è il nome se la testata lo scrive minuscolo in almeno l'80% dei suoi articoli
QUOTA_FIRMA = 0.3                     # la coda di titolo ripetuta in almeno il 30% dei titoli (e in almeno 3) è la firma
PAROLE_NON_PERSONA = {"press", "news", "staff", "editor", "editors", "editorial", "team", "board", "reuters", "agency",
                      "media", "service", "services", "desk", "wire", "newsroom", "report", "reports", "contributor",
                      "contributors", "guest", "admin", "administrator", "writer", "writers", "the"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?i:https?://|www\.)\S+"                       # link completi
                 r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|info|news|media|press|co|uk|us|ca|au|nz|cn|eu|io|tv|"
                 r"scot|su|one)\b(?:/\S*)?")                     # domini nudi, con il suffisso minuscolo (Bellingcat.com)
ACCOUNT = re.compile(r"(?<![\w@])@\w{2,}")


# ===================================================================================== 1. il nome della testata
def nomi_testata(dominio):
    """I nomi della testata dal dominio: la parte prima del suffisso (100percentfedup.com → 100percentfedup,
    spectator.co.uk → spectator); sulle piattaforme il sottodominio (chriscillizza.substack.com → chriscillizza).
    Un «online» in coda non fa parte del nome: spiked-online.com → anche spiked."""
    import tldextract
    e = tldextract.TLDExtract(suffix_list_urls=())(dominio)
    nome = re.sub(r"[^a-z0-9]", "", (e.subdomain.split(".")[-1] if e.domain in PIATTAFORME and e.subdomain else e.domain).lower())
    return [nome] + ([nome[:-6]] if nome.endswith("online") and len(nome) > 9 else [])


def _scritto_staccato(nome):
    """Il nome con al più uno spazio o un trattino tra due caratteri: «theamericanconservative» trova anche
    «The American Conservative», «100percentfedup» anche «100PercentFedUp» e «100 Percent Fed Up»."""
    return r"[\s\-]?".join(map(re.escape, nome))


def _testi_senza_link(testi):
    return pd.Series(testi).astype(str).map(lambda t: ACCOUNT.sub(" ", URL.sub(" ", EMAIL.sub(" ", t))))


def parole_comuni(nomi, testi, domini):
    """I nomi (e i nomi senza «the») che sono anche parole comuni: compaiono minuscoli, come parola intera, più di
    SOGLIA_PAROLA_COMUNE volte negli articoli delle altre testate, fuori da link, email e account. Le parole si
    contano una volta sola per testata."""
    import collections
    conta = collections.defaultdict(collections.Counter)            # testata → parola esatta → volte
    for d, t in zip(domini, _testi_senza_link(testi)):               # i nomi dentro link e account non contano
        conta[d].update(re.findall(r"[A-Za-z0-9]+", t))
    totale = sum(conta.values(), collections.Counter())
    comuni = set()
    for d, lista in nomi.items():
        for nome in lista:
            for n in {nome, nome[3:] if nome.startswith("the") else nome}:
                if len(n) >= 2 and totale[n] - conta[d][n] > SOGLIA_PAROLA_COMUNE:
                    comuni.add(n)
    return comuni


def nomi_minuscoli(nomi, comuni, testi, domini):
    """I nomi che sono parole comuni ma che la testata usa come nome scritto minuscolo: la parola compare in almeno
    QUOTA_MINUSCOLO dei suoi articoli (spiked-online.com scrive «spiked» nel 100% degli articoli, time.com «time» nel 56%).
    Per loro vale la regola dei nomi distintivi."""
    testi, domini = _testi_senza_link(testi).reset_index(drop=True), pd.Series(domini).reset_index(drop=True)
    propri = set()
    for d, lista in nomi.items():
        for n in lista:
            if n in comuni:
                pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])")
                if testi[domini == d].str.contains(pat).mean() >= QUOTA_MINUSCOLO:
                    propri.add(n)
    return propri


def schema_testata(nomi, comuni, minuscoli=frozenset()):
    """L'espressione che trova i nomi della testata nel testo.
    - Nome distintivo (bellingcat): in qualunque forma, maiuscolo o minuscolo, attaccato o staccato, con o senza «the».
    - Nome che è anche una parola comune (politics, time): solo con l'iniziale maiuscola (Politics, TIME), così la
      parola comune scritta minuscola resta; tranne quando la testata scrive così il proprio nome (`nomi_minuscoli`).
    - Nome che comincia con «the» (theweek): «the» si può omettere solo se il resto non è una parola comune
      (American Conservative sì, week no)."""
    maiuscola = lambda n: f"(?-i:{re.escape(n[0].upper())})" + "".join(r"[\s\-]?" + re.escape(c) for c in n[1:])
    forme = []
    for nome in nomi:
        resto = nome[3:] if nome.startswith("the") and len(nome) > 3 else None
        if nome in comuni and nome not in minuscoli:       # politics: «Politics», «POLITICS», mai «politics»
            forme.append(maiuscola(nome))
        elif resto and resto in comuni:                    # theweek: «The Week», «TheWeek», mai «the week»
            forme.append(maiuscola("the") + r"[\s\-]?" + maiuscola(resto))
        else:
            forme += [r"(?:the[\s\-]?)?" + _scritto_staccato(n) for n in [nome] + ([resto] if resto else [])]
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(forme) + r")(?![A-Za-z0-9])", re.I)


def coda_titolo(titolo):
    """L'ultima parte del titolo dopo « | », « – », « — » o « - » (Children in foster care … | The Imprint → The Imprint)."""
    m = re.search(r"\s[|–—-]\s([^|–—]{2,60})$", str(titolo))
    return m.group(1).strip() if m else None


def firme_titoli(art):
    """La firma di ogni testata: la coda di titolo che ripete in almeno QUOTA_FIRMA dei suoi titoli (e in almeno 3).
    Trova i nomi che dal dominio non si ricavano: cfr.org → «Council on Foreign Relations», nysun.com → «The New York Sun».
    Una firma di una sola parola che è anche una parola comune (bamapolitics.com: «Signed») non è un nome e si scarta."""
    testi, domini = _testi_senza_link(art.text).reset_index(drop=True), art.domain.reset_index(drop=True)
    firme = {}
    for d, g in art.groupby("domain"):
        code = g.titolo.map(coda_titolo).value_counts()
        if len(code) and code.iloc[0] >= max(3, QUOTA_FIRMA * len(g)):
            f = code.index[0]
            if " " not in f:
                pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(f.lower()) + r"(?![A-Za-z0-9])")
                if testi[domini != d].str.count(pat).sum() > SOGLIA_PAROLA_COMUNE:
                    continue
            firme[d] = f
    return firme


def schema_firma(firma):
    """La firma, maiuscole o minuscole, fino alla fine dell'ultima parola: «Frontpage Mag» trova anche «FrontPage Magazine»."""
    return re.compile(r"(?<![\w])" + re.escape(firma) + r"\w*", re.I) if firma else None


# ===================================================================================== 2. gli autori
def autori_personali(art):
    """Per ogni testata, gli autori che firmano solo lì e hanno l'aspetto di un nome di persona: da 1 a 4 parole con
    l'iniziale maiuscola, senza cifre e senza parole come Press, Staff, Editor, Team. Chi firma in più testate (le
    agenzie, «Staff») non identifica la testata e non si tocca. Un nome di una sola parola (Jack) si toglie solo come
    firma, dopo «by» (vedi `schema_autori`)."""
    righe = []
    for d, a in zip(art.domain, art.autore.fillna("")):
        for nome in re.split(r"\s*(?:;|,|\||&|\band\b)\s*", str(a)):
            nome = re.sub(r"^\s*(?:by|di)\s+", "", nome.strip(), flags=re.I).strip(" .")
            parole = nome.split()
            if (1 <= len(parole) <= 4 and len(nome) >= 3 and all(re.fullmatch(r"[A-ZÀ-Ý][\w.'’-]*", p) for p in parole)
                    and not any(p.lower().strip(".") in PAROLE_NON_PERSONA for p in parole)):
                righe.append((d, nome))
    r = pd.DataFrame(righe, columns=["domain", "autore"]).drop_duplicates()
    quante = r.groupby("autore").domain.nunique()
    r = r[r.autore.map(quante) == 1]
    return r.groupby("domain").autore.apply(lambda s: sorted(s, key=len, reverse=True)).to_dict()


def schema_autori(nomi):
    """I nomi degli autori, esatti (maiuscole comprese), come parole intere, i più lunghi prima: quelli di 2-4 parole
    ovunque, quelli di una parola solo come firma («by Jack»)."""
    completi = [n for n in nomi if " " in n]
    singoli = [n for n in nomi if " " not in n]
    forme = []
    if completi:
        forme.append(r"(?<![\w])(?:" + "|".join(re.escape(n) for n in completi) + r")(?![\w])")
    if singoli:
        forme.append(r"(?<=\b[Bb]y )(?:" + "|".join(re.escape(n) for n in singoli) + r")(?![\w])")
    return re.compile("|".join(forme)) if forme else None


# ===================================================================================== 3. le regole insieme
def anonimizza_testo(testo, schema_nome, schema_aut, schema_fir=None):
    """Le sei regole, nell'ordine. Ritorna il testo e quante sostituzioni ha fatto ogni regola."""
    conta = {}
    testo = str(testo) if isinstance(testo, str) else ""
    for chiave, schema, segno in [("firma", schema_fir, "[TESTATA]"), ("email", EMAIL, "[EMAIL]"), ("url", URL, "[URL]"),
                                  ("account", ACCOUNT, "[ACCOUNT]"),
                                  ("autore", schema_aut, "[AUTORE]"), ("testata", schema_nome, "[TESTATA]")]:
        if schema is None:
            conta[chiave] = 0
            continue
        testo, conta[chiave] = schema.subn(segno, testo)
    return testo, conta


def anonimizza(art, nomi, comuni, minuscoli, autori, firme):
    """Tutti gli articoli: testo e titolo. Ritorna il dataset anonimo e una tabella con le sostituzioni per articolo."""
    out, conti = [], []
    for d, g in art.groupby("domain", sort=False):
        sn, sa, sf = schema_testata(nomi[d], comuni, minuscoli), schema_autori(autori.get(d, [])), schema_firma(firme.get(d))
        for i, r in g.iterrows():
            testo, c = anonimizza_testo(r.text, sn, sa, sf)
            titolo, ct = anonimizza_testo(r.titolo, sn, sa, sf)
            out.append((i, testo, titolo))
            conti.append({"article_id": r.article_id, "domain": d, **c, **{f"titolo_{k}": v for k, v in ct.items()}})
    nuovo = art.copy()
    idx, testi, titoli = zip(*out)
    nuovo.loc[list(idx), "text"] = list(testi)
    nuovo.loc[list(idx), "titolo"] = list(titoli)
    return nuovo, pd.DataFrame(conti)


# ===================================================================================== 4. i controlli automatici
def residui(art, nomi, comuni, minuscoli, firme):
    """Dopo l'anonimizzazione, per ogni testata: in quanti articoli (testo o titolo) restano i suoi nomi e la sua firma,
    come parola intera e anche dentro altre parole (es. «Quilletteessay», «AAFF»). Per i nomi che sono parole comuni
    resta per scelta la parola minuscola: si segnano a parte."""
    righe = []
    for d, g in art.groupby("domain"):
        testo = g.text.astype(str) + "\n" + g.titolo.astype(str)
        cerca = [re.escape(n) for n in nomi[d]] + ([re.escape(firme[d])] if d in firme else [])
        largo = re.compile("|".join(cerca), re.I)
        parola = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(cerca) + r")(?![A-Za-z0-9])", re.I)
        comune = any((n in comuni or n[3:] in comuni) and n not in minuscoli for n in nomi[d])
        righe.append({"domain": d, "nomi": " / ".join(nomi[d]), "firma": firme.get(d, ""), "parola_comune": comune,
                      "articoli": len(g), "residuo_come_parola": int(testo.str.contains(parola).sum()),
                      "residuo_anche_dentro_parole": int(testo.str.contains(largo).sum())})
    return pd.DataFrame(righe)


def contesti(testi, schema, quanti=5, larghezza=70, seme=0):
    """Qualche esempio del testo intorno a una corrispondenza, per vedere che cosa resta o che cosa si è tolto."""
    rng = np.random.default_rng(seme)
    trovati = []
    for t in testi:
        for m in schema.finditer(str(t)):
            trovati.append(str(t)[max(0, m.start() - larghezza): m.end() + larghezza].replace("\n", " "))
    if not trovati:
        return []
    return [trovati[i] for i in rng.choice(len(trovati), size=min(quanti, len(trovati)), replace=False)]
