"""Tentativo 09, cambio di paradigma: dall'HTML di ogni articolo ai segnali della testata.

Due famiglie, entrambe osservabili per qualunque sito (niente metadati NewsGuard in ingresso):
- link: i domini citati nel testo dell'articolo (i link che trafilatura conserva nel testo estratto), escluso il
  sito stesso;
- pagina: com'è fatta la pagina che contiene l'articolo (servizi di terze parti come pubblicità e tracker, firma,
  data, link alle pagine di trasparenza del sito, contenuti incorporati, appelli a donazioni e newsletter).

Non si usano Wikipedia, Twitter, traffico web né la forma dell'URL (le famiglie di Baly et al. 2018).
"""
import json
import re
from pathlib import Path

import tldextract

VERSIONI = Path("../../../../../Dataset/tentativi/09_CambioParadigma/versioni")
HTML = Path("../../../../../Dataset/scraping/NuovoScraping/html")
SEGNALI = Path("../../../../../Dataset/tentativi/09_CambioParadigma/segnali")
DISCONNECT = Path("../../../../../Dataset/tentativi/09_CambioParadigma/derived/disconnect_services.json")
NEWSGUARD = Path("../../../../../Dataset/etichette/newsguard_2022.csv")
RESULTS = Path("../Results/2-segnali")
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]

_ESTRAI = tldextract.TLDExtract(suffix_list_urls=())      # lista dei suffissi inclusa nel pacchetto: niente rete


def dominio(url):
    """Il dominio registrato di un URL (news.bbc.co.uk -> bbc.co.uk); "" se non c'è."""
    try:
        e = _ESTRAI(url)
        return e.top_domain_under_public_suffix.lower() if e.suffix else ""
    except Exception:
        return ""


def categorie_disconnect():
    """Dominio -> categoria di servizio secondo la lista pubblica di Disconnect (quella dei blocchi di Firefox):
    Advertising, Analytics, Social, Content, Fingerprinting…"""
    priorita = ["Advertising", "Analytics", "Social", "FingerprintingInvasive", "FingerprintingGeneral", "Cryptomining",
                "Content", "Email", "EmailAggressive"]                 # un dominio in più categorie: vince la prima
    rango = lambda c: priorita.index(c) if c in priorita else len(priorita)
    out = {}
    for cat, lista in json.load(open(DISCONNECT))["categories"].items():
        for ente in lista:
            for sotto in ente.values():
                for url, domini in sotto.items():
                    if isinstance(domini, list):
                        for d in domini:
                            k = dominio("http://" + d)
                            if k and (k not in out or rango(cat) < rango(out[k])):
                                out[k] = cat
    return out


# pagine di trasparenza del sito: si cercano nei link della pagina (percorso o testo del link)
TRASPARENZA = {
    "chi siamo": r"about|chi-siamo|who-we-are|our-story|mission",
    "contatti": r"contact",
    "rettifiche": r"correction|retraction|errata",
    "etica e standard": r"ethic|editorial-(standards|policy|policies|guidelines)|standards|code-of-conduct",
    "redazione": r"masthead|our-team|/team|/staff|our-people|leadership|editors",
    "proprieta e finanziamento": r"ownership|funding|financ|transparency|who-funds|supporters",
    "pubblicita (inserzionisti)": r"advertis",
    "donazioni": r"donat|support-us|contribute|become-a-member|membership|patreon|givesendgo|subscribestar",
    "newsletter": r"newsletter|subscribe|sign-up|signup",
}
# contenuti incorporati: iframe o link a piattaforme video e social
INCORPORATI = {
    "video mainstream": r"youtube\.com|youtu\.be|vimeo\.com",
    "video alternativi": r"rumble\.com|bitchute\.com|odysee\.com|brighteon\.com",
    "social": r"twitter\.com|x\.com|facebook\.com|instagram\.com|tiktok\.com|t\.me|truthsocial\.com|gab\.com|gettr\.com",
}
