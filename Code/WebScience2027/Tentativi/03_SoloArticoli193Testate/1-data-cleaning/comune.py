"""Percorsi, costanti e funzioni comuni ai notebook di data cleaning."""
import re

import pandas as pd

DATASET = "../Dataset/NuovoScraping/dataset_articoli.csv"
GREZZI = "../Dataset/NuovoScraping/"          # stato/, html/ (l'HTML grezzo sta solo sul server)
NEWSGUARD = "../Dataset/newsguard.csv"
DERIVED = "../Dataset/derived/"
VERSIONI = "../Dataset/versioni/"
RESULTS = "../Results/1-data-cleaning/"
SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]


def carica_articoli():
    """Il dataset di articoli costruito da 0-scraping, con le colonne in inglese usate dal resto del codice."""
    return pd.read_csv(DATASET).rename(columns={"dominio": "domain", "testo": "text"})


def normalizza(testo):
    """Minuscole, solo lettere e cifre, spazi singoli: per confrontare testi senza badare alla forma."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(testo).lower())).strip()


def frasi(riga):
    """Le frasi di una riga di testo (si spezza dopo . ! ?)."""
    return [f for f in re.split(r"(?<=[.!?])\s+", riga.strip()) if f]


def normalizza_owner(owner):
    """Il campo Owner di NewsGuard, senza forme societarie: serve SOLO a verificare le reti trovate."""
    o = re.sub(r"[^a-z0-9 ]", " ", str(owner).lower())
    o = re.sub(r"\b(inc|llc|ltd|limited|gmbh|co|kg|corp|corporation|company|the|group|media|plc|sa|ag|"
               r"holdings?|news|publishing)\b", " ", o)
    o = re.sub(r"\s+", " ", o).strip()
    return "" if o in ("", "nan", "unknown") else o
