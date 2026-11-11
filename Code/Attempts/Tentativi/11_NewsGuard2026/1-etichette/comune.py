"""Tentativo 11: percorsi e costanti per le etichette NewsGuard aggiornate (feed S3, 23 ottobre 2026).

I dati NewsGuard sono riservati: stanno in Dataset/ nella radice del repo (fuori da git) e non si ridistribuiscono. Le credenziali del feed
S3 non stanno nel repository: si usano con il profilo AWS "newsguard" di chi lancia (~/.aws/credentials).
"""
from pathlib import Path

import pandas as pd

DATI = Path("../../../../../Dataset")             # la cartella unica dei dati, nella radice del repo
DATASET = DATI / "tentativi" / "11_NewsGuard2026"
RESULTS = Path("../Results/1-etichette")
NG2026 = DATI / "etichette" / "NewsGuard2026" / "metadata-2026102308.csv"
NG2022 = DATI / "etichette" / "newsguard_2022.csv"
ARTICOLI = DATI / "scraping" / "NuovoScraping" / "dataset_articoli.csv"
PULITI = DATASET / "derived" / "01_articoli_puliti.csv"
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]
CONFINI = [-1, 39, 59, 74, 99, 100]                 # dal punteggio ai livelli, come nei tentativi 05-10
PUNTI = [22, 18, 12.5, 12.5, 10, 7.5, 7.5, 5, 5]
# i 9 criteri nell'ordine dei punti; nel 2026 due hanno cambiato nome (stesso peso)
CRITERI_2026 = ["Does not publish false or egregiously misleading content at a significant level",
                "Gathers and presents information responsibly", "Has effective practices for correcting errors",
                "Handles the difference between news and opinion responsibly", "Avoids deceptive headlines",
                "Website discloses ownership and financing", "Clearly labels advertising",
                "Reveals who's in charge, including any possible conflicts of interest",
                "The site provides names of content creators, along with either contact or biographical information"]
CRITERI_2022 = ["Does not repeatedly publish false content"] + CRITERI_2026[1:2] + ["Regularly corrects or clarifies errors"] + CRITERI_2026[3:]


def leggi_newsguard(percorso):
    """Una riga per dominio (minuscolo), come nei tentativi precedenti."""
    ng = pd.read_csv(percorso, low_memory=False)
    return ng.assign(d=ng.Domain.str.lower()).drop_duplicates("d").set_index("d")


def livello(score):
    return pd.cut(pd.Series(score), CONFINI, labels=False)
