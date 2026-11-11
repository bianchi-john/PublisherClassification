"""Tentativo 10: percorsi e regole comuni per le etichette MBFC (Media Bias/Fact Check).

I dati MBFC vengono dall'API ufficiale (piano gratuito su RapidAPI, scaricati da John il 23 ottobre 2026) e, per un
controllo incrociato, dal file pubblico dell'estensione ufficiale MBFC per browser (22 ottobre 2026). Come i dati
NewsGuard, stanno in Dataset/ (fuori da git) e non si ridistribuiscono; chi li usa deve citare MediaBiasFactCheck.com.
"""
from pathlib import Path

import pandas as pd
import tldextract

DATI = Path("../../../../../Dataset")             # la cartella unica dei dati, nella radice del repo
DATASET = DATI / "tentativi" / "10_EtichetteMBFC"
RESULTS = Path("../Results/1-etichette")
MBFC_API = DATASET / "mbfc_api_2026-10-23.json"
MBFC_ESTENSIONE = DATASET / "mbfc_estensione_2026-10-22.json"
NEWSGUARD = DATI / "etichette" / "newsguard_2022.csv"
FINALE = DATASET / "versioni" / "fold_finale.csv"
TENTATE = DATI / "scraping" / "NuovoScraping" / "elenco_testate.csv"
ARTICOLI = DATI / "scraping" / "NuovoScraping" / "dataset_articoli.csv"

# livelli MBFC dal migliore al peggiore
FACTUAL = ["Very High", "High", "Mostly Factual", "Mixed", "Low", "Very Low"]
CREDIBILITA = ["High", "Medium", "Low"]
ORIENTAMENTO = ["Extreme Left", "Left", "Left-Center", "Least Biased", "Right-Center", "Right", "Extreme Right"]
CATEGORIA = ["Left", "Left-Center", "Least Biased", "Right-Center", "Right", "Pro-Science", "Conspiracy-Pseudoscience",
             "Questionable", "Satire"]
# livelli NewsGuard dal punteggio, come nei tentativi 05-09
LIVELLI_NG = [-1, 39, 59, 74, 99, 100]

_estrai = tldextract.TLDExtract(suffix_list_urls=())        # lista dei suffissi inclusa nel pacchetto, senza rete


def registrato(dominio):
    """Il dominio registrato (es. news.bbc.co.uk -> bbc.co.uk)."""
    return _estrai(dominio).top_domain_under_public_suffix or dominio


def livello_newsguard(score):
    return pd.cut(pd.Series(score), LIVELLI_NG, labels=False)
