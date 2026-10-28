"""Percorsi e dati per la pulizia, sul nuovo dataset (post-scraping)."""
import re
import pandas as pd

DATASET = "../../../../../Dataset/scraping/NuovoScraping/dataset_articoli.csv"
NEWSGUARD = "../../../../../Dataset/etichette/newsguard_2022.csv"
DERIVED = "../../../../../Dataset/tentativi/02_ScrapingConPagine/derived/"
RESULTS = "../Results/1-data-cleaning/"
SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]


def carica_dataset():
    """Il nuovo dataset. Rinomino le colonne italiane in quelle usate dal resto del codice."""
    df = pd.read_csv(DATASET)
    df = df.rename(columns={"dominio": "domain", "testo": "text"})
    return df


def normalizza(testo):
    """Minuscole, numeri sostituiti da #, niente punteggiatura."""
    testo = re.sub(r"\d+", "#", str(testo).lower())
    testo = re.sub(r"[^a-z# ]", " ", testo)
    return re.sub(r"\s+", " ", testo).strip()
