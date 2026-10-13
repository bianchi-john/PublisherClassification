"""Percorsi e caricamento dei dati, usati dai notebook di pulizia."""
import re

import pandas as pd

DATASET = "../Dataset/CleannedDataset/newDataframeWithOldCleanedPoliticsVersion.csv"
NEWSGUARD = "../Dataset/newsguard.csv"
DERIVED = "../Dataset/derived/"          # file intermedi grandi, fuori da git
PULITO = "../Dataset/DatasetPulito/"     # le versioni pulite del dataset, fuori da git
RESULTS = "../Results/1-data-cleaning/"
SEED = 42
LIVELLI = ["l1", "l2", "l3", "l4", "l5"]


def carica_dataset():
    df = pd.read_csv(DATASET)
    df["article_id"] = df.index
    return df


def normalizza(testo):
    """Minuscole, numeri sostituiti da #, niente punteggiatura."""
    testo = re.sub(r"\d+", "#", str(testo).lower())
    testo = re.sub(r"[^a-z# ]", " ", testo)
    return re.sub(r"\s+", " ", testo).strip()
