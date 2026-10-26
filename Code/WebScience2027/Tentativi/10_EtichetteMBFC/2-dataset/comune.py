"""Tentativo 10: percorsi e costanti per costruire i dataset con le etichette MBFC."""
import re
from pathlib import Path

DATASET = Path("../Dataset")
RESULTS = Path("../Results/2-dataset")
VERSIONI = DATASET / "versioni"
SEED = 42
# factual reporting MBFC dal peggiore al migliore: lo stesso verso dei livelli NewsGuard (l1 peggiore, l5 migliore).
# Very High (3 testate su 432 nel dataset finale) va con High
LIVELLI = ["Very Low", "Low", "Mixed", "Mostly Factual", "High"]
DA_FACTUAL = {"Very Low": 0, "Low": 1, "Mixed": 2, "Mostly Factual": 3, "High": 4, "Very High": 4}
SOGLIA_BINARIO = 3          # affidabile = almeno Mostly Factual


def normalizza_owner(owner):
    """Il campo Owner di NewsGuard, senza forme societarie (come in 09_CambioParadigma/1-data-cleaning/comune.py)."""
    o = re.sub(r"[^a-z0-9 ]", " ", str(owner).lower())
    o = re.sub(r"\b(inc|llc|ltd|limited|gmbh|co|kg|corp|corporation|company|the|group|media|plc|sa|ag|"
               r"holdings?|news|publishing)\b", " ", o)
    o = re.sub(r"\s+", " ", o).strip()
    return "" if o in ("", "nan", "unknown") else o
