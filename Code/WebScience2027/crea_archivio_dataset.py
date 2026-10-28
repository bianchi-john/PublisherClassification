"""Crea Dataset.zip nella radice del repo: la parte di Dataset/ che serve a rifare i tentativi 14-16 (il protocollo
attuale), in meno di 1 GB, da caricare su Google Drive. Si estrae nella radice del repo e il codice la trova da solo.

Restano fuori, perché pesano decine di GB e servono solo ai tentativi vecchi o a rifare lo scraping: l'HTML e i testi
grezzi dello scraping, gli embedding salvati, il feed NewsGuard completo, i dati intermedi dei tentativi 01-13. Sono sul
server, in ~/John/Dataset, e nel Dataset/ completo sul PC di John.

Uso, dalla radice del repo:   python Code/WebScience2027/crea_archivio_dataset.py
"""
import zipfile
from pathlib import Path

RADICE = Path(__file__).resolve().parents[2]
D = RADICE / "Dataset"
FILE = ["LEGGIMI.md",
        "etichette/newsguard_2022.csv", "etichette/newsguard_2026.csv",
        "risorse/lessici/mfd2.0.dic",
        "risorse/lessici/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "risorse/lessici/NRC-VAD-Lexicon/OneFilePerDimension/arousal-NRC-VAD-Lexicon.txt",
        "tentativi/12_PiuTestatePolitiche/derived/disconnect_services.json",
        "tentativi/14_DatasetRipensatoChunk/articoli.csv.gz", "tentativi/14_DatasetRipensatoChunk/testate.csv",
        "tentativi/16_DataPreparation/articoli.csv.gz", "tentativi/16_DataPreparation/testate.csv",
        "tentativi/16_DataPreparation/cache/estratti.csv.gz"]
# dello scraping di ottobre 2026 basta lo stato di ogni testata: il testo si riprende dalla cache di trafilatura
for giro in ["NuovoScraping_ottobre", "NuovoScraping_giro3"]:
    FILE += [f"scraping/{giro}/{f}" for f in ["elenco_testate.csv", "config.json", "stato_generale.json"]]
    FILE += [str(f.relative_to(D)) for f in sorted((D / "scraping" / giro / "stato").glob("*.json"))]

mancano = [f for f in FILE if not (D / f).exists()]
assert not mancano, f"mancano in Dataset/: {mancano}"
uscita = RADICE / "Dataset.zip"
with zipfile.ZipFile(uscita, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in FILE:
        # i file già compressi (.gz) si mettono così come sono
        z.write(D / f, f"Dataset/{f}", compress_type=zipfile.ZIP_STORED if f.endswith(".gz") else zipfile.ZIP_DEFLATED)
print(f"{uscita}: {len(FILE)} file, {uscita.stat().st_size / 2**20:.0f} MB")
