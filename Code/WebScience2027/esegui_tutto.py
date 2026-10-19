"""Tutta la catena sul server, senza nessuno davanti: scraping -> dataset -> pulizia -> classificazione.

    cd ~/John/WebScience2027
    screen -dmS tutto bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'

- ogni fase ha un tempo massimo; allo scadere si uccide il suo gruppo di processi e si va avanti;
- ogni fase finita lascia un segno in Results/esegui_tutto/: se lo script viene rilanciato la salta;
- se la costruzione del dataset o la pulizia falliscono, la classificazione usa il dataset di riserva
  (quello del weekend, 193 testate, Dataset/riserva_weekend/) e il log lo dice chiaramente;
- la classificazione è 2-classificazione/coda_weekend.py, che ha già i suoi tentativi e i suoi tempi massimi;
- log in Results/esegui_tutto/esegui_tutto.log, stato in Results/esegui_tutto/stato.json.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

W = Path(__file__).resolve().parent
R = W / "Results" / "esegui_tutto"
LOG = R / "esegui_tutto.log"
STATO = R / "stato.json"
MBVENV = Path.home() / "John" / "johnVenv" / "bin" / "python"
AMBIENTE = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
            "PYTHON": str(Path.home() / "John" / "johnVenv" / "bin" / "python")}


def papermill(cartella, nb, uscita):
    return [str(MBVENV), "-m", "papermill", "-k", "johnvenv", "--cwd", str(W / cartella), str(W / cartella / nb), str(W / uscita)]


FASI = [
    {"nome": "scraping", "ore": 3, "cwd": "0-scraping", "obbligatoria": False,
     "cmd": ["bash", "avvia.sh", "../Dataset/NuovoScraping", "64"]},
    {"nome": "costruisci_dataset", "ore": 2, "cwd": "0-scraping", "obbligatoria": True,
     "cmd": papermill("0-scraping", "costruisci_dataset.ipynb", "Results/0-scraping/costruisci_dataset.ipynb")},
    {"nome": "01_pulizia_articoli", "ore": 2, "cwd": "1-data-cleaning", "obbligatoria": True,
     "cmd": papermill("1-data-cleaning", "01_pulizia_articoli.ipynb", "Results/1-data-cleaning/esecuzioni/01_pulizia_articoli.ipynb")},
    {"nome": "02_reti", "ore": 1, "cwd": "1-data-cleaning", "obbligatoria": True,
     "cmd": papermill("1-data-cleaning", "02_reti.ipynb", "Results/1-data-cleaning/esecuzioni/02_reti.ipynb")},
    {"nome": "03_dataset_finale", "ore": 1, "cwd": "1-data-cleaning", "obbligatoria": True,
     "cmd": papermill("1-data-cleaning", "03_dataset_finale.ipynb", "Results/1-data-cleaning/esecuzioni/03_dataset_finale.ipynb")},
    {"nome": "classificazione", "ore": 60, "cwd": "2-classificazione", "obbligatoria": False,
     "cmd": [str(MBVENV), "coda_weekend.py"]},
]
DATASET_FASI = {"costruisci_dataset", "01_pulizia_articoli", "02_reti", "03_dataset_finale"}


def ora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scrivi_log(m):
    riga = f"{ora()}  {m}"
    print(riga, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(riga + "\n")


def leggi_stato():
    return json.loads(STATO.read_text()) if STATO.exists() else {"fasi": {}, "riserva": False}


def salva_stato(s):
    s["aggiornato"] = ora()
    tmp = STATO.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    os.replace(tmp, STATO)


def esegui(fase):
    out = R / f"{fase['nome']}.out"
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"\n===== {ora()} {' '.join(fase['cmd'])}\n"); f.flush()
        p = subprocess.Popen(fase["cmd"], cwd=W / fase["cwd"], stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, env=AMBIENTE)
        try:
            return p.wait(timeout=fase["ore"] * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "tempo scaduto"


def usa_riserva(stato, motivo):
    versioni = W / "Dataset" / "versioni"
    versioni.mkdir(parents=True, exist_ok=True)
    for f in (W / "Dataset" / "riserva_weekend").glob("*.csv"):
        shutil.copy(f, versioni / f.name)
    stato["riserva"] = True
    scrivi_log(f"!!! {motivo}: la classificazione usa il DATASET DI RISERVA (weekend, 193 testate)")


def main():
    for d in ["Results/esegui_tutto", "Results/0-scraping", "Results/1-data-cleaning/esecuzioni", "Results/2-classificazione/esecuzioni"]:
        (W / d).mkdir(parents=True, exist_ok=True)
    stato = leggi_stato()
    scrivi_log("=== avvio (o ripresa) della catena ===")
    dataset_ok = True
    for fase in FASI:
        s = stato["fasi"].setdefault(fase["nome"], {"stato": "in attesa", "tentativi": 0})
        if s["stato"] in ("fatta", "conclusa con errori"):
            if s["stato"] != "fatta" and fase["nome"] in DATASET_FASI:
                dataset_ok = False
            continue
        if fase["nome"] in DATASET_FASI and not dataset_ok:
            s["stato"] = "saltata"; salva_stato(stato); continue
        if fase["nome"] == "classificazione" and not dataset_ok and not stato["riserva"]:
            usa_riserva(stato, "il dataset nuovo non è stato costruito")
            salva_stato(stato)
        if fase["nome"] == "costruisci_dataset":         # copia di sicurezza del dataset del primo giro
            vecchio = W / "Dataset" / "NuovoScraping" / "dataset_articoli.csv"
            copia = W / "Dataset" / "NuovoScraping" / "dataset_articoli_13ottobre.csv"
            if vecchio.exists() and not copia.exists():
                shutil.copy(vecchio, copia)
        s["tentativi"] += 1; s["stato"] = "in corso"; s["inizio"] = ora(); salva_stato(stato)
        scrivi_log(f">>> {fase['nome']} (tentativo {s['tentativi']}, al massimo {fase['ore']} h)")
        t0 = time.time()
        esito = esegui(fase)
        s["minuti"] = round((time.time() - t0) / 60, 1); s["esito"] = str(esito); s["fine"] = ora()
        if esito == 0:
            s["stato"] = "fatta"; scrivi_log(f"<<< {fase['nome']}: fatta in {s['minuti']} min")
        elif fase["obbligatoria"] and s["tentativi"] < 2:
            s["stato"] = "in attesa"; salva_stato(stato)
            scrivi_log(f"!!! {fase['nome']}: non riuscita (esito {esito}); la riprovo una volta")
            return 1                                         # il ciclo di screen rilancia lo script tra 2 minuti
        else:
            s["stato"] = "conclusa con errori"
            scrivi_log(f"!!! {fase['nome']}: non riuscita (esito {esito}); vedi Results/esegui_tutto/{fase['nome']}.out; vado avanti")
            if fase["nome"] in DATASET_FASI:
                dataset_ok = False
        salva_stato(stato)
    scrivi_log("=== catena finita ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
