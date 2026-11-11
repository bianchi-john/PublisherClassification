"""Versione 2 sul server, senza nessuno davanti: pulizia (01 -> 02 -> 03) e poi la coda della classificazione.

    cd ~/John/WebScience2027/Tentativi/06_Versione2_PiuPuliziaPiuTraining
    screen -dmS v2 bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'

- lo scraping non si rifà: si parte da ../../../../../Dataset/scraping/NuovoScraping (lo stesso della prima versione);
- ogni fase ha un tempo massimo; allo scadere si uccide il suo gruppo di processi;
- ogni fase finita lascia un segno in Results/esegui_tutto/stato.json: se lo script viene rilanciato la salta;
- una fase che fallisce si riprova (lo script esce e il ciclo di screen lo rilancia tra 2 minuti): la pulizia una
  volta, la classificazione fino a 4 volte; se non basta, la catena si ferma e lo scrive nel log;
- la classificazione è 2-classificazione/coda.py, che ha già i suoi tentativi per esperimento, i suoi tempi massimi
  e la ripresa (salta quello che è fatto, riparte dal fold a cui era arrivata);
- log in Results/esegui_tutto/esegui_tutto.log, uscita di ogni fase in Results/esegui_tutto/<fase>.out.
"""
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

W = Path(__file__).resolve().parent
R = W / "Results" / "esegui_tutto"
LOG = R / "esegui_tutto.log"
STATO = R / "stato.json"
PY = Path.home() / "John" / "johnVenv" / "bin" / "python"
AMBIENTE = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}


def papermill(cartella, nb, uscita):
    return [str(PY), "-m", "papermill", "-k", "johnvenv", "--cwd", str(W / cartella), str(W / cartella / nb), str(W / uscita)]


FASI = [
    {"nome": "01_pulizia_articoli", "ore": 2, "cwd": "1-data-cleaning", "tentativi_max": 2,
     "cmd": papermill("1-data-cleaning", "01_pulizia_articoli.ipynb", "Results/1-data-cleaning/esecuzioni/01_pulizia_articoli.ipynb")},
    {"nome": "02_reti", "ore": 1, "cwd": "1-data-cleaning", "tentativi_max": 2,
     "cmd": papermill("1-data-cleaning", "02_reti.ipynb", "Results/1-data-cleaning/esecuzioni/02_reti.ipynb")},
    {"nome": "03_dataset_finale", "ore": 1, "cwd": "1-data-cleaning", "tentativi_max": 2,
     "cmd": papermill("1-data-cleaning", "03_dataset_finale.ipynb", "Results/1-data-cleaning/esecuzioni/03_dataset_finale.ipynb")},
    {"nome": "classificazione", "ore": 40, "cwd": "2-classificazione", "tentativi_max": 4,
     "cmd": [str(PY), "coda.py"]},
]


def ora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scrivi_log(m):
    riga = f"{ora()}  {m}"
    print(riga, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(riga + "\n")


def leggi_stato():
    return json.loads(STATO.read_text()) if STATO.exists() else {"fasi": {}}


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


def main():
    for d in ["Results/esegui_tutto", "Results/1-data-cleaning/esecuzioni", "Results/2-classificazione/esecuzioni", "../../../../Dataset/tentativi/06_Versione2_PiuPuliziaPiuTraining/derived"]:
        (W / d).mkdir(parents=True, exist_ok=True)
    stato = leggi_stato()
    scrivi_log("=== avvio (o ripresa) della catena della Versione 2 ===")
    for fase in FASI:
        s = stato["fasi"].setdefault(fase["nome"], {"stato": "in attesa", "tentativi": 0})
        if s["stato"] == "fatta":
            continue
        if s["stato"] == "fallita":
            scrivi_log(f"!!! {fase['nome']} è fallita troppe volte: la catena è ferma. Vedi Results/esegui_tutto/{fase['nome']}.out")
            return 0                                         # niente riavvii a vuoto
        s["tentativi"] += 1; s["stato"] = "in corso"; s["inizio"] = ora(); salva_stato(stato)
        scrivi_log(f">>> {fase['nome']} (tentativo {s['tentativi']}, al massimo {fase['ore']} h)")
        t0 = time.time()
        esito = esegui(fase)
        s["minuti"] = round((time.time() - t0) / 60, 1); s["esito"] = str(esito); s["fine"] = ora()
        if esito == 0:
            s["stato"] = "fatta"; scrivi_log(f"<<< {fase['nome']}: fatta in {s['minuti']} min")
        elif s["tentativi"] < fase["tentativi_max"]:
            s["stato"] = "in attesa"; salva_stato(stato)
            scrivi_log(f"!!! {fase['nome']}: non riuscita (esito {esito}); la riprovo tra 2 minuti "
                       f"(tentativo {s['tentativi'] + 1} di {fase['tentativi_max']}); riparte da dove era arrivata")
            return 1                                         # il ciclo di screen rilancia lo script
        else:
            s["stato"] = "fallita"; salva_stato(stato)
            scrivi_log(f"!!! {fase['nome']}: fallita {s['tentativi']} volte (esito {esito}); la catena si ferma qui. "
                       f"Vedi Results/esegui_tutto/{fase['nome']}.out")
            return 0
        salva_stato(stato)
    scrivi_log("=== catena finita ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
