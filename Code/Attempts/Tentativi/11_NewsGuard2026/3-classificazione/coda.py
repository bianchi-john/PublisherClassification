"""La coda del weekend del tentativo 11 (dal 23 ottobre), in due corsie che girano insieme:

    cd ~/John/Tentativo11/3-classificazione
    screen -dmS t11_cpu bash -c 'until ~/John/johnVenv/bin/python coda.py cpu; do sleep 60; done'
    screen -dmS t11_gpu bash -c 'until ~/John/johnVenv/bin/python coda.py gpu; do sleep 60; done'

- cpu: il classificatore a feature (base, titolo, politici; etichette 2026 e 2022);
- gpu: ModernBERT-base, un modello alla volta sulla GPU condivisa (base 2026 e 2022, poi titolo e politici 2026, poi
  titolo e politici 2022), seme 42.

Ogni esperimento è un notebook eseguito con papermill in un processo a parte; è "fatto" quando c'è il suo FATTO (i
modelli ModernBERT ripartono dai fold già finiti). Se fallisce o supera il tempo massimo si riprova fino a 3 volte,
poi si passa al successivo. Dopo ogni esperimento finito si rifà il confronto (03), una corsia alla volta.
Stato in Results/3-classificazione/coda_<corsia>_stato.json, log in coda_<corsia>.log.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUI = Path(__file__).resolve().parent
RESULTS = QUI.parent / "Results" / "3-classificazione"
ESEC = RESULTS / "esecuzioni"
PY = sys.executable
MAX_TENTATIVI = 3
CORSIA = sys.argv[1] if len(sys.argv) > 1 else "cpu"
STATO = RESULTS / f"coda_{CORSIA}_stato.json"
LOG = RESULTS / f"coda_{CORSIA}.log"


def nome_di(base, titolo, politici, anno, seme=None):
    return base + ("_titolo" if titolo else "") + ("_politici" if politici else "") + f"_{anno}" + (f"_s{seme}" if seme else "")


def feature(titolo, politici, anno):
    n = nome_di("feature_tuned", titolo, politici, anno)
    return {"nome": n, "notebook": "01_feature_tfidf.ipynb", "ore_max": 3, "fatto": RESULTS / f"{n}_FATTO",
            "parametri": {"ETICHETTA": anno, "TITOLO": titolo, "SOLO_POLITICI": politici, "N_JOBS": 5}}


def bert(titolo, politici, anno, seme=42):
    n = nome_di("modernbert", titolo, politici, anno, seme)
    return {"nome": n, "notebook": "02_modernbert.ipynb", "ore_max": 8, "fatto": RESULTS / "transformer" / n / "FATTO",
            "parametri": {"ETICHETTA": anno, "TITOLO": titolo, "SOLO_POLITICI": politici, "SEED": seme}}


ESPERIMENTI = {
    "cpu": [feature(False, False, "2026"), feature(False, False, "2022"),
            feature(True, False, "2026"), feature(False, True, "2026"),
            feature(True, False, "2022"), feature(False, True, "2022")],
    "gpu": [bert(False, False, "2026"), bert(False, False, "2022"),
            bert(True, False, "2026"), bert(False, True, "2026"),
            bert(True, False, "2022"), bert(False, True, "2022")],
}[CORSIA]


def ora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scrivi_log(m):
    riga = f"{ora()}  {m}"
    print(riga, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(riga + "\n")


def salva(stato):
    stato["aggiornato"] = ora()
    tmp = STATO.with_suffix(".tmp"); tmp.write_text(json.dumps(stato, indent=1, ensure_ascii=False)); os.replace(tmp, STATO)


def lancia(notebook, uscita, parametri, ore_max, file_out):
    cmd = [PY, "-m", "papermill", "-k", "johnvenv", "--cwd", str(QUI), str(QUI / notebook), str(uscita)]
    for k, v in parametri.items():
        cmd += ["-p", k, str(v)]
    ambiente = {**os.environ, "PYTHONUTF8": "1", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    with open(file_out, "a", encoding="utf-8") as f:
        f.write(f"\n===== {ora()} {' '.join(cmd)}\n"); f.flush()
        p = subprocess.Popen(["nice", "-n", "10"] + cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=ambiente)
        try:
            return p.wait(timeout=ore_max * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); return "tempo scaduto"


def confronto():
    with open(RESULTS / ".confronto.lock", "w") as lucchetto:          # una corsia alla volta
        fcntl.flock(lucchetto, fcntl.LOCK_EX)
        esito = lancia("03_confronto.ipynb", ESEC / "03_confronto.ipynb", {}, 2, ESEC / "03_confronto.out")
    scrivi_log(f"    confronto aggiornato (esito {esito})")


def main():
    ESEC.mkdir(parents=True, exist_ok=True)
    stato = json.loads(STATO.read_text()) if STATO.exists() else {"esperimenti": {}}
    scrivi_log(f"=== corsia {CORSIA}: {len(ESPERIMENTI)} esperimenti ===")
    for e in ESPERIMENTI:
        s = stato["esperimenti"].setdefault(e["nome"], {"stato": "in attesa", "tentativi": 0})
        if e["fatto"].exists():
            if s["stato"] != "fatto":
                s["stato"] = "fatto"; salva(stato)
            continue
        while not e["fatto"].exists() and s["tentativi"] < MAX_TENTATIVI:
            s["tentativi"] += 1; s.update(stato="in corso", inizio=ora()); salva(stato)
            scrivi_log(f">>> {e['nome']}: tentativo {s['tentativi']}/{MAX_TENTATIVI}")
            t0 = time.time()
            esito = lancia(e["notebook"], ESEC / f"{e['nome']}.ipynb", e["parametri"], e["ore_max"], ESEC / f"{e['nome']}.out")
            s["minuti"] = round(s.get("minuti", 0) + (time.time() - t0) / 60, 1)
            if e["fatto"].exists():
                s.update(stato="fatto", fine=ora()); scrivi_log(f"<<< {e['nome']}: fatto in {(time.time() - t0) / 60:.0f} min")
            else:
                s.update(stato="fallito" if s["tentativi"] >= MAX_TENTATIVI else "da riprovare", ultimo_errore=str(esito))
                scrivi_log(f"!!! {e['nome']}: non riuscito (esito {esito}); vedi esecuzioni/{e['nome']}.out")
                if s["tentativi"] < MAX_TENTATIVI:
                    time.sleep(120)
            salva(stato)
        if e["fatto"].exists():
            confronto()
    scrivi_log(f"=== corsia {CORSIA} finita ===")


if __name__ == "__main__":
    main()
