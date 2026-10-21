"""Coda della Versione 3 (a parità di genere), in due corsie che girano insieme:

    cd ~/John/WebScience2027/Versione3/2-classificazione
    screen -dmS v3_cpu bash -c 'until ~/John/johnVenv/bin/python coda.py cpu; do sleep 60; done'
    screen -dmS v3_gpu bash -c 'until ~/John/johnVenv/bin/python coda.py gpu; do sleep 60; done'

- corsia cpu: TF-IDF con C = 1 (subito: serve da confronto fold per fold nei log dei transformer), poi con C scelto
  in validazione interna (il riferimento);
- corsia cpu, aggiunti la sera del 20 ottobre: TF-IDF solo sulle parole funzione e senza nomi propri, embedding
  congelati e5 e bge (04_embedding_genere, riusa gli embedding della Versione 2);
- corsia gpu: ModernBERT-base con i seed 42, 1, 2, poi 3 e 4 (l'ensemble lo fa 03_confronto_genere);
- ogni esperimento è un notebook eseguito con papermill in un processo a parte; "fatto" quando c'è il suo FATTO;
  se fallisce si riprova fino a 3 volte, poi si passa al successivo; dopo ogni esperimento si aggiorna il confronto;
- stato in Results/2-classificazione/coda_<corsia>_stato.json, log in coda_<corsia>.log.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUI = Path(__file__).resolve().parent
RESULTS = QUI.parent / "Results" / "2-classificazione"
ESEC = RESULTS / "esecuzioni"
PY = sys.executable
MAX_TENTATIVI = 3
CORSIA = sys.argv[1] if len(sys.argv) > 1 else "cpu"
STATO = RESULTS / f"coda_{CORSIA}_stato.json"
LOG = RESULTS / f"coda_{CORSIA}.log"


def feature(nome, griglia="", testo="intero"):
    return {"nome": nome, "notebook": "01_feature_genere.ipynb", "ore_max": 4, "fatto": RESULTS / nome / "FATTO",
            "parametri": {"NOME": nome, "TESTO": testo, **({"C_GRIGLIA": griglia} if griglia else {})}}


def embedding(nome, emb):
    return {"nome": nome, "notebook": "04_embedding_genere.ipynb", "ore_max": 2, "fatto": RESULTS / nome / "FATTO",
            "parametri": {"NOME": nome, "EMB": emb}}


def transformer(seed):
    nome = f"modernbert_base_genere_s{seed}"
    return {"nome": nome, "notebook": "02_transformer_genere.ipynb", "ore_max": 4, "fatto": RESULTS / "transformer" / nome / "FATTO",
            "parametri": {"NOME": nome, "SEED": seed}}


# la sera del 20 ottobre, dopo il seed 42: le varianti del testo (stile o contenuto?), gli embedding congelati e altri
# due seed di ModernBERT (ensemble di 5, più stabile)
ESPERIMENTI = {"cpu": [feature("feature_genere"), feature("feature_genere_tuned", "0.1,1,10,100"),
                       feature("feature_genere_solo_funzione", testo="solo_funzione"),
                       feature("feature_genere_senza_nomi", testo="senza_nomi"),
                       embedding("e5_genere", "e5_large_v2"), embedding("bge_genere", "bge_large_v15")],
               "gpu": [transformer(s) for s in (42, 1, 2, 3, 4)]}[CORSIA]


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
    ambiente = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    with open(file_out, "a", encoding="utf-8") as f:
        f.write(f"\n===== {ora()} {' '.join(cmd)}\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=ambiente)
        try:
            return p.wait(timeout=ore_max * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); return "tempo scaduto"


def main():
    ESEC.mkdir(parents=True, exist_ok=True)
    stato = json.loads(STATO.read_text()) if STATO.exists() else {"esperimenti": {}}
    scrivi_log(f"=== corsia {CORSIA}: {len(ESPERIMENTI)} esperimenti ===")
    for e in ESPERIMENTI:
        s = stato["esperimenti"].setdefault(e["nome"], {"stato": "in attesa", "tentativi": 0})
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
        if e["fatto"].exists() and s["stato"] != "fatto":
            s["stato"] = "fatto"; salva(stato)
        esito = lancia("03_confronto_genere.ipynb", ESEC / f"03_confronto_genere_{CORSIA}.ipynb", {}, 0.5, ESEC / "03_confronto_genere.out")
        scrivi_log(f"confronto aggiornato (esito {esito})")
    scrivi_log(f"=== corsia {CORSIA} finita ===")


if __name__ == "__main__":
    main()
