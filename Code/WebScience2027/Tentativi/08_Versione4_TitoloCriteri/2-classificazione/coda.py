"""Coda della Versione 4 (titolo + articolo, criteri NewsGuard), in corsie che girano insieme:

    cd ~/John/WebScience2027/Tentativi/08_Versione4_TitoloCriteri/2-classificazione
    screen -dmS v4_<corsia> bash -c 'until ~/John/johnVenv/bin/python coda.py <corsia>; do sleep 60; done'

Corsie delle prove preliminari (21 ottobre, sulla sola divisione di sviluppo: il fold 1 non si guarda mai):
- sviluppo_cpu: TF-IDF con e senza titolo;
- sviluppo_gpu: ModernBERT solo articolo, titolo + articolo, titolo + articolo + criteri (peso 0,5 e 1,0), seed 42 e
  poi seed 1.
Corsie della notte (21 ottobre): notte_cpu (TF-IDF con il titolo, C = 1 e C in validazione), notte_gpu
(ModernBERT titolo + articolo e titolo + articolo + criteri con peso 0,5, seed 42, 1, 2), sui 5 fold.

Ogni esperimento è un notebook eseguito con papermill in un processo a parte; "fatto" quando c'è il suo FATTO; se
fallisce si riprova fino a 3 volte, poi si passa al successivo. Stato in Results/2-classificazione/coda_<corsia>_stato.json,
log in coda_<corsia>.log.
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
CORSIA = sys.argv[1] if len(sys.argv) > 1 else "sviluppo_cpu"
STATO = RESULTS / f"coda_{CORSIA}_stato.json"
LOG = RESULTS / f"coda_{CORSIA}.log"


SVIL = RESULTS / "sviluppo"


def feature(nome, titolo, sviluppo, griglia=""):
    cartella = (SVIL if sviluppo else RESULTS) / nome
    return {"nome": nome, "notebook": "01_feature_titolo.ipynb", "ore_max": 4, "fatto": cartella / "FATTO",
            "parametri": {"NOME": nome, "TITOLO": titolo, "SVILUPPO": sviluppo, **({"C_GRIGLIA": griglia} if griglia else {})}}


def bert(nome, seed, titolo, lam, sviluppo):
    nome = f"{nome}_s{seed}"
    cartella = (SVIL if sviluppo else RESULTS) / "transformer" / nome
    return {"nome": nome, "notebook": "02_transformer_titolo.ipynb", "ore_max": 4 if not sviluppo else 1.5,
            "fatto": cartella / "FATTO",
            "parametri": {"NOME": nome, "SEED": seed, "TITOLO": titolo, "LAMBDA_CRITERI": lam, "SVILUPPO": sviluppo,
                          "BATCH": 4, "PATIENCE": 0, "EPOCHE_MAX": 2}}


PROVE = [("bert_solo_articolo", False, 0.0), ("bert_titolo", True, 0.0), ("bert_titolo_criteri05", True, 0.5), ("bert_titolo_criteri10", True, 1.0)]
ESPERIMENTI = {
    "sviluppo_cpu": [feature("feature_solo_articolo", False, True), feature("feature_titolo", True, True)],
    "sviluppo_gpu": [bert(n, s, t, l, True) for s in (42, 1) for n, t, l in PROVE],
    # la notte del 21 ottobre: i 5 fold, con la ricetta decisa a priori (le prove di sviluppo, 26 testate, non
    # discriminano: hanno solo confermato che tutto funziona). Riferimento sul solo articolo: la Versione 2.
    "notte_cpu": [feature("feature_titolo", True, False), feature("feature_titolo_tuned", True, False, "0.1,1,10,100")],
    "notte_gpu": [bert("bert_titolo", s, True, 0.0, False) for s in (42, 1, 2)]
                 + [bert("bert_titolo_criteri05", s, True, 0.5, False) for s in (42, 1, 2)],
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
        if Path(QUI / "03_confronto.ipynb").exists() and not CORSIA.startswith("sviluppo"):
            esito = lancia("03_confronto.ipynb", ESEC / f"03_confronto_{CORSIA}.ipynb", {}, 0.5, ESEC / "03_confronto.out")
            scrivi_log(f"confronto aggiornato (esito {esito})")
    scrivi_log(f"=== corsia {CORSIA} finita ===")


if __name__ == "__main__":
    main()
