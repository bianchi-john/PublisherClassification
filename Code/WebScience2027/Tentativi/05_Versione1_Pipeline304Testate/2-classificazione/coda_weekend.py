"""Coda degli esperimenti: li lancia uno dopo l'altro, riprova, salta, registra.

    cd ~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate/2-classificazione
    screen -dmS coda bash -c 'until ~/John/johnVenv/bin/python coda_weekend.py; do sleep 60; done'

- ogni esperimento è un notebook eseguito con papermill in un processo a parte: quando finisce (o muore) la GPU
  torna libera;
- un esperimento è "fatto" quando il notebook ha scritto il suo file FATTO; se viene rilanciato riparte dal
  fold a cui era arrivato;
- se fallisce si riprova fino a MAX_TENTATIVI volte, poi si passa al successivo;
- prima di un esperimento su GPU si aspetta che la memoria libera basti (il server è condiviso);
- dopo ogni esperimento si riesegue 03_confronto: i risultati parziali sono sempre aggiornati;
- stato in Results/2-classificazione/coda_stato.json, log in coda.log. Rilanciare lo script è sicuro: salta
  quello che è fatto e ricorda i tentativi.
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
STATO = RESULTS / "coda_stato.json"
LOG = RESULTS / "coda.log"
PY = sys.executable
MAX_TENTATIVI = 3
ATTESA_RIPROVA = 300            # secondi tra un tentativo fallito e il successivo
ATTESA_GPU_MAX = 6 * 3600       # dopo tanto tempo si prova comunque


def transformer(nome, modello, max_token, batch, gb, ore, seed=42, **altri):
    return {"nome": f"{nome}_s{seed}", "notebook": "02_transformer.ipynb", "gpu_gb": gb, "ore_max": ore,
            "fatto": RESULTS / "transformer" / f"{nome}_s{seed}" / "FATTO",
            "parametri": {"NOME": f"{nome}_s{seed}", "MODELLO": modello, "MAX_TOKEN": max_token, "BATCH": batch, "SEED": seed, **altri}}


def embedding(nome, modello, prefisso, pooling):
    return {"nome": nome, "notebook": "04_embedding.ipynb", "gpu_gb": 8, "ore_max": 3,
            "fatto": RESULTS / "transformer" / nome / "FATTO",
            "parametri": {"NOME": nome, "MODELLO_EMB": modello, "PREFISSO": prefisso, "POOLING": pooling}}


# l'ordine conta: prima i riferimenti e i sistemi principali, in fondo i più lunghi (se il tempo finisce, mancano quelli)
ESPERIMENTI = [
    {"nome": "feature", "notebook": "01_classificatore_feature.ipynb", "gpu_gb": 0, "ore_max": 3,
     "fatto": RESULTS / "feature_FATTO", "parametri": {"NOME": "feature"}},
    {"nome": "feature_tuned", "notebook": "01_classificatore_feature.ipynb", "gpu_gb": 0, "ore_max": 6,
     "fatto": RESULTS / "feature_tuned_FATTO", "parametri": {"NOME": "feature_tuned", "C_GRIGLIA": "0.1,1,10,100"}},
    embedding("e5_large_v2", "intfloat/e5-large-v2", "query: ", "media"),
    embedding("bge_large_v15", "BAAI/bge-large-en-v1.5", "", "cls"),
] + [transformer("modernbert_base_1024_ep2", "answerdotai/ModernBERT-base", 1024, 16, 16, 6, seed=s, PATIENCE=0, EPOCHE_MAX=2)
     for s in (42, 1, 2)] \
  + [transformer("modernbert_large_1024_ep2", "answerdotai/ModernBERT-large", 1024, 4, 20, 10, seed=s, PATIENCE=0, EPOCHE_MAX=2)
     for s in (42, 1, 2)]


def ora():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scrivi_log(m):
    riga = f"{ora()}  {m}"
    print(riga, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(riga + "\n")


def leggi_stato():
    if STATO.exists():
        return json.loads(STATO.read_text())
    return {"esperimenti": {e["nome"]: {"stato": "in attesa", "tentativi": 0} for e in ESPERIMENTI}}


def salva_stato(stato):
    stato["aggiornato"] = ora()
    tmp = STATO.with_suffix(".tmp")
    tmp.write_text(json.dumps(stato, indent=1, ensure_ascii=False))
    os.replace(tmp, STATO)


def gpu_libera_gb():
    try:
        r = subprocess.run([PY, "-c", "import torch; print(torch.cuda.mem_get_info()[0] / 2**30)"],
                           capture_output=True, text=True, timeout=120)
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return -1.0


def aspetta_gpu(gb):
    inizio = time.time()
    while True:
        libera = gpu_libera_gb()
        if libera >= gb or time.time() - inizio > ATTESA_GPU_MAX:
            scrivi_log(f"GPU: {libera:.1f} GB liberi (servono {gb})")
            return
        scrivi_log(f"GPU occupata da altri: {libera:.1f} GB liberi, ne servono {gb}; riprovo tra 5 minuti")
        time.sleep(300)


def lancia(notebook, uscita, parametri, ore_max, file_out):
    """papermill in un gruppo di processi a parte, così allo scadere si uccidono anche i figli (il kernel)."""
    cmd = [PY, "-m", "papermill", "-k", "johnvenv", "--cwd", str(QUI), str(QUI / notebook), str(uscita)]
    for k, v in parametri.items():
        cmd += ["-p", k, str(v)]
    with open(file_out, "a", encoding="utf-8") as f:
        f.write(f"\n===== {ora()} {' '.join(cmd)}\n"); f.flush()
        ambiente = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=ambiente)
        try:
            return p.wait(timeout=ore_max * 3600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "tempo scaduto"


def main():
    ESEC.mkdir(parents=True, exist_ok=True)
    stato = leggi_stato()
    for e in ESPERIMENTI:
        stato["esperimenti"].setdefault(e["nome"], {"stato": "in attesa", "tentativi": 0})
    salva_stato(stato)
    scrivi_log(f"=== coda avviata: {len(ESPERIMENTI)} esperimenti ===")
    for e in ESPERIMENTI:
        s = stato["esperimenti"][e["nome"]]
        if e["fatto"].exists():
            if s["stato"] != "fatto":
                s.update(stato="fatto"); salva_stato(stato)
            continue
        while s["tentativi"] < MAX_TENTATIVI and not e["fatto"].exists():
            if e["gpu_gb"]:
                aspetta_gpu(e["gpu_gb"])
            s["tentativi"] += 1
            s.update(stato="in corso", inizio=ora())
            stato["in_corso"] = e["nome"]; salva_stato(stato)
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
                    time.sleep(ATTESA_RIPROVA)
            stato["in_corso"] = None; salva_stato(stato)
        if not e["fatto"].exists():
            scrivi_log(f"--- {e['nome']}: troppi tentativi falliti, passo al successivo")
        esito = lancia("03_confronto.ipynb", ESEC / "03_confronto.ipynb", {}, 0.5, ESEC / "03_confronto.out")
        scrivi_log(f"confronto aggiornato (esito {esito})")
    scrivi_log("=== coda finita ===")


if __name__ == "__main__":
    main()
