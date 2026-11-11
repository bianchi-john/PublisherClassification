#!/bin/bash
# A che punto è il tentativo 11 (coda del weekend). Sul server: ~/John/Tentativo11/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"; R=$Q/Results/3-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora del server, UTC: 2 ore indietro)"
echo; echo "=== screen (t11_cpu: classificatore a feature; t11_gpu: ModernBERT)"; screen -ls | grep -E "t11" || echo "  nessuno: le corsie sono finite"
echo; echo "=== esperimenti"
~/John/johnVenv/bin/python - $R <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
for corsia in ["cpu", "gpu"]:
    f = R / f"coda_{corsia}_stato.json"
    if not f.exists():
        print(f"  {corsia}: non ancora partita"); continue
    for nome, s in json.loads(f.read_text())["esperimenti"].items():
        extra = ""
        if nome.startswith("modernbert") and s["stato"] != "fatto":
            log = R / f"avanzamento_{nome}.log"
            if log.exists():
                righe = log.read_text().splitlines()
                extra = f"  [{sum('finito in' in r for r in righe)} fold su 5; ultimo: {righe[-1][20:110] if righe else ''}]"
        print(f"  {corsia}  {nome:34s} {s['stato']:12s} tentativi {s['tentativi']}  {s.get('minuti', '')} min{extra}")
PY
echo; echo "=== ultime righe delle corsie"; tail -q -n 3 $R/coda_cpu.log $R/coda_gpu.log 2>/dev/null | sed 's/^/  /'
for f in $R/esecuzioni/*.out; do [ -f "$f" ] && tail -n 30 "$f" | grep -q "Traceback" && { echo; echo "!!! errore recente in $(basename $f):"; grep -A2 "Error" "$f" | grep -v NVML | tail -n 4; }; done
if [ -f $R/confronto/intervalli.csv ]; then
    echo; echo "=== risultati (per testata; tutte = 296 valutate, politici = 143)   aggiornati: $(cat $R/confronto/aggiornato.txt 2>/dev/null)"
    ~/John/johnVenv/bin/python - $R/confronto <<'PY'
import sys, pandas as pd
d = sys.argv[1]
for _, r in pd.read_csv(f"{d}/intervalli.csv").iterrows():
    print(f"  {r.variante:8s} {r.modello[:16]:16s} addestr. {r['addestrato su']} valut. {r['valutato su']} {r.testate:8s}: "
          f"F1 {r.f1_macro:.3f} [{r['f1_macro 2.5%']:.3f}, {r['f1_macro 97.5%']:.3f}]  binario {r.bin_accuratezza:.3f}")
try:
    for _, r in pd.read_csv(f"{d}/differenze.csv").iterrows():
        print(f"  {r.confronto[:70]:70s} ΔF1 {r['Δ f1_macro']:+.3f} [{r['Δ f1_macro 2.5%']:+.3f}, {r['Δ f1_macro 97.5%']:+.3f}]  Δbin {r['Δ bin_accuratezza']:+.3f}")
except Exception:
    pass
PY
fi
echo; echo "=== server: carico $(cut -d' ' -f1-3 /proc/loadavg) su 128 core; GPU $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null)"
