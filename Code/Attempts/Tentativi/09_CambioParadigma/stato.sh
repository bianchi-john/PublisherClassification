#!/bin/bash
# A che punto è il tentativo 09. Sul server: ~/John/Tentativo09/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"; R=$Q/Results/3-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora della macchina: sul server UTC, 2 ore indietro)"
echo; echo "=== screen del tentativo 09"; screen -ls | grep -E "t09" || echo "  nessuno: tutto finito (o fermo: guarda le ultime righe qui sotto)"

echo; echo "=== 1. catena principale (riferimento TF-IDF, modelli, confronto)"; sed 's/^/  /' $Q/Results/esegui_tutto.log 2>/dev/null | tail -n 3
grep -h "fold [0-9] combinato\|criteri fatti\|finito in" $R/modelli/avanzamento.log 2>/dev/null | sed 's/^/  /' | grep . || echo "  modelli non ancora partiti"

echo; echo "=== 2. coda della notte (ModernBERT semi 42, 1, 2: circa 4 ore ciascuno; dopo ogni seme la combinazione)"
tail -n 6 $Q/Results/coda_notte.log 2>/dev/null | sed 's/^/  /' | grep . || echo "  non ancora partita"
for f in $R/modernbert/avanzamento_modernbert_s*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c ": fatto in" "$f"); echo "  $(basename $f .log | sed 's/avanzamento_//'): $n addestramenti su 20; ultimo: $(grep -v '^ *$' "$f" | tail -n 1 | cut -c1-150)"
done
grep -h "GPU occupata" $R/modernbert/avanzamento_*.log 2>/dev/null | tail -n 1 | sed 's/^/  ATTESA GPU: /'

for f in $R/esecuzioni/*.out; do [ -f "$f" ] && grep -q "Traceback" "$f" && { echo; echo "!!! errore in $(basename $f):"; grep -B1 -A2 "Error" "$f" | grep -v NVML | tail -n 5; }; done

if [ -f $R/confronto/intervalli.csv ]; then
    echo; echo "=== risultati (F1 a 5 livelli e binario, per testata, 301 testate valutate; intervalli al 95% per gruppo)"
    ~/John/johnVenv/bin/python - $R/confronto/intervalli.csv <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1], index_col=0)
for m, r in t.iterrows():
    d = "" if pd.isna(r.get("ΔF1 vs riferimento")) else f"   ΔF1 {r['ΔF1 vs riferimento']:+.3f} [{r['ΔF1 2.5%']:+.3f}, {r['ΔF1 97.5%']:+.3f}]"
    print(f"  {m[:45]:45s} F1 {r['F1 macro']:.3f} [{r['F1 2.5%']:.3f}, {r['F1 97.5%']:.3f}]  bin {r['binario']:.3f}{d}")
PY
fi
echo; echo "=== carico del server (128 core): $(cut -d' ' -f1-3 /proc/loadavg); GPU: $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null)"
