#!/bin/bash
# A che punto è il tentativo 10. Sul server: ~/John/Tentativo10/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"; R=$Q/Results/3-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora del server, UTC: 2 ore indietro)"
echo; echo "=== screen"; screen -ls | grep -E "t10" || echo "  nessuno: finito (o fermo: guarda le ultime righe qui sotto)"
echo; echo "=== catena"; sed 's/^/  /' $Q/Results/esegui_tutto.log 2>/dev/null | tail -n 8
echo; echo "=== classificatore a feature (5 fold in parallelo)"
for v in grezzo pulito; do f=$R/avanzamento_feature_tuned_$v.log; [ -f $f ] && { echo "  $v:"; grep -h "fatto in" $f | sed 's/^/    /'; }; done
echo; echo "=== ModernBERT (5 fold, circa 15 minuti ciascuno)"
for v in grezzo pulito; do f=$R/avanzamento_modernbert_${v}_s42.log; [ -f $f ] && { echo "  $v:"; grep -h "finito in\|FINITO\|GPU occupata" $f | tail -n 6 | sed 's/^/    /'; }; done
for f in $R/esecuzioni/*.out; do [ -f "$f" ] && grep -q "Traceback" "$f" && { echo; echo "!!! errore in $(basename $f):"; grep -A2 "Error" "$f" | grep -v NVML | tail -n 5; }; done
if [ -f $R/confronto/intervalli.csv ]; then
    echo; echo "=== risultati (per testata, 222 testate valutate)"
    ~/John/johnVenv/bin/python - $R/confronto <<'PY'
import sys, pandas as pd
d = sys.argv[1]
t = pd.read_csv(f"{d}/intervalli.csv")
for _, r in t.iterrows():
    ci = "" if pd.isna(r.get("f1_macro 2.5%")) else f" [{r['f1_macro 2.5%']:.3f}, {r['f1_macro 97.5%']:.3f}]"
    print(f"  {r.modello:22s} {str(r.versione):7s} F1 {r.f1_macro:.3f}{ci}  binario {r.bin_accuratezza:.3f}")
try:
    for _, r in pd.read_csv(f"{d}/differenze.csv").iterrows():
        print(f"  {r.confronto:32s} ΔF1 {r['Δ f1_macro']:+.3f} [{r['Δ f1_macro 2.5%']:+.3f}, {r['Δ f1_macro 97.5%']:+.3f}]  Δbin {r['Δ bin_accuratezza']:+.3f}")
except FileNotFoundError:
    pass
PY
fi
echo; echo "=== server: carico $(cut -d' ' -f1-3 /proc/loadavg) su 128 core; GPU $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null)"
