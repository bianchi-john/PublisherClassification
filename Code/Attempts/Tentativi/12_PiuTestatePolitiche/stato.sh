#!/bin/bash
# A che punto è il tentativo 12. Sul server: ~/John/Tentativo12/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"; G=$Q/../../../../Dataset/scraping/NuovoScraping_giro3; R=$Q/Results/3-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora UTC del server: 1 ora indietro d'inverno, 2 d'estate)"
echo; echo "=== screen"; screen -ls | grep -E "t12" || echo "  nessuno: tutto finito (o fermo: guarda le ultime righe qui sotto)"
echo; echo "=== terzo giro di scraping"
[ -f $G/stato_generale.json ] && ~/John/johnVenv/bin/python -c "
import json; s = json.load(open('$G/stato_generale.json'))
print(f\"  {s['testate_finite']}/{s['testate_totali']} testate, {s['articoli_totali']} articoli, fine stimata tra {s['ore_mancanti_stimate']} ore\")
print('  esiti:', dict(list(s['esiti'].items())[:5]))"
echo; echo "=== catena"; tail -n 12 $Q/Results/esegui_tutto.log 2>/dev/null | sed 's/^/  /' || echo "  non ancora partita"
[ -f $R/avanzamento_modernbert_2026_s42.log ] && echo "  ModernBERT: $(grep -c 'finito in' $R/avanzamento_modernbert_2026_s42.log) fold su 5; $(tail -n 1 $R/avanzamento_modernbert_2026_s42.log | cut -c1-110)"
for f in $Q/Results/esecuzioni/*.out; do [ -f "$f" ] && tail -n 40 "$f" | grep -q "Traceback" && { echo; echo "!!! errore recente in $(basename $f):"; grep -A2 "Error" "$f" | grep -v NVML | tail -n 4; }; done
[ -f $Q/Results/2-dataset/01_composizione.csv ] && { echo; echo "=== dataset solo politici"; column -s, -t < $Q/Results/2-dataset/01_composizione.csv | sed 's/^/  /'; }
if [ -f $R/confronto/intervalli.csv ]; then
    echo; echo "=== risultati (per testata, etichette NewsGuard 2026)   aggiornati: $(cat $R/confronto/aggiornato.txt)"
    ~/John/johnVenv/bin/python - $R/confronto <<'PY'
import sys, pandas as pd
d = sys.argv[1]
for _, r in pd.read_csv(f"{d}/intervalli.csv").iterrows():
    ci = "" if pd.isna(r.get("f1_macro 2.5%")) else f" [{r['f1_macro 2.5%']:.3f}, {r['f1_macro 97.5%']:.3f}]"
    print(f"  {r.modello[:36]:36s} {r.testate[:34]:34s} F1 {r.f1_macro:.3f}{ci}  ±1 {r.tolerant_1:.3f}  binario {r.bin_accuratezza:.3f}")
for _, r in pd.read_csv(f"{d}/differenze.csv").iterrows():
    print(f"  {r.confronto[:44]:44s} ΔF1 {r['Δ f1_macro']:+.3f} [{r['Δ f1_macro 2.5%']:+.3f}, {r['Δ f1_macro 97.5%']:+.3f}]  Δ±1 {r['Δ tolerant_1']:+.3f}  Δbin {r['Δ bin_accuratezza']:+.3f}")
PY
fi
echo; echo "=== server: carico $(cut -d' ' -f1-3 /proc/loadavg) su 128 core; GPU $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null); disco $(df -h ~ | tail -1 | awk '{print $4}') liberi"
