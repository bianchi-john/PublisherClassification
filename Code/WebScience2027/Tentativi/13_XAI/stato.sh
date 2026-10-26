#!/bin/bash
# A che punto è il tentativo 13. Sul server: ~/John/Tentativo13/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"; R=$Q/Results/2-modelli
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora UTC del server: 1 ora indietro d'inverno, 2 d'estate)"
echo; echo "=== screen"; screen -ls | grep -E "t13" || echo "  nessuno: finito (o fermo: guarda le ultime righe qui sotto)"
echo; echo "=== catena"; tail -n 10 $Q/Results/esegui.log 2>/dev/null | sed 's/^/  /'
[ -f $R/avanzamento_feature_tuned_2026.log ] && echo "  feature: $(grep -c 'fatto in' $R/avanzamento_feature_tuned_2026.log) fold su 5; $(grep -h 'modello finale' $R/avanzamento_feature_tuned_2026.log | cut -c1-100)"
[ -f $R/avanzamento_modernbert_2026_s42.log ] && echo "  ModernBERT: $(grep -c 'finito in' $R/avanzamento_modernbert_2026_s42.log) fold su 5; ultimo: $(tail -n 1 $R/avanzamento_modernbert_2026_s42.log | cut -c1-110)"
X=$Q/Results/3-xai
[ -d $X ] && { echo; echo "=== XAI (fase 2)"; ls $X | grep -vE "^prova$" | sed 's/^/  /' | head -12; }
[ -f $X/analisi/categorie.csv ] && ~/John/johnVenv/bin/python - $X/analisi <<'PY'
import sys, pandas as pd
c = pd.read_csv(f"{sys.argv[1]}/categorie.csv"); c = c[(c.testate == "tutte") & (c.misura == "quota")]
print("  quota di importanza per categoria (tutte le testate):")
print(c.pivot(index="categoria", columns="modello", values="media").round(3).to_string().replace("\n", "\n    "))
PY
[ -f $X/maschere/maschere.csv ] && ~/John/johnVenv/bin/python - $X/maschere <<'PY'
import sys, pandas as pd
m = pd.read_csv(f"{sys.argv[1]}/maschere.csv"); m = m[m.testate == "tutte"]
print("  maschere, F1 (variazione rispetto all'originale):")
for _, r in m.iterrows(): print(f"    {r.modello:10s} {r.condizione:32s} F1 {r.f1_macro:.3f} ({r['variazione f1_macro']:+.3f})  binario {r.bin_accuratezza:.3f}")
PY
echo; echo "=== modelli salvati"; (cd $Q/Modelli 2>/dev/null && find . -maxdepth 2 \( -name "*.joblib" -o -name "model.safetensors" \) | sort | sed 's/^/  /'; du -sh $Q/Modelli 2>/dev/null | sed 's/^/  in tutto /')
for f in $Q/Results/esecuzioni/*.out; do [ -f "$f" ] && tail -n 40 "$f" | grep -q "Traceback" && { echo; echo "!!! errore recente in $(basename $f):"; grep -A2 "Error" "$f" | grep -v NVML | tail -n 4; }; done
if [ -f $R/confronto/intervalli.csv ]; then
    echo; echo "=== risultati (per testata, etichette NewsGuard 2026)   aggiornati: $(cat $R/confronto/aggiornato.txt)"
    ~/John/johnVenv/bin/python - $R/confronto <<'PY'
import sys, pandas as pd
d = sys.argv[1]
for _, r in pd.read_csv(f"{d}/intervalli.csv").iterrows():
    ci = "" if pd.isna(r.get("f1_macro 2.5%")) else f" [{r['f1_macro 2.5%']:.3f}, {r['f1_macro 97.5%']:.3f}]"
    print(f"  {r.modello[:28]:28s} F1 {r.f1_macro:.3f}{ci}  ±1 {r.tolerant_1:.3f}  binario {r.bin_accuratezza:.3f}")
for _, r in pd.read_csv(f"{d}/differenze.csv").iterrows():
    print(f"  {r.confronto[:44]:44s} ΔF1 {r['Δ f1_macro']:+.3f} [{r['Δ f1_macro 2.5%']:+.3f}, {r['Δ f1_macro 97.5%']:+.3f}]  Δbin {r['Δ bin_accuratezza']:+.3f}")
print("  F1 per fold:"); print(pd.read_csv(f"{d}/per_fold.csv").pivot(index="fold", columns="modello", values="f1_macro").round(3).to_string().replace("\n", "\n    "))
PY
fi
echo; echo "=== server: carico $(cut -d' ' -f1-3 /proc/loadavg); GPU $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null); disco $(df -h ~ | tail -1 | awk '{print $4}') liberi"
