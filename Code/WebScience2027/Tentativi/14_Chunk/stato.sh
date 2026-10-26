#!/bin/bash
# A che punto è il tentativo 14. Sul server: ~/John/Tentativo14/stato.sh (dal PC: ./controlla_server.sh)
Q="$(cd "$(dirname "$0")" && pwd)"
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')"
screen -ls | grep -q "\.t14\s" && echo "=== in corso (screen t14)" || echo "=== nessuno screen t14: finito o fermo"
echo; tail -n 4 $Q/Results/esegui.log 2>/dev/null
echo; echo "=== ultime righe di avanzamento (ModernBERT)"; tail -n 6 $Q/Results/avanzamento.log 2>/dev/null
echo; echo "=== passi salvati"; ls $Q/Results/cache 2>/dev/null | sed 's/^/  /'; for d in modernbert_troncato modernbert_pezzi; do [ -d $Q/Results/cache/$d ] && echo "  $d: $(ls $Q/Results/cache/$d | tr '\n' ' ')"; done
tail -n 30 $Q/Results/esegui.out 2>/dev/null | grep -q "Traceback\|Error" && { echo; echo "!!! errore:"; grep -B2 -A6 "Error" $Q/Results/esegui.out | grep -v NVML | tail -n 10; }
[ -f $Q/Results/confronto.csv ] && { echo; echo "=== risultati"; ~/John/johnVenv/bin/python -c "
import pandas as pd; t = pd.read_csv('$Q/Results/confronto.csv', index_col=0)
print(t[['F1 macro', 'F1 macro IC', 'tolerant ±1', 'binario']].round(3).to_string())"; }
echo; echo "=== server: carico $(cut -d' ' -f1-3 /proc/loadavg); GPU $(~/John/johnVenv/bin/python -c 'import torch; f,t=torch.cuda.mem_get_info(); print(f"{f/2**30:.1f} GB liberi su {t/2**30:.0f}")' 2>/dev/null)"
