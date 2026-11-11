#!/bin/bash
# A che punto è la Versione 3 (a parità di genere). Sul server: ~/John/WebScience2027/Tentativi/07_Versione3_PariGenere/stato.sh
R=~/John/WebScience2027/Tentativi/07_Versione3_PariGenere/Results/2-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora del server, UTC: 2 ore indietro)"
echo; echo "=== screen della Versione 3"; screen -ls | grep -E "v3_" || echo "  nessuno"
for c in cpu gpu; do
    echo; echo "=== corsia $c"; tail -4 $R/coda_$c.log 2>/dev/null | sed 's/^/  /'
done
echo; echo "=== fold finiti dei transformer (con il TF-IDF sullo stesso fold)"
grep -h "finito in\|FINITO" $R/avanzamento_modernbert_base_genere_s*.log 2>/dev/null | sed 's/^/  /'
echo; echo "=== TF-IDF ed embedding (FINITO = risultato su tutte le 304 testate)"; grep -h "FINITO" $R/avanzamento_feature_genere*.log $R/avanzamento_*_genere.log 2>/dev/null | sort -u | sed 's/^/  /'
echo; echo "=== ultimi passi"; tail -3 "$(ls -t $R/avanzamento_*.log 2>/dev/null | head -1)" 2>/dev/null | sed 's/^/  /'
echo; echo "=== GPU"
~/John/johnVenv/bin/python -c "import torch; f,t = torch.cuda.mem_get_info(); print(f'  memoria libera {f/2**30:.1f} / {t/2**30:.0f} GB')" 2>/dev/null
