#!/bin/bash
# A che punto è la Versione 4 (tentativo 08). Sul server: ~/John/WebScience2027/Tentativi/08_Versione4_TitoloCriteri/stato.sh
# la corsa del 21 ottobre gira ancora nella vecchia posizione ~/John/WebScience2027/Versione4: se il percorso nuovo non c'è, si usa quella
R=~/John/WebScience2027/Tentativi/08_Versione4_TitoloCriteri/Results/2-classificazione
[ -d $R ] || R=~/John/WebScience2027/Versione4/Results/2-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora del server, UTC: 2 ore indietro)"
echo; echo "=== screen della Versione 4"; screen -ls | grep -E "v4_" || echo "  nessuno"
for c in notte_cpu notte_gpu; do echo; echo "=== corsia $c"; tail -n 3 $R/coda_$c.log 2>/dev/null | sed 's/^/  /'; done
echo; echo "=== ModernBERT, fold per fold (titolo + articolo; con _criteri05 anche i criteri)"
for f in $R/avanzamento_bert_*.log; do [ -f "$f" ] && { echo "  $(basename $f .log | sed 's/avanzamento_//')"; grep -h "finito in\|FINITO" $f | sed 's/^/    /'; }; done
echo; echo "=== TF-IDF con il titolo"; grep -h "fold .*: C\|FINITO" $R/avanzamento_feature_titolo*.log 2>/dev/null | sed 's/^/  /'
echo; echo "=== riferimento, ModernBERT solo articolo (Versione 2), F1 per testata fold per fold"
echo "    fold:    1     2     3     4     5"
echo "    s42:   0.436 0.291 0.363 0.347 0.326"
echo "    s1:    0.308 0.268 0.270 0.412 0.335"
echo "    s2:    0.446 0.318 0.325 0.315 0.358"
echo; echo "=== GPU"
~/John/johnVenv/bin/python -c "import torch; f,t = torch.cuda.mem_get_info(); print(f'  memoria libera {f/2**30:.1f} / {t/2**30:.0f} GB')" 2>/dev/null
