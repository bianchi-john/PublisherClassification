#!/bin/bash
# Dal PC: a che punto è la Versione 2 sul server.
#   ./controlla_server.sh             lo stato (catena, coda, fold, stime, GPU)
#   ./controlla_server.sh log         segue i log in diretta (Ctrl+C per uscire)
#   ./controlla_server.sh risultati   il confronto aggiornato (con gli intervalli) e la tabella Versione 1 contro 2
SERVER=fabio.pinelli@10.181.100.6
R='~/John/WebScience2027/Tentativi/06_Versione2_PiuPuliziaPiuTraining/Results'
case "$1" in
  log)        ssh -t $SERVER "tail -f $R/esegui_tutto/esegui_tutto.log $R/2-classificazione/coda.log $R/2-classificazione/avanzamento_*.log" ;;
  risultati)  ssh $SERVER "for f in $R/2-classificazione/03_confronto_intervalli.csv $R/2-classificazione/03_confronto_versione1.csv; do echo; echo \"== \$(basename \$f)\"; column -s, -t < \$f 2>/dev/null | cut -c1-220 || echo 'ancora niente'; done" ;;
  *)          ssh $SERVER '~/John/WebScience2027/Tentativi/06_Versione2_PiuPuliziaPiuTraining/stato.sh' ;;
esac
