#!/bin/bash
# Dal PC: a che punto sono gli esperimenti sul server.
#   ./controlla_server.sh          lo stato (coda, fold, stime, GPU)
#   ./controlla_server.sh log      segue il log della coda in diretta (Ctrl+C per uscire)
#   ./controlla_server.sh risultati  la tabella del confronto aggiornata
SERVER=fabio.pinelli@10.181.100.6
case "$1" in
  log)        ssh -t $SERVER 'tail -f ~/John/WebScience2027/Results/2-classificazione/coda.log ~/John/WebScience2027/Results/2-classificazione/avanzamento_*.log' ;;
  risultati)  ssh $SERVER 'cat ~/John/WebScience2027/Results/2-classificazione/03_confronto_intervalli.csv 2>/dev/null | column -s, -t | cut -c1-200 || echo "ancora nessun confronto"' ;;
  *)          ssh $SERVER '~/John/WebScience2027/stato.sh' ;;
esac
