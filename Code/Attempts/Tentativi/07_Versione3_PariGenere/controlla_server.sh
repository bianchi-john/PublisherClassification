#!/bin/bash
# Dal PC: ./controlla_server.sh            lo stato (con i risultati fold per fold)
#         ./controlla_server.sh risultati  il confronto con gli intervalli (quando c'è)
SERVER=fabio.pinelli@10.181.100.6
R='~/John/WebScience2027/Tentativi/07_Versione3_PariGenere/Results/2-classificazione'
case "$1" in
  risultati) ssh $SERVER "for f in $R/03_genere_intervalli.csv $R/03_genere_metriche.csv; do echo; echo \"== \$(basename \$f)\"; column -s, -t < \$f 2>/dev/null | cut -c1-230 || echo 'ancora niente'; done" ;;
  *)         ssh $SERVER '~/John/WebScience2027/Tentativi/07_Versione3_PariGenere/stato.sh' ;;
esac
