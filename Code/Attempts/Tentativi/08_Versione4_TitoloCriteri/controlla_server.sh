#!/bin/bash
# Dal PC: ./controlla_server.sh            lo stato (con i risultati fold per fold)
#         ./controlla_server.sh risultati  il confronto con gli intervalli (si aggiorna dopo ogni esperimento)
SERVER=fabio.pinelli@10.181.100.6
N='~/John/WebScience2027/Tentativi/08_Versione4_TitoloCriteri'; V='~/John/WebScience2027/Versione4'      # V: dove gira la corsa del 21 ottobre
case "$1" in
  risultati) ssh $SERVER "R=$N/Results/2-classificazione; [ -d \$R ] || R=$V/Results/2-classificazione; column -s, -t < \$R/03_confronto_intervalli.csv 2>/dev/null | cut -c1-200 || echo 'ancora niente'" ;;
  *)         ssh $SERVER "S=$N/stato.sh; [ -x \$S ] || S=$V/stato.sh; \$S" ;;
esac
