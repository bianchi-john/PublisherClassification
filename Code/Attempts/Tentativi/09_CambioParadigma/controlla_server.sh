#!/bin/bash
# Dal PC: ./controlla_server.sh            lo stato della corsa sul server
#         ./controlla_server.sh risultati  il confronto con gli intervalli (alla fine della catena)
SERVER=fabio.pinelli@10.181.100.6; T='~/John/Tentativo09'
[ -S ~/.ssh/artes4.sock ] && SERVER="-S $HOME/.ssh/artes4.sock $SERVER"   # riusa la connessione già aperta, se c'è
case "$1" in
  risultati) ssh $SERVER "cd $T/Results/3-classificazione/confronto 2>/dev/null && for f in *.csv; do echo \"=== \$f\"; column -s, -t < \$f | cut -c1-180 | head -n 25; done || echo 'ancora niente'" ;;
  *)         ssh $SERVER "$T/stato.sh" ;;
esac
