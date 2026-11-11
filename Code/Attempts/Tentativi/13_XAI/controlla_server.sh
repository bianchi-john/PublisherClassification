#!/bin/bash
# Dal PC: ./controlla_server.sh      lo stato del tentativo 13 sul server
SERVER=fabio.pinelli@10.181.100.6
[ -S ~/.ssh/artes4.sock ] && SERVER="-S $HOME/.ssh/artes4.sock $SERVER"   # riusa la connessione già aperta, se c'è
ssh $SERVER '~/John/Tentativo13/stato.sh'
