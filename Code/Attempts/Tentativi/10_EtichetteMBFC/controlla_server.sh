#!/bin/bash
# Dal PC: ./controlla_server.sh      lo stato della corsa del tentativo 10 sul server
SERVER=fabio.pinelli@10.181.100.6
[ -S ~/.ssh/artes4.sock ] && SERVER="-S $HOME/.ssh/artes4.sock $SERVER"   # riusa la connessione già aperta, se c'è
ssh $SERVER '~/John/Tentativo10/stato.sh'
