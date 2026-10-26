#!/bin/bash
# Dal PC: ./controlla_server.sh      lo stato del tentativo 14 sul server
SERVER=fabio.pinelli@10.181.100.6
[ -S ~/.ssh/artes4.sock ] && SERVER="-S $HOME/.ssh/artes4.sock $SERVER"
ssh $SERVER '~/John/Tentativo14/stato.sh'
