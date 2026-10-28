#!/bin/bash
# Dal PC: scarica dal server i modelli addestrati del tentativo 13 in Modelli/ (ignorata da git) e i risultati in Results/.
#   ./scarica_modelli.sh
cd "$(dirname "$0")"
SSH="ssh"; [ -S ~/.ssh/artes4.sock ] && SSH="ssh -S $HOME/.ssh/artes4.sock"
rsync -av --partial -e "$SSH" fabio.pinelli@10.181.100.6:John/Tentativo13/Modelli/ ../../../../Modelli/13_XAI/
rsync -a -e "$SSH" fabio.pinelli@10.181.100.6:John/Tentativo13/Results/ Results/
du -sh ../../../../Modelli/13_XAI Results
