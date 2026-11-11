#!/bin/bash
# From the PC: bring back the executed notebooks and logs, and the result tables and figure (to Dataset/, outside git).
# With "all" also the article outputs (predictions/). The models stay on the server.  Usage: ./download.sh [all]
cd "$(dirname "$0")"
R="ssh -o BatchMode=yes -i $HOME/.ssh/id_ed25519_lumina_vm"; S=fabio.pinelli@10.181.100.6
rsync -av -e "$R" --include "*/" --include "*.ipynb" --include "*.log" --exclude "*" $S:John/Tentativo20/ ./ | tail -n 2
D=../../../../Dataset/tentativi/20_FromScratch; mkdir -p $D
rsync -av -e "$R" --include "*.csv" --include "*.png" --include "*.txt" --exclude "*" $S:John/Dataset/tentativi/20_FromScratch/ $D/ | tail -n 2
[ "$1" = "all" ] && rsync -av -e "$R" $S:John/Dataset/tentativi/20_FromScratch/predictions $D/ | tail -n 2
