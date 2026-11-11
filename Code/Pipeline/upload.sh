#!/bin/bash
# From the PC: copy the code of attempt 20 to the server (~/John/Tentativo20). The data are built there from scratch.
# Notebooks are NOT copied (the server keeps its executed ones); "./upload.sh notebooks" copies them too.
cd "$(dirname "$0")"
EXTRA=(--exclude "*.ipynb"); [ "$1" = "notebooks" ] && EXTRA=()
rsync -av -e "ssh -o BatchMode=yes -i $HOME/.ssh/id_ed25519_lumina_vm" --exclude "Results/" --exclude "__pycache__/" \
      "${EXTRA[@]}" ./ fabio.pinelli@10.181.100.6:John/Tentativo20/ | tail -n 2
