#!/bin/bash
# Dal PC: porta qui i notebook eseguiti e i grafici del tentativo 18 (senza le previsioni per articolo). Uso: ./scarica.sh
cd "$(dirname "$0")"
rsync -av --include "*/" --include "*.ipynb" --include "*.png" --include "*.csv" --include "*.log" --exclude "*" \
      --exclude "lavori/" fabio.pinelli@10.181.100.6:John/Tentativo18/ ./ | tail -n 3
