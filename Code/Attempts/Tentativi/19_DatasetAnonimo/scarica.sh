#!/bin/bash
# Dal PC: porta qui i notebook eseguiti, i grafici e le tabelle del tentativo 19; con "tutto" anche le uscite per articolo
# (previsioni/, in Dataset, fuori da git). I modelli restano sul server. Uso: ./scarica.sh [tutto]
cd "$(dirname "$0")"
S=fabio.pinelli@10.181.100.6
rsync -av --include "*/" --include "*.ipynb" --include "*.png" --include "*.csv" --include "*.log" --exclude "*" \
      $S:John/Tentativo19/ ./ | tail -n 3
if [ "$1" = "tutto" ]; then
    rsync -av $S:John/Dataset/tentativi/19_DatasetAnonimo/previsioni ../../../../Dataset/tentativi/19_DatasetAnonimo/ | tail -n 2
fi
