#!/bin/bash
# Dal PC: porta qui i notebook eseguiti, i grafici e le tabelle del tentativo 16 (senza dati pesanti). Uso: ./scarica.sh
cd "$(dirname "$0")"
rsync -av --exclude cache --exclude "*.npy" --exclude "articoli.csv.gz" --exclude "previsioni_articoli.csv" --exclude "fold*.csv" \
      fabio.pinelli@10.181.100.6:John/Tentativo16/Results/ Results/ | tail -n 3
echo "I notebook eseguiti sono in Results/*_eseguito.ipynb; i grafici in Results/*/riassunto.png e Results/confronto/confronto.png"
