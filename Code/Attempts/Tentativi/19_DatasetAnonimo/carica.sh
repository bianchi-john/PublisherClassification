#!/bin/bash
# Dal PC: porta sul server il codice del tentativo 19 e il dataset anonimo (articoli, testate, split). Uso: ./carica.sh
cd "$(dirname "$0")"
S=fabio.pinelli@10.181.100.6
rsync -av --exclude "Results/" --exclude "__pycache__/" ./ $S:John/Tentativo19/ | tail -n 2
D=../../../../Dataset/tentativi/19_DatasetAnonimo
rsync -av $D/articoli.csv.gz $D/testate.csv $D/split.csv \
      $S:John/Dataset/tentativi/19_DatasetAnonimo/ | tail -n 2
