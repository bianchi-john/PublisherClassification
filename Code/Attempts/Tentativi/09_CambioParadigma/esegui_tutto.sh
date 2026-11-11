#!/bin/bash
# Tentativo 09: la catena dopo la pulizia, in ordine, con papermill (in locale basta la CPU).
#   ./esegui_tutto.sh [python] [kernel]          es. ./esegui_tutto.sh ~/.cache/irgood-venv/bin/python irgood-venv
#   N_JOBS=5 ./esegui_tutto.sh ...               fold del riferimento TF-IDF in parallelo (default 2: PC con 14 GB)
# La pulizia (1-data-cleaning 01-02) è quella del tentativo 06 (uscite riusate: sono deterministiche); 03 va
# rieseguito per i gruppi con il Parent Domain. Ogni passo già fatto (notebook eseguito presente) si salta.
set -e
PY="${1:-python}"; K="${2:-python3}"; QUI="$(cd "$(dirname "$0")" && pwd)"
passo() {   # cartella, notebook, uscita, parametri papermill...
    local c="$1" nb="$2" out="$3"; shift 3
    if [ -f "$QUI/Results/$out" ]; then echo "$(date '+%F %T')  salto $nb (già fatto)"; return; fi
    mkdir -p "$(dirname "$QUI/Results/$out")"
    echo "$(date '+%F %T')  >>> $nb"
    (cd "$QUI/$c" && "$PY" -m papermill -k "$K" "$nb" "$QUI/Results/$out.tmp.ipynb" "$@" > "$QUI/Results/$out.out" 2>&1)
    mv "$QUI/Results/$out.tmp.ipynb" "$QUI/Results/$out"
    echo "$(date '+%F %T')  <<< $nb fatto"
}
passo 1-data-cleaning 03_dataset_finale.ipynb 1-data-cleaning/esecuzioni/03_dataset_finale.ipynb
passo 2-segnali 01_estrai_segnali.ipynb 2-segnali/esecuzioni/01_estrai_segnali.ipynb -p N_JOBS 6
passo 2-segnali 02_segnali_testata.ipynb 2-segnali/esecuzioni/02_segnali_testata.ipynb
passo 3-classificazione 01_riferimento_tfidf.ipynb 3-classificazione/esecuzioni/01_riferimento_tfidf.ipynb -p NOME feature_tuned -p C_GRIGLIA 0.1,1,10,100 -p N_JOBS "${N_JOBS:-2}" -y 'SCHEMI_DA_FARE: [fold_gruppo]'
passo 3-classificazione 02_modelli.ipynb 3-classificazione/esecuzioni/02_modelli.ipynb
passo 3-classificazione 03_confronto.ipynb 3-classificazione/esecuzioni/03_confronto.ipynb
echo "$(date '+%F %T')  === catena finita ==="
