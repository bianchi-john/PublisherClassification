#!/bin/bash
# Tentativo 12, tutto in automatico sul server (lanciato domenica 25 ottobre):
#   0. aspetta che finisca il terzo giro di scraping (screen t12_scraping, 0-scraping/avvia.sh);
#   1. unisce i due giri di ottobre e il terzo giro (0-scraping/unisci.sh);
#   2. costruisce il dataset di articoli (0-scraping/costruisci_dataset), con le etichette NewsGuard 2026;
#   3. pulizia (1-data-cleaning/01) e reti (1-data-cleaning/02), come nei tentativi 06-09;
#   4. il dataset solo politici (2-dataset/01_dataset_politici);
#   5. il classificatore a feature (CPU) e ModernBERT (GPU, seme 42) insieme, etichette 2026;
#   6. il confronto (3-classificazione/03).
# Ogni passo si riprova fino a 3 volte; se un passo non riesce, la catena si ferma (i passi dopo dipendono da lui).
# Un passo già fatto (segno in Results/fatti/) si salta: rilanciare lo script riprende da dove era arrivato.
#   screen -dmS t12 ./esegui_tutto.sh            stato: ./stato.sh            log: Results/esegui_tutto.log
cd "$(dirname "$0")"
PY=~/John/johnVenv/bin/python; K=johnvenv; F=Results/fatti; E=Results/esecuzioni
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUTF8=1
mkdir -p $F $E
log() { echo "$(date '+%F %T')  $*" | tee -a Results/esegui_tutto.log; }
notebook() {   # nome, cartella, notebook, parametri papermill...
    local nome=$1 dir=$2 nb=$3; shift 3
    (cd $dir && nice -n 10 $PY -m papermill -k $K $nb ../$E/$nome.ipynb "$@" >> ../$E/$nome.out 2>&1)
}
passo() {      # nome, comando...: fino a 3 tentativi
    local nome=$1; shift
    if [ -f $F/$nome ]; then log "salto $nome (già fatto)"; return 0; fi
    for t in 1 2 3; do
        log ">>> $nome (tentativo $t/3)"
        if "$@"; then date '+%F %T' > $F/$nome; log "<<< $nome fatto"; return 0; fi
        log "!!! $nome non riuscito: vedi $E/$nome.out"; sleep 120
    done
    return 1
}
ferma() { log "=== catena FERMATA: $1 non è riuscito dopo 3 tentativi ==="; exit 1; }

log "=== tentativo 12: scraping integrativo, pulizia, dataset solo politici, feature e ModernBERT ==="
G=../../../../Dataset/scraping/NuovoScraping_giro3
until grep -q "=== FINITO ===" $G/scraping.log 2>/dev/null && ! screen -ls | grep -q "\.t12_scraping\s"; do sleep 300; done
log "terzo giro di scraping finito: $(ls $G/stato | wc -l) testate tentate su $(($(wc -l < $G/elenco_testate.csv) - 1))"
passo unisci bash -c "0-scraping/unisci.sh >> $E/unisci.out 2>&1" || ferma unisci
passo costruisci_dataset notebook costruisci_dataset 0-scraping costruisci_dataset.ipynb || ferma costruisci_dataset
passo 01_pulizia notebook 01_pulizia_articoli 1-data-cleaning 01_pulizia_articoli.ipynb || ferma 01_pulizia
passo 02_reti notebook 02_reti 1-data-cleaning 02_reti.ipynb || ferma 02_reti
passo dataset_politici notebook 01_dataset_politici 2-dataset 01_dataset_politici.ipynb || ferma dataset_politici
passo feature notebook feature_2026 3-classificazione 01_feature_tfidf.ipynb -p ETICHETTA 2026 -p N_JOBS 5 &
passo modernbert notebook modernbert_2026_s42 3-classificazione 02_modernbert.ipynb -p ETICHETTA 2026 -p SEED 42 &
wait
rm -f $F/confronto; passo confronto notebook 03_confronto 3-classificazione 03_confronto.ipynb
log "=== finito ==="
