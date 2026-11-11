#!/bin/bash
# Tentativo 16 sul server (~/John/Tentativo16), da lanciare in uno screen:   screen -dmS t16 ./esegui_tutto.sh
# Con l'argomento "resto" salta i notebook 1–3 e la prova (già fatti). Va avanti anche se ci si disconnette.
# Avanzamento: Results/esegui_tutto.log (dal PC: ./stato.sh). Ogni notebook si salva eseguito al posto suo, con gli output;
# gli embedding 0,6B in Results/04_embedding_0.6B_eseguito.ipynb, gli 8B in 04_embedding.ipynb.
export TZ=Europe/Rome                       # orari italiani nei log
cd ~/John/Tentativo16
source ~/John/johnVenv/bin/activate
mkdir -p Results/prova
LOG=Results/esegui_tutto.log
nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_tutto.out 2>&1; }

nota "================ inizio ${1:-}"
if [ "$1" != "resto" ]; then
    nota ">>> 1. data preparation (circa 6 min)";   pm 01_data_preparation.ipynb 01_data_preparation.ipynb || { nota "FERMATO: errore nel notebook 1"; exit 1; }
    nota ">>> 2. data analysis (circa 10 min)";     pm 02_data_analysis.ipynb 02_data_analysis.ipynb || nota "ATTENZIONE: errore nel notebook 2, continuo"
    nota ">>> 3. feature (circa 5 min)";            pm 03_feature.ipynb 03_feature.ipynb || { nota "FERMATO: errore nel notebook 3"; exit 1; }
    # prova veloce delle parti con la GPU (2 articoli per testata, 1 epoca): se qualcosa non va, ci si ferma subito
    nota ">>> prova: embedding";  pm -p PROVA 2 04_embedding.ipynb Results/prova/04_prova.ipynb || { nota "FERMATO: errore nella prova degli embedding"; exit 1; }
    nota ">>> prova: ModernBERT"; pm -p PROVA 2 -p EPOCHE 1 -p DIAGNOSTICA False 05_modernbert.ipynb Results/prova/05_prova.ipynb || { nota "FERMATO: errore nella prova di ModernBERT"; exit 1; }
    nota "prova superata"
fi

nota ">>> 4. embedding 0.6B (circa 15 min; pochi minuti se già calcolati)"
pm -p MODELLO Qwen/Qwen3-Embedding-0.6B 04_embedding.ipynb Results/04_embedding_0.6B_eseguito.ipynb || nota "ATTENZIONE: errore negli embedding 0.6B, continuo"
nota ">>> 6. confronto (TF-IDF ed embedding 0.6B)"; pm 06_confronto.ipynb 06_confronto.ipynb || nota "ATTENZIONE: errore nel confronto, continuo"
for i in 1 2 3; do                          # ModernBERT: se si interrompe riparte dai fold già salvati
    nota ">>> 5. ModernBERT (diagnostica in cache + 2 configurazioni, 5 epoche: circa 12 ore; tentativo $i/3)"
    pm 05_modernbert.ipynb 05_modernbert.ipynb && { nota "ModernBERT finito"; break; }
    nota "ATTENZIONE: ModernBERT interrotto, riprovo tra 5 minuti"; sleep 300
done
nota ">>> 6. confronto (con ModernBERT)"; pm 06_confronto.ipynb 06_confronto.ipynb || nota "ATTENZIONE: errore nel confronto, continuo"
nota ">>> 4. embedding 8B (circa 2-3 ore)"
pm -p MODELLO Qwen/Qwen3-Embedding-8B -p BATCH 4 04_embedding.ipynb 04_embedding.ipynb || nota "ATTENZIONE: errore negli embedding 8B"
nota ">>> 6. confronto finale"; pm 06_confronto.ipynb 06_confronto.ipynb
nota "<<< FINITO"
