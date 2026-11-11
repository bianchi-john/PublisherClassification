#!/bin/bash
# Notte del 28/10: i vettori di Mistral-Small 24B (notebook 8, GPU, circa 4 ore; se si interrompe riparte dal lavoro
# salvato), poi il notebook 7 con Mistral e con il voto degli articoli (solo CPU, circa 20 minuti).
# Da lanciare in uno screen:   screen -dmS t16 ./esegui_notte.sh        Avanzamento dal PC: ./stato.sh
export TZ=Europe/Rome
cd ~/John/Tentativo16
source ~/John/johnVenv/bin/activate
LOG=Results/esegui_tutto.log
nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv "$@" >> Results/esegui_tutto.out 2>&1; }

nota "================ notte: Mistral-Small 24B e voto degli articoli"
for i in 1 2 3; do
    nota ">>> 8. Mistral-Small 24B, strati 27 e 40 (circa 4 ore; tentativo $i/3)"
    pm 08_llm_grande.ipynb 08_llm_grande.ipynb && { nota "Mistral finito"; break; }
    nota "ATTENZIONE: notebook 8 interrotto, riprovo tra 5 minuti dal lavoro salvato"; sleep 300
done
nota ">>> 7. analisi degli embedding con Mistral e con il voto degli articoli (circa 20 minuti)"
pm 07_analisi_embedding.ipynb 07_analisi_embedding.ipynb || nota "ATTENZIONE: errore nel notebook 7"
nota "<<< FINITO"
