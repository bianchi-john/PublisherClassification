#!/bin/bash
# Tentativo 18, coda CPU: TF-IDF e Mistral 24B (regressori e classificatori, 5 condizioni, 3 split), poi un primo
# confronto. Da lanciare in uno screen:  screen -dmS t18_cpu ./esegui_cpu.sh       Avanzamento dal PC: ./stato.sh
export TZ=Europe/Rome
cd ~/John/Tentativo18 && source ~/John/johnVenv/bin/activate
LOG=Results/esegui.log; nota() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a $LOG; }
pm() { python -m papermill -k johnvenv -p T16_DIR ~/John/Tentativo16 "$@" >> Results/esegui_cpu.out 2>&1; }
rm -f Results/cpu_fatto
for i in 1 2; do
    nota "CPU: 2. TF-IDF e Mistral, 300 lavori (tentativo $i/2)"
    pm 02_lineari.ipynb 02_lineari.ipynb && { nota "CPU: modelli lineari finiti"; break; }
    nota "ATTENZIONE: notebook 2 interrotto, riparto dai lavori salvati"
done
nota "CPU: 4. primo confronto (con i BERT già finiti)"; pm 04_confronto.ipynb 04_confronto.ipynb || nota "ATTENZIONE: errore nel confronto"
touch Results/cpu_fatto; nota "CPU: fine"
