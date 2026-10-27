#!/bin/bash
# Aspetta che la diagnostica di ModernBERT salvi la sua curva, poi ferma la catena partita con i parametri sbagliati
# e rilancia quella corretta (esegui_tutto.sh resto). Da lanciare in uno screen.
export TZ=Europe/Rome
cd ~/John/Tentativo16
until [ -f Results/modernbert/diagnostica/curva_fold2.csv ]; do sleep 30; done
echo "$(date '+%Y-%m-%d %H:%M:%S')  diagnostica salvata: fermo la catena e rilancio con i parametri giusti" >> Results/esegui_tutto.log
screen -S t16 -X quit; sleep 10
pkill -f "papermill -k johnvenv -p PROVA"; pkill -f "John/johnVenv/bin/python -Xfrozen_modules=off -m ipykernel_launcher"; sleep 10
echo "--- rilancio $(date)" >> Results/esegui_tutto.log
screen -dmS t16 ./esegui_tutto.sh resto
