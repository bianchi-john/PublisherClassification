#!/bin/bash
# Dal PC: stato del server e, se è finito, scarica i notebook eseguiti e i risultati.
cd "$(dirname "$0")"
S="ssh -o ConnectTimeout=15 fabio.pinelli@10.181.100.6"
$S 'tail -n 8 ~/John/Tentativo14/Results/esegui.log; tail -n 3 ~/John/Tentativo14/Results/cache/avanzamento.log 2>/dev/null'
if $S 'grep -q "<<< finito" ~/John/Tentativo14/Results/esegui.log'; then
    rsync -a fabio.pinelli@10.181.100.6:John/Tentativo14/Results/ Results/
    cp Results/01_preparazione_dataset_eseguito.ipynb 01_preparazione_dataset.ipynb
    cp Results/02_classificazione_eseguito.ipynb 02_classificazione.ipynb
    echo "scaricato: i notebook in questa cartella ora hanno gli output"
fi
