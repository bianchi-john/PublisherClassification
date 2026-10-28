#!/bin/bash
# Dal PC: a che punto è il tentativo 16 sul server. Uso: ./stato.sh   (chiede la password una volta)
ssh -o ConnectTimeout=20 fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTO'
export TZ=Europe/Rome
cd ~/John/Tentativo16 || { echo "cartella non trovata"; exit 1; }
echo "ora: $(date '+%d/%m %H:%M')"
if screen -ls | grep -q "\.t16"; then echo "STATO: la catena è IN CORSO"; else echo "STATO: la catena NON è in corso (finita o fermata: vedi sotto)"; fi
echo; echo "---- passi"; tail -n 12 Results/esegui_tutto.log 2>/dev/null || echo "non ancora partita"
echo; echo "---- notebook in esecuzione"; tail -c 300 Results/esegui_tutto.out 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -n 1
f=$(ls -t Results/04_embedding_*_eseguito.ipynb 2>/dev/null | head -1)
[ -n "$f" ] && { echo; echo "---- embedding ($f)"; grep -o '[0-9,]*/[0-9,]* articoli ([0-9]* s)' "$f" | tail -n 1; }
for f in Results/prova/modernbert/avanzamento.log Results/modernbert/avanzamento.log; do
    [ -f $f ] && { echo; echo "---- $f (ultime righe)"; tail -n 4 $f; }
done
echo; echo "---- errori"; grep -A3 "Cell In" Results/esegui_tutto.out 2>/dev/null | tail -n 8 || true
grep -q "Cell In" Results/esegui_tutto.out 2>/dev/null || echo "nessuno"
echo; echo "---- risorse"
echo "disco libero: $(df -h ~ | tail -1 | awk '{print $4}')"
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB usati su {t/2**30:.0f}')" 2>/dev/null
REMOTO
