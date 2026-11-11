#!/bin/bash
# Dal PC: a che punto è il tentativo 18 sul server. Uso: ./stato.sh
ssh -o ConnectTimeout=20 fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTO'
export TZ=Europe/Rome; cd ~/John/Tentativo18 || exit 1
echo "ora: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t18_[a-z]*" | sed 's/^/in corso: /'
echo; echo "---- passi"; tail -n 12 Results/esegui.log 2>/dev/null
echo; echo "---- modelli lineari: $(ls Results/lineari/lavori/*_info.csv 2>/dev/null | wc -l) lavori su 300"
echo; echo "---- BERT (ultime righe)"; tail -n 4 Results/bert_avanzamento.log 2>/dev/null
echo; echo "---- errori"; grep -h "Error" Results/esegui_*.out 2>/dev/null | tail -n 3 || true
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB usati su {t/2**30:.0f}')" 2>/dev/null
REMOTO
