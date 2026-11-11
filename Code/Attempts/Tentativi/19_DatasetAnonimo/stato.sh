#!/bin/bash
# Dal PC: a che punto è il tentativo 19 sul server. Uso: ./stato.sh
ssh -o ConnectTimeout=20 fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTO'
export TZ=Europe/Rome; cd ~/John/Tentativo19 || exit 1
echo "ora: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t19_[a-z]*" | sed 's/^/in corso: /'
echo; echo "---- passi"; tail -n 12 Results/esegui.log 2>/dev/null
echo; echo "---- fold salvati"
for c in classificazione regressione; do for m in "TF-IDF" "Qwen3 8B" "Mistral 24B" "ModernBERT"; do
    echo "$c / $m: $(ls ~/John/Dataset/tentativi/19_DatasetAnonimo/previsioni/$c/"$m"/fold*.csv.gz 2>/dev/null | wc -l) su 5"; done; done
echo; echo "---- ultime righe"; tail -n 4 Results/avanzamento.log 2>/dev/null
echo; echo "---- errori"; grep -h "Error" Results/esegui_*.out 2>/dev/null | tail -n 3 || true
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB usati su {t/2**30:.0f}')" 2>/dev/null
REMOTO
