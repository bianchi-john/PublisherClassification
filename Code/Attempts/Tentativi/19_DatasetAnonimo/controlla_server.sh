#!/bin/bash
# Dal PC, prima di lanciare: sul server ci sono i modelli nella cache, le librerie, lo spazio e la GPU? Non scarica e non
# lancia niente. Uso: ./controlla_server.sh
ssh -o ConnectTimeout=20 fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTO'
source ~/John/johnVenv/bin/activate || { echo "MANCA il venv ~/John/johnVenv"; exit 1; }
echo "---- modelli nella cache di HuggingFace (se mancano, il notebook li scaricherebbe)"
for m in answerdotai/ModernBERT-base Qwen/Qwen3-Embedding-8B mistralai/Mistral-Small-24B-Base-2501; do
    d=~/.cache/huggingface/hub/models--${m/\//--}
    if [ -d "$d" ]; then echo "ok      $m ($(du -sh "$d" | cut -f1))"; else echo "MANCA   $m"; fi
done
echo; echo "---- librerie"
python -c "import torch, transformers, bitsandbytes, sklearn, papermill, joblib, scipy; print('torch', torch.__version__, '| transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__, '| sklearn', sklearn.__version__)"
jupyter kernelspec list 2>/dev/null | grep -q johnvenv && echo "kernel johnvenv: ok" || echo "MANCA il kernel johnvenv"
echo; echo "---- cartelle e spazio"
ls -d ~/John/Dataset/LEGGIMI.md ~/John/Tentativo19 ~/John/Dataset/tentativi/19_DatasetAnonimo 2>&1 | sed 's/^/  /'
df -h ~ | tail -1 | awk '{print "  disco: " $4 " liberi su " $2}'
echo; echo "---- GPU e screen"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'
screen -ls | grep -v "^There\|^No\|Sockets" | sed 's/^/  /'
REMOTO
