#!/bin/bash
# From the PC, before starting: models in the cache, libraries, data, disk and GPU. Downloads and starts nothing.
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_lumina_vm fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTE'
source ~/John/johnVenv/bin/activate || { echo "MISSING ~/John/johnVenv"; exit 1; }
for m in answerdotai/ModernBERT-base Qwen/Qwen3-Embedding-8B mistralai/Mistral-Small-24B-Base-2501; do
    d=~/.cache/huggingface/hub/models--${m/\//--}; [ -d "$d" ] && echo "ok       $m" || echo "MISSING  $m"; done
python -c "import torch, transformers, bitsandbytes, sklearn, papermill, trafilatura, langdetect, datasketch, tldextract; print('libraries ok')" 2>&1 | tail -1
for p in ~/John/Dataset/LEGGIMI.md ~/John/Dataset/etichette/newsguard_2026.csv ~/John/Dataset/scraping/NuovoScraping_ottobre/html ~/John/Dataset/scraping/NuovoScraping_giro3/html; do
    [ -e "$p" ] && echo "ok       $p" || echo "MISSING  $p"; done
df -h ~ | tail -1 | awk '{print "disk: " $4 " free"}'
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {f/2**30:.1f} GB free of {t/2**30:.0f}')" 2>/dev/null
screen -ls | grep -E "t19|t20" || echo "no t19/t20 screens"
REMOTE
