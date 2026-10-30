#!/bin/bash
# From the PC: where is attempt 20 on the server.  Usage: ./status.sh
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_lumina_vm fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTE'
export TZ=Europe/Rome; cd ~/John/Tentativo20 || exit 1
echo "now: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t20_[a-z]*" | sed 's/^/running: /'
echo; echo "---- steps"; tail -n 10 Results/run.log 2>/dev/null
echo; echo "---- folds saved"
for t in classification regression; do for m in TF-IDF Qwen3-8B Mistral-24B ModernBERT; do
    echo "$t / $m: $(ls ~/John/Dataset/tentativi/20_FromScratch/predictions/$t/$m/fold*.csv.gz 2>/dev/null | wc -l) of 5"; done; done
echo; echo "---- last lines"; tail -n 3 ~/John/Dataset/tentativi/20_FromScratch/log.txt 2>/dev/null
echo; echo "---- errors"; grep -h "Error" Results/run_*.out 2>/dev/null | tail -n 3 || true
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB used of {t/2**30:.0f}')" 2>/dev/null
REMOTE
