#!/bin/bash
# From the PC: where is the review queue (run_review.sh) on the server.  Usage: ./status_review.sh
ssh -o BatchMode=yes -o ConnectTimeout=20 -i ~/.ssh/id_ed25519_lumina_vm fabio.pinelli@10.181.100.6 'bash -s' <<'REMOTE'
export TZ=Europe/Rome; cd ~/John/Tentativo20 || exit 1
O=~/John/Dataset/tentativi/20_FromScratch
echo "now: $(date '+%d/%m %H:%M')"; screen -ls | grep -o "t20_review\|t20_after" | sed 's/^/running: /'
[ -f Results/review_done ] && echo "done: review queue"
[ -f Results/after_done ] && echo "done: after the queue (cleaned code)" && grep -h "VERDICT\|^zero-shot\|^Mistral" Results/after_check.log
echo; echo "---- steps"; tail -n 8 Results/review.log 2>/dev/null
echo; echo "---- regression folds saved"
for m in Qwen3-0.6B Mistral-24B ModernBERT-frozen LLM-zero-shot; do echo "$m: $(ls $O/predictions/regression/$m/fold*.csv.gz 2>/dev/null | wc -l) of 5"; done
echo "ladder: $(ls $O/ladder/predictions/*/*/fold*.csv.gz 2>/dev/null | wc -l) of 75 folds"
echo; echo "---- last lines"; tail -n 3 $O/log.txt 2>/dev/null | cut -c1-150
echo; echo "---- errors"; grep -h "Error" Results/run_review.out 2>/dev/null | tail -n 3 || true
source ~/John/johnVenv/bin/activate && python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU: {(t-f)/2**30:.1f} GB used of {t/2**30:.0f}')" 2>/dev/null
REMOTE
