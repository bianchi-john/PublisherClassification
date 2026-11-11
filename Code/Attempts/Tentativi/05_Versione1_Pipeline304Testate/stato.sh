#!/bin/bash
# Cosa gira sul server, a che punto è la coda degli esperimenti e quanto manca.
#   ~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate/stato.sh          (sul server)
#   ./controlla_server.sh                    (dal PC: lo lancia via ssh)
W=~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate
R=$W/Results/2-classificazione
echo "=== ora italiana $(TZ=Europe/Rome date '+%d/%m %H:%M')  (i log usano l'ora del server, UTC: 2 ore indietro)"

echo; echo "=== screen attivi (nome, partenza in ora italiana, da quanto gira)"
screen -ls | sed -nE 's/^\s+[0-9]+\.(\S+)\s+\(([^)]*)\).*/\1|\2/p' | while IFS='|' read nome inizio; do
    s=$(( $(date +%s) - $(date -d "$inizio" +%s) ))
    printf "  %-18s %s, da %dh %02dm\n" "$nome" "$(TZ=Europe/Rome date -d @$(date -d "$inizio" +%s) '+%d/%m %H:%M')" $((s/3600)) $((s%3600/60))
done

if [ -f $W/Results/esegui_tutto/stato.json ]; then
    echo; echo "=== catena (scraping -> dataset -> pulizia -> classificazione)"
    ~/John/johnVenv/bin/python -c "
import json; s = json.load(open('$W/Results/esegui_tutto/stato.json'))
print('  dataset di riserva in uso!' if s.get('riserva') else '  dataset: quello nuovo')
for n, f in s['fasi'].items(): print(f\"  {n:22s} {f['stato']:20s} {f.get('minuti', '')!s:>7} min\")"
    tail -3 $W/Results/esegui_tutto/esegui_tutto.log | sed 's/^/  /'
    if [ -f $W/../../../../Dataset/scraping/NuovoScraping/stato_generale.json ]; then
        echo "  scraping: $(~/John/johnVenv/bin/python -c "import json; s=json.load(open('$W/../../../../Dataset/scraping/NuovoScraping/stato_generale.json')); print(s['testate_finite'], 'su', s['testate_totali'], 'testate tentate; ore mancanti stimate', s.get('ore_mancanti_stimate'))")"
    fi
fi

if [ -f $R/coda_stato.json ]; then
    echo; echo "=== coda del weekend"
    ~/John/johnVenv/bin/python - <<'EOF'
import json, re, glob, os, time
from pathlib import Path
R = Path(os.path.expanduser("~/John/WebScience2027/Tentativi/05_Versione1_Pipeline304Testate/Results/2-classificazione"))
s = json.load(open(R / "coda_stato.json"))
print(f"  aggiornata: {s.get('aggiornato')} (ora del server) | in corso: {s.get('in_corso') or '-'}")
for nome, e in s["esperimenti"].items():
    d = R / "transformer" / nome
    fold = len(glob.glob(str(d / "fold*_test.csv"))) if d.exists() else None
    durate = []
    log = R / f"avanzamento_{nome}.log"
    if log.exists():
        durate = [int(x) for x in re.findall(r"finito in (\d+) min", log.read_text())]
    extra = ""
    if fold is not None:
        extra = f"fold {fold}/5"
        if durate:
            extra += f", {sum(durate) / len(durate):.0f} min per fold"
            if e["stato"] == "in corso" and fold < 5:
                extra += f", finisce tra circa {(5 - fold) * sum(durate) / len(durate) / 60:.1f} h"
    print(f"  {nome:30s} {e['stato']:13s} tentativi {e.get('tentativi', 0)}  {e.get('minuti', '')!s:>6} min  {extra}")
EOF
    echo; echo "=== ultime righe del log della coda"
    tail -5 $R/coda.log | sed 's/^/  /'
fi

echo; echo "=== ultimi passi (log più recente: $(ls -t $R/avanzamento_*.log 2>/dev/null | head -1 | xargs -r basename))"
tail -6 "$(ls -t $R/avanzamento_*.log 2>/dev/null | head -1)" 2>/dev/null | sed 's/^/  /'

echo; echo "=== GPU"
~/John/johnVenv/bin/python -c "import torch; f,t = torch.cuda.mem_get_info(); print(f'  memoria libera {f/2**30:.1f} / {t/2**30:.0f} GB')" 2>/dev/null
