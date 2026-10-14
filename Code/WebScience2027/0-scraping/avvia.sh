#!/usr/bin/env bash
# Tiene in vita lo scraper: se il processo principale si ferma senza aver finito, lo rilancia.
# Le testate già finite vengono saltate, quindi ogni rilancio riprende da dove era arrivato.
# Uso (dentro screen):  bash avvia.sh [cartella_output] [processi]
cd "$(dirname "$0")"
OUT="${1:-../../Dataset/NuovoScraping}"
PROCESSI="${2:-64}"
PY="${PYTHON:-$HOME/johnVenv/bin/python}"
mkdir -p "$OUT"
for tentativo in $(seq 1 50); do
    echo "$(date '+%F %T')  avvio numero $tentativo" >> "$OUT/riavvii.log"
    "$PY" scraper.py --out "$OUT" --processi "$PROCESSI" >> "$OUT/console.log" 2>&1
    if grep -q "=== FINITO ===" "$OUT/scraping.log" 2>/dev/null && \
       [ "$(ls "$OUT/stato" | wc -l)" -ge "$(($(wc -l < "$OUT/elenco_testate.csv") - 1))" ]; then
        echo "$(date '+%F %T')  finito" >> "$OUT/riavvii.log"
        exit 0
    fi
    echo "$(date '+%F %T')  lo scraper si è fermato prima della fine: riparto tra 60 s" >> "$OUT/riavvii.log"
    sleep 60
done
