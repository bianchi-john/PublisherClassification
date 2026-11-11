#!/usr/bin/env bash
# Step 1 (not re-run in this attempt): keeps the scraper alive. If the main process stops before the end it is
# restarted; finished outlets are skipped, so every restart resumes where it was.
# Usage (inside screen):  bash run_scraper.sh <newsguard.csv> <output folder> [processes]
cd "$(dirname "$0")"
NEWSGUARD="$1"; OUT="$2"; PROCESSES="${3:-64}"
mkdir -p "$OUT"
for attempt in $(seq 1 50); do
    echo "$(date '+%F %T')  start number $attempt" >> "$OUT/restarts.log"
    python scraper.py --newsguard "$NEWSGUARD" --out "$OUT" --processes "$PROCESSES" >> "$OUT/console.log" 2>&1
    if grep -q "=== FINISHED ===" "$OUT/scraping.log" 2>/dev/null; then
        echo "$(date '+%F %T')  finished" >> "$OUT/restarts.log"; exit 0
    fi
    echo "$(date '+%F %T')  stopped before the end: restarting in 60 s" >> "$OUT/restarts.log"
    sleep 60
done
