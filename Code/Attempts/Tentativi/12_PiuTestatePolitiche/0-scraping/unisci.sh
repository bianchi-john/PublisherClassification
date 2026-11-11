#!/bin/bash
# Tentativo 12: unisce i due giri di ottobre (NuovoScraping_ottobre) e il terzo giro del 25 ottobre (NuovoScraping_giro3)
# in ../../../../../Dataset/tentativi/12_PiuTestatePolitiche/NuovoScraping, con collegamenti fisici (nessuno spazio in più). Le testate dei due elenchi sono diverse.
set -e
cd "$(dirname "$0")/../../../../../Dataset"     # la cartella unica dei dati, nella radice del repo
T=tentativi/12_PiuTestatePolitiche
rm -rf $T/NuovoScraping.tmp && mkdir -p $T/NuovoScraping.tmp/{articoli,stato,html}
for giro in NuovoScraping_ottobre NuovoScraping_giro3; do
    for sotto in articoli stato html; do
        find "scraping/$giro/$sotto" -maxdepth 1 -type f -exec cp -al {} $T/NuovoScraping.tmp/$sotto/ \;
    done
done
T=$T python3 - <<'PY'
import csv, os
T = os.environ["T"]
righe = []
for giro in ["NuovoScraping_ottobre", "NuovoScraping_giro3"]:
    with open(f"scraping/{giro}/elenco_testate.csv") as f:
        righe += [r["dominio"] for r in csv.DictReader(f)]
assert len(righe) == len(set(righe)), "una testata è in tutti e due i giri"
with open(f"{T}/NuovoScraping.tmp/elenco_testate.csv", "w") as f:
    f.write("dominio\n" + "\n".join(righe) + "\n")
print(len(righe), "testate in tutto")
PY
cp scraping/NuovoScraping_ottobre/config.json $T/NuovoScraping.tmp/config.json
rm -rf $T/NuovoScraping && mv $T/NuovoScraping.tmp $T/NuovoScraping
echo "unite: $(ls $T/NuovoScraping/stato | wc -l) stati, $(ls $T/NuovoScraping/articoli | wc -l) file di articoli, $(ls $T/NuovoScraping/html | wc -l) file html"
