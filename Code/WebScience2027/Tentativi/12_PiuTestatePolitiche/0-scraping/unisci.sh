#!/bin/bash
# Tentativo 12: unisce i due giri di ottobre (NuovoScraping_ottobre) e il terzo giro del 25 ottobre (NuovoScraping_giro3)
# in Dataset/NuovoScraping, con collegamenti fisici (nessuno spazio in più). Le testate dei due elenchi sono diverse.
set -e
cd "$(dirname "$0")/../Dataset"
rm -rf NuovoScraping.tmp && mkdir -p NuovoScraping.tmp/{articoli,stato,html}
for giro in NuovoScraping_ottobre NuovoScraping_giro3; do
    for sotto in articoli stato html; do
        find "$giro/$sotto" -maxdepth 1 -type f -exec cp -al {} NuovoScraping.tmp/$sotto/ \;
    done
done
python3 - <<'PY'
import csv
righe = []
for giro in ["NuovoScraping_ottobre", "NuovoScraping_giro3"]:
    with open(f"{giro}/elenco_testate.csv") as f:
        righe += [r["dominio"] for r in csv.DictReader(f)]
assert len(righe) == len(set(righe)), "una testata è in tutti e due i giri"
with open("NuovoScraping.tmp/elenco_testate.csv", "w") as f:
    f.write("dominio\n" + "\n".join(righe) + "\n")
print(len(righe), "testate in tutto")
PY
cp NuovoScraping_ottobre/config.json NuovoScraping.tmp/config.json
rm -rf NuovoScraping && mv NuovoScraping.tmp NuovoScraping
echo "unite: $(ls NuovoScraping/stato | wc -l) stati, $(ls NuovoScraping/articoli | wc -l) file di articoli, $(ls NuovoScraping/html | wc -l) file html"
