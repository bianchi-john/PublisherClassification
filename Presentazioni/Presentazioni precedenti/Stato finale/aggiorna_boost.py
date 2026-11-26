"""Revisione dell'8 novembre di FinalState.odp: dopo «Solo stirati», due slide NUOVE sulle altre strategie provate sugli
embedding congelati (tentativo 20, passo 10; figure/grafici_boost.py). Le slide di John restano com'erano.
Uso:  python aggiorna_boost.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zb"
S.animazione = lambda ids: ""          # niente animazioni
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"
(i,) = [i for i, p in enumerate(pagine) if "Regressori" in S.titolo_di(p) and "Solo stirati" in S.testo(p)]
nuove = [
    con_grafico("Regressori", "Altre strategie sugli embedding", [
        "Qwen e Mistral congelati: **14 metodi** al posto del ridge",
        "Ogni scelta fatta sul solo training (fold interni)",
        "Quasi tutti tra 0,72 e 0,75: il ridge era **già vicino al tetto**",
    ], V / "boost_strategie.png"),
    con_grafico("Regressori", "Il migliore: una piccola rete neurale", [
        "Rete neurale su Mistral: errore **14,2** contro 15,2, ordine quasi uguale",
        "Batte il TF-IDF sempre, anche sull'errore con tutti gli articoli",
        "«auto», scelto sui fold interni: 0,743 e 14,6, come il ridge",
    ], V / "boost_curve.png"),
]
pagine[i + 1:i + 1] = nuove
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide ->", OUT)
