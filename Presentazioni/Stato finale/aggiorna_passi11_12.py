"""Revisione del 9 novembre di FinalState.odp: dopo «Il migliore: una piccola rete neurale», due slide NUOVE sulle
ultime prove del tentativo 20 (passo 11, meta-articoli; passo 12, testate in più nel training; grafici da
figure/grafici_passi11_12.py). Le slide di John restano com'erano.
Uso:  python aggiorna_passi11_12.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zc"
S.animazione = lambda ids: ""          # niente animazioni
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"
(i,) = [i for i, p in enumerate(pagine) if "Regressori" in S.titolo_di(p) and "Il migliore: una piccola rete neurale" in S.testo(p)]
nuove = [
    con_grafico("Regressori", "Meta-articoli: la testata in un solo testo", [
        "L'attacco di ~20 articoli della stessa testata in un testo, 10 per testata",
        "Peggio o uguale per **tutti** i modelli, anche per ModernBERT",
    ], V / "meta_confronto.png"),
    con_grafico("Regressori", "Più testate nel training", [
        "+232 testate non politiche **solo nel training**; test identico",
        "Piccoli guadagni, quasi mai significativi: il tetto resta **≈ 0,75**",
    ], V / "piu_testate.png"),
]
pagine[i + 1:i + 1] = nuove
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide ->", OUT)
