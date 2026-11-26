"""Revisione del 5 novembre (sera) di FinalState.odp: John ha tolto le due slide dello split; una sola slide nuova le
sostituisce, dopo «Anonimizzazione – Il risultato», con il grafico figure/split_fold.png (figure/grafico_split.py):
fold vecchi (per livello) e nuovi (per voto), testate per livello e singoli voti. Le altre slide non si toccano.
Uso:  python aggiorna_split.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zt"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 51, len(pagine)
assert not any("split" in S.titolo_di(p).lower() for p in pagine), "c'è già una slide dello split"
dopo = next(i for i, p in enumerate(pagine) if "Anonimizzazione" in S.titolo_di(p) and "Il risultato" in S.testo(p))
pagine.insert(dopo + 1, con_grafico("Lo split", "Vecchio e nuovo, fold per fold", [], FIG / "split_fold.png", y=3.3, largo_max=27))
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide (erano 51): split alla slide", dopo + 2, "->", OUT)
