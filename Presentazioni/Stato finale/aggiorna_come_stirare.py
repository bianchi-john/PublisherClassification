"""Revisione del 5 novembre (sera, 7) di FinalState.odp: una slide nuova «Come si stira», subito dopo «Stirare i voti»,
che spiega il procedimento su un giro vero (Mistral, giro 1; figure/valutazione/come_stirare.png). Le altre slide restano.
Prima: figure/grafico_come_stirare.py. Uso:  python aggiorna_come_stirare.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zz"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)

(dopo,) = [i for i, p in enumerate(pagine) if "Regressori" in S.titolo_di(p) and "Stirare i voti" in S.testo(p)]
come = con_grafico("Regressori", "Come si stira", [
    "Le 45 testate del test vanno **in fila** per voto previsto: dallo 0% al 100%",
    "Voto stirato = il voto del **training** alla **stessa posizione** nella sua fila",
    "Solo voti del training; l'ordine delle testate non cambia",
], FIG / "valutazione" / "come_stirare.png")
pagine.insert(dopo + 1, come)
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide, nuova la", dopo + 2, ",", S.contatore[0], "punti nuovi ->", OUT)
