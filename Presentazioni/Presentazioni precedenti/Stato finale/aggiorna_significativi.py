"""Revisione del 5 novembre (sera, 5) di FinalState.odp: nelle due slide delle curve dei regressori sono evidenziati i
punti in cui Qwen, Mistral o ModernBERT battono il TF-IDF in modo significativo (figure/grafici_curve.py).
Si rigenerano solo quelle due slide (erano generate e non toccate a mano). Uso:
python aggiorna_significativi.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zx"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


pagine[trova("Regressori", "Le prestazioni, articolo dopo articolo")] = con_grafico("Regressori", "Le prestazioni, articolo dopo articolo", [
    "Ogni punto: **k articoli a caso** per testata (20 volte); «tutti» = il risultato finale",
    "**Cerchiati**: battono il TF-IDF in modo significativo. **Mistral sempre**, Qwen sull'errore; ModernBERT mai",
], V / "curve_normale.png")
pagine[trova("Regressori", "Normale e stirato")] = con_grafico("Regressori", "Normale e stirato", [
    "Stirato (tratteggiato): meglio **solo con molti articoli**; lo Spearman non cambia",
    "Stirato, con tutti gli articoli **nessuno batte più il TF-IDF**: stirare aiuta anche lui",
], V / "curve_confronto.png")
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide,", S.contatore[0], "punti nuovi ->", OUT)
