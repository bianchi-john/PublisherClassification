"""Revisione del 5 novembre (sera, 16) di FinalState.odp: dopo l'ultima slide «Classificatore o regressore?» (le
curve), una slide in più con le curve della tolleranza ½ e ±1 sui livelli (figure/valutazione/curve_cls_reg_tol.png,
da figure/grafici_finali.py). Niente animazioni. Uso:  python aggiorna_tolleranze.py <file di John.odp> <uscita.odp>"""
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
senza_clic = lambda p: re.sub(r'<anim:par[^>]*presentation:node-type="timing-root".*?</anim:seq></anim:par>', "", p, flags=re.S)

(i,) = [i for i, p in enumerate(pagine) if "Classificatore o regressore?" in S.titolo_di(p) and "Pieno: classificatore" in S.testo(p)]
nuova = senza_clic(con_grafico("Classificatore o regressore?", "Livelli con tolleranza", [
    "**Tolleranza ½**: livello giusto = 1, sbagliato di un livello = ½, altrimenti 0",
    "**Tolleranza ±1**: giusto se sbaglia al massimo di un livello",
    "Il regressore (tratteggiato) è **pari o meglio** quasi sempre: con tutti gli articoli ±1 Mistral 0,86 contro 0,83",
], FIG / "valutazione" / "curve_cls_reg_tol.png"))
pagine.insert(i + 1, nuova)
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide; nuova la", i + 2, "->", OUT)
