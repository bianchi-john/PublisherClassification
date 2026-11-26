"""Revisione del 5 novembre (sera, 8) di FinalState.odp: nella slide «TF-IDF – Sopra, una regressione ridge» un
sottopunto sotto «Ridge: voto = somma di ...» che dice che cos'è un ridge; le due schermate di codice scendono di
1,6 cm. Poi rilanciare aggiusta_animazioni.py (un clic per punto). Uso:  python aggiorna_ridge.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
(i,) = [i for i, p in enumerate(pagine) if "su tutte le parole, più una costante" in S.testo(p)]
p = pagine[i]
assert "cioè una regressione lineare" not in S.testo(p), "già fatto"

SOTTO = ("cioè una **regressione lineare con penalità**: preferisce pesi **piccoli**, così nessuna parola da sola "
         "sposta troppo il voto")
m = [m for m in re.finditer(r"<text:p(?: [^>]*)?>.*?</text:p>", p, flags=re.S) if "su tutte le parole" in S.testo(m.group(0))][0]
p = p[:m.end()] + f'<text:list><text:list-item><text:p xml:id="zridge1" text:id="zridge1">{S.runs(SOTTO, 20)}</text:p></text:list-item></text:list>' + p[m.end():]

# le due schermate di codice (immagini sotto il testo) scendono di una riga
SPOSTA = 1.6
p = re.sub(r'(<draw:frame draw:style-name="[^"]+"[^>]*svg:y=")([\d.]+)(cm"[^>]*><draw:image)',
           lambda m: f"{m.group(1)}{float(m.group(2)) + SPOSTA:.2f}{m.group(3)}", p)
# il riquadro del testo cresce di una riga, così LibreOffice non rimpicciolisce il testo
p = re.sub(r'(<draw:frame [^>]*svg:height=")([\d.]+)(cm"[^>]*presentation:class="outline")',
           lambda m: f"{m.group(1)}{float(m.group(2)) + 1.7:.3f}{m.group(3)}", p, count=1)
pagine[i] = p
S.scrivi(IN, OUT, c, manifest, pagine)
print("slide", i + 1, "->", OUT)
