"""Ventisettesima revisione (27 novembre) di PaperRivista.odp (76 slide, dopo aggiorna_commenti_26.py): i sottotitoli verdi.
Si tolgono quelli che ripetono il contenuto della slide («Due passi: prima le regole, poi GLiNER», «Sezione 5.1»,
«Che cosa dice») e si accorciano quelli dell'appendice A come nelle altre appendici (il tema solo nella prima slide).
Restano quelli che dicono quale tabella o figura del paper si vede, quale metrica, quale passo della risposta.
Uso:
    python aggiorna_commenti_27.py <deck.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
p = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(p) == 76, len(p)


def sottotitolo(pag):
    """Il riquadro del sottotitolo: il secondo riquadro «title» della slide (il primo è il titolo)."""
    fr = list(re.finditer(r'<draw:frame [^>]*presentation:class="title"[^>]*>.*?</draw:frame>', pag, flags=re.S))
    return fr[1] if len(fr) > 1 else None


TOGLIERE = ["Due passi: prima le regole, poi GLiNER", "Sezione 5.1", "Che cosa dice"]
ACCORCIARE = {"Metriche in più: le tre metriche": "Le tre metriche", "Metriche in più: il risultato": "Il risultato"}

tolti, accorciati = [], []
for i, pag in enumerate(p):
    m = sottotitolo(pag)
    if m is None:
        continue
    testo = S.testo(m.group(0))
    if testo in TOGLIERE:
        p[i] = pag[:m.start()] + pag[m.end():]
        tolti.append(testo)
    elif testo in ACCORCIARE:
        p[i] = pag[:m.start()] + m.group(0).replace(testo, ACCORCIARE[testo]) + pag[m.end():]
        accorciati.append(testo)
assert sorted(tolti) == sorted(TOGLIERE) and sorted(accorciati) == sorted(ACCORCIARE), (tolti, accorciati)

S.immagini.clear()
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: tolti {tolti}, accorciati {accorciati}")
