"""Ventinovesima revisione (27 novembre) di PaperRivista.odp (77 slide): l'appendice I è ora solo descrittiva (quanto si
somigliano le sorelle nel voto, negli articoli e nel testo); gli effetti sui modelli restano nella Tabella 7. John ha
tenuto solo la critica e le tabelle: le due tabelle diventano la sola tabella nuova, e i sottotitoli «Tabella N» si
riallineano a main.aux.
Prima: compilare il paper e  python figure/schermate.py . Uso:
    python aggiorna_commenti_29.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "ye"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
AUX = QUI.parents[1] / "Paper" / "Journal" / "main.aux"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

numero = dict(re.findall(r"\\newlabel\{(tab:[^}]+)\}\{\{(\d+)\}", AUX.read_text()))
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 77, len(p)


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto in giù, tagliata ai lati sul contenuto."""
    a = np.array(Image.open(SCH / f"{nome}.png").convert("L")) < 128

    def tratto(riga):
        d = np.diff(np.concatenate([[0], riga.astype(int), [0]]))
        return (np.where(d == -1)[0] - np.where(d == 1)[0]).max() if riga.any() else 0

    tratti = [tratto(r) for r in a]
    alto = next(y for y, t in enumerate(tratti) if t >= 0.9 * max(tratti))
    sotto = a[alto:]
    col, rig = np.where(sotto.any(0))[0], np.where(sotto.any(1))[0]
    uscita = SCH / f"solo_{nome}.png"
    Image.open(SCH / f"{nome}.png").crop((max(col.min() - 6, 0), alto, col.max() + 6, alto + rig.max() + 6)).save(uscita)
    return uscita


idx = [i for i, x in enumerate(p) if "Appendice I" in S.titolo_di(x)]
assert len(idx) == 3 and "la critica" in S.testo(p[idx[0]]), idx       # la critica e le due tabelle di prima
img = solo_tabella("tab_sisters")
w, h = Image.open(img).size
largo = min(24.0, 11.6 * w / h)
appendice_i = [
    p[idx[0]],
    S.pagina("Appendice I", f"Tabella {numero['tab:sisters']}", [],
             extra=S.immagine(img, (28 - largo) / 2, 3.6 + (11.6 - largo * h / w) / 2, largo=largo)),
]
p = p[:idx[0]] + appendice_i + p[idx[-1] + 1:]

for lettera, nome in [("J", "tab:anon"), ("K", "tab:reading"), ("O", "tab:orientation")]:
    i = next(i for i, x in enumerate(p) if f"Appendice {lettera}" in S.titolo_di(x) and re.search(r"Tabella \d+", S.testo(x)))
    vecchio = re.search(r"Tabella \d+", S.testo(p[i])).group(0)
    assert p[i].count(vecchio) == 1, vecchio
    p[i] = p[i].replace(vecchio, f"Tabella {numero[nome]}")
    print(lettera, vecchio, "→", f"Tabella {numero[nome]}")

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide (erano 77)")
