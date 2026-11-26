"""Quarta revisione del 26 novembre di PaperRivista.odp (56 slide), sul file di John: la Tabella 7 senza i blocchi
«31 testate con una sorella» e «195 senza», e le slide del protocollo che spiegano che le sorelle stanno solo in
training (nel dataset non ce ne sono: una testata per gruppo di proprietà).
Prima: compilare il paper e  python figure/schermate.py . Uso:
    python aggiorna_commenti_4.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zf"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 56, len(p)


def titolo(i, atteso):
    assert atteso in S.titolo_di(p[i]), (i, S.titolo_di(p[i]))
    return p[i]


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto (una riga quasi tutta scura) in giù."""
    im = Image.open(SCH / f"{nome}.png").convert("L")
    w, h = im.size
    px = im.load()
    riga = next(y for y in range(h) if sum(px[x, y] < 128 for x in range(w)) > 0.85 * w)
    uscita = SCH / f"solo_{nome}.png"
    im.crop((0, riga - 4, w, h)).save(uscita)
    return uscita


titolo(14, "Quanto conta il protocollo (1/3)")
p[14] = S.riscrivi(p[14], "le 89 testate tolte dalla pulizia", "**sorelle** solo in **training**: rimettiamo le 89 "
                   "testate tolte dalla pulizia; il test resta sulle **226** (31 hanno così una sorella in training)")
titolo(15, "Quanto conta il protocollo (2/3)")
p[15] = S.con_grafico("Quanto conta il protocollo (2/3)", "Tabella 7: ✓ = sì, × = no", [
    "Nel dataset **non ci sono sorelle**: teniamo una testata per gruppo di proprietà",
    "**Sisters**: le 89 sorelle tolte dalla pulizia tornano **solo in training**, mai in test",
    "**Article split**: niente sorelle, perché le testate di test sono **già** in training",
], solo_tabella("tab_ladder"), 20, largo_max=26.0)
titolo(16, "Quanto conta il protocollo (3/3)")
p[16] = S.riscrivi(p[16], "sulle 31 testate con una sorella", "Sorelle in training: **−7 / −12** sulle 31 testate con "
                   "una sorella (su tutte le 226 solo −0,3 / −0,8: la tabella le diluisce)")

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide, {len(S.immagini)} immagini nuove")
