"""Terza revisione del 26 novembre di PaperRivista.odp (61 slide), sul file di John:
- Figura 8: il nome della testata stessa (regole) separato dalle testate trovate da GLiNER;
- Tabella 5 senza la riga ripesata;
- tolte le cinque slide «Testate non raccolte»: l'analisi è tolta anche dal paper; nell'appendice D resta il
  ragionamento (senza articoli nessun metodo basato sul testo può valutare una testata).
Prima: compilare il paper, python figure/schermate.py e solo_anonymization.png senza didascalia. Uso:
    python aggiorna_commenti_3.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "Presentazioni precedenti" / "Stato finale"))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "ze"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 61, len(p)


def titolo(i, atteso):
    assert atteso in S.titolo_di(p[i]), (i, S.titolo_di(p[i]))
    return p[i]


# Figura 8, la più grande possibile sotto il sottotitolo
titolo(6, "Anonimizzazione")
p[6] = S.pagina("Anonimizzazione", "Figura messa in appendice/supplementary", [],
                extra=S.figura(SCH / "solo_anonymization.png", 3.6, largo_max=26.0, basso=15.5))
# Tabella 5 rifatta (una riga in meno)
p[13] = S.cambia_immagine(titolo(13, "Contro TF-IDF"), SCH / "tab_beats.png")
# via le slide delle testate non raccolte
for i in range(14, 19):
    titolo(i, "Testate non raccolte")
# appendice D: niente stima, un ragionamento
titolo(37, "Appendice D")
p[37] = S.togli_punto(p[37], "testate non valutate")
titolo(39, "Appendice D")
p[39] = S.riscrivi(p[39], "Non è rappresentativo", "Senza articoli leggibili **nessun metodo** basato sul testo può "
                   "valutare una testata: sono fuori dal compito, e lo diciamo")

nuove = p[:14] + p[19:]
# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 61), {len(S.immagini)} immagini nuove")
