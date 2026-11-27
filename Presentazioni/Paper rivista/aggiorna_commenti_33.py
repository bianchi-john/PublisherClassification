"""Trentatreesima revisione (27 novembre) di PaperRivista.odp: via l'appendice K (boilerplate), tolta anche dal paper per
scelta di John. Le appendici dopo cambiano lettera come nel paper (L, N → K, M; O → N; il tool M → L), e i rimandi e il
numero della tabella dell'orientamento si riallineano a main.aux.
Prima: compilare il paper. Uso:
    python aggiorna_commenti_33.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

AUX = QUI.parents[1] / "Paper" / "Journal" / "main.aux"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
etichetta = dict(re.findall(r"\\newlabel\{([a-z]+:[^}]+)\}\{\{([^}]+)\}", AUX.read_text()))
assert etichetta["app:errors"] == "K" and etichetta["app:tried"] == "M" and etichetta["app:tool"] == "L"
assert etichetta["app:orientation"] == "N"

z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
p = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
prima = len(p)
p = [x for x in p if "Appendice K" not in S.titolo_di(x)]          # il boilerplate
print(f"tolte {prima - len(p)} slide dell'appendice K")


def cambia(pag, vecchio, nuovo):
    n = pag.count(vecchio)
    return pag.replace(vecchio, nuovo), n


conti = {}
nuove = []
for pag in p:
    for vecchio, nuovo in [("Appendice L, N", "Appendice K, M"), ("Appendice O", "Appendice N"),
                           ("Nel paper (Appendice M)", "Nel paper (Appendice L)"), ("(appendice N)", "(appendice M)"),
                           ("Tabella 17", f"Tabella {etichetta['tab:orientation']}")]:
        pag, n = cambia(pag, vecchio, nuovo)
        conti[vecchio] = conti.get(vecchio, 0) + n
    nuove.append(pag)
print(conti)
assert all(conti.values()), conti

S.immagini.clear()
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano {prima})")
