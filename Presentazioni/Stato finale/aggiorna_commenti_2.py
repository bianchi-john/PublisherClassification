"""Revisione del 5 novembre (sera, 15) di FinalState.odp, commenti di John:
- «NewsGuard e MBFC – Un voto contro una classe»: perché le posizioni MBFC hanno la virgola (pari merito = posizione
  media: le due «High» occupano i posti 1 e 2 → 1,5);
- prima di «Classificatore o regressore?»: una slide su come si confrontano (classificatore → voto atteso; regressore →
  livelli con le quote del training), figure/valutazione/cls_reg_come.png.
Tolti i due commenti. Niente animazioni (presentazione interna).
Prima: figure/grafico_cls_reg_come.py. Uso:  python aggiorna_commenti_2.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zk"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"
senza_clic = lambda p: re.sub(r'<anim:par[^>]*presentation:node-type="timing-root".*?</anim:seq></anim:par>', "", p, flags=re.S)


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


pagine[trova("NewsGuard e MBFC", "Un voto contro una classe")] = senza_clic(con_grafico(
    "NewsGuard e MBFC", "Un voto contro una classe: come si confrontano", [
        "Nessuna conversione: contano solo le **posizioni** (Low < Mixed < Mostly Factual < High)",
        "Stessa classe = **pari merito**: le due «High» sono ai posti 1 e 2 → entrambe **(1 + 2) / 2 = 1,5**",
        "Su tutte le 170 testate in comune: **0,71**",
    ], V / "mbfc_posizioni.png"))

i = trova("Classificatore o regressore?", "")
pagine[i] = S.senza_commenti(pagine[i])
come = senza_clic(con_grafico("Classificatore o regressore?", "Come si confrontano", [
    "Classificatore → voto: **voto atteso** (probabilità × voto medio del livello)",
    "Regressore → livello: in fila, tagli alle **quote** dei livelli nel training",
    "Livello del classificatore: il **più votato** dai suoi articoli",
], V / "cls_reg_come.png"))
pagine.insert(i, come)
assert not any("<officeooo:annotation" in p for p in pagine)
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide; nuova la", i + 1, "->", OUT)
