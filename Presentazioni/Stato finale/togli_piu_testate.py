"""Revisione del 10 novembre di FinalState.odp: John toglie dalla narrazione la prova con più testate nel training
(slide «Regressori – Più testate nel training»). Le altre slide restano com'erano.
Uso:  python togli_piu_testate.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
tenute = [p for p in pagine if not ("Regressori" in S.titolo_di(p) and "Più testate nel training" in S.testo(p))]
assert len(tenute) == len(pagine) - 1
S.scrivi(IN, OUT, c, manifest, tenute)
print(len(tenute), f"slide (erano {len(pagine)}) ->", OUT)
