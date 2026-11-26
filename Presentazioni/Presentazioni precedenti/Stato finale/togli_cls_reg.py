"""Revisione del 6 novembre (4) di FinalState.odp: John toglie il confronto classificatore contro regressore, cioè
tutte le slide «Classificatore o regressore?» (le altre restano com'erano).
Uso:  python togli_cls_reg.py <file di John.odp> <file in uscita.odp>"""
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
tenute = [p for p in pagine if "Classificatore o regressore?" not in S.titolo_di(p)]
S.scrivi(IN, OUT, c, manifest, tenute)
print(len(tenute), f"slide (erano {len(pagine)}) ->", OUT)
