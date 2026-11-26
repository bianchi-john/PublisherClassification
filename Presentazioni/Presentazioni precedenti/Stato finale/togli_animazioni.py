"""Toglie TUTTE le animazioni da FinalState.odp (presentazione interna: ogni slide compare intera, senza clic).
Per rimetterle: aggiusta_animazioni.py. Uso:  python togli_animazioni.py <file di John.odp> <uscita.odp>"""
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
finale = [re.sub(r'<anim:par[^>]*presentation:node-type="timing-root".*?</anim:seq></anim:par>', "", p, flags=re.S) for p in pagine]
assert not any("<anim:" in p for p in finale)
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide, tolte le animazioni di", sum(p != q for p, q in zip(pagine, finale)), "->", OUT)
