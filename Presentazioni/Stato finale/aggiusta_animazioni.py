"""Rifà le animazioni di TUTTE le slide di FinalState.odp: ogni punto (paragrafo non vuoto) dei riquadri di testo
«outline» compare con un clic, nell'ordine in cui è scritto; titoli, sottotitoli e immagini sono sempre visibili.
I paragrafi senza xml:id ne ricevono uno nuovo; le sequenze che c'erano (anche quelle fatte a mano) si sostituiscono.
Uso:  python aggiusta_animazioni.py <file di John.odp> <file in uscita.odp>"""
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
nuovo = [0]


def con_id(m, ids):
    """Un paragrafo: se ha testo, prende (o tiene) un xml:id e si aggiunge ai clic."""
    apertura, corpo = m.group(1), m.group(2)
    if not S.testo(corpo):
        return m.group(0)
    pid = re.search(r'xml:id="([^"]+)"', apertura)
    if pid:
        pid = pid.group(1)
    else:
        nuovo[0] += 1
        pid = f"za{nuovo[0]}"
        apertura = apertura[:-1] + f' xml:id="{pid}" text:id="{pid}">'
    ids.append(pid)
    return apertura + corpo + "</text:p>"


def aggiusta(p):
    note = re.search(r"<presentation:notes.*?</presentation:notes>", p, flags=re.S)
    note = note.group(0) if note else ""
    p = p.replace(note, "")
    p = re.sub(r'<anim:par[^>]*presentation:node-type="timing-root".*?</anim:seq></anim:par>', "", p, flags=re.S)
    assert "<anim:" not in p, "animazioni fuori dalla sequenza principale"
    ids = []

    def riquadro(f):
        testa = f.group(0)[:f.group(0).find(">")]
        if 'presentation:class="outline"' not in testa:
            return f.group(0)
        return re.sub(r"(<text:p(?: [^>]*)?>)(.*?)</text:p>", lambda m: con_id(m, ids), f.group(0), flags=re.S)

    p = re.sub(r"<draw:frame .*?</draw:frame>", riquadro, p, flags=re.S)
    return p.replace("</draw:page>", S.animazione(ids) + note + "</draw:page>"), len(ids)


finale, clic = [], []
for p in pagine:
    q, n = aggiusta(p)
    finale.append(q); clic.append(n)
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide,", sum(clic), "clic,", nuovo[0], "paragrafi con id nuovo ->", OUT)
print("clic per slide:", clic)
