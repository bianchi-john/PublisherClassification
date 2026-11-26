"""Commenti di John del 6 novembre (3) su FinalState.odp:
- «Il modello come annotatore» (curve con NewsGuard sulle 170 testate in comune): «sono i regressori stirati?» Sì:
  il grafico ritagliato da John (senza legenda) diventa curve_annotatore.png, un riquadro solo, con legenda e
  «stirato» nel titolo; stessa altezza e stessa posizione verticale, centrato;
- «Come si confrontano»: «spiega meglio i tagli alle quote del training (perché sopra 80 è l5?)» → slide nuova subito
  dopo (quote_tagli.png). Figure: figure/grafici_commenti_mbfc.py. I due commenti risolti si tolgono.
Uso:  python aggiorna_commenti_mbfc.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zq"
S.animazione = lambda ids: ""          # niente animazioni
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(S.senza_commenti(p))]
    return i


# 1. il modello come annotatore: il grafico intero, un riquadro solo, senza ritaglio
i = trova("Il modello come annotatore", "Al crescere degli articoli")
p = S.senza_commenti(pagine[i])
m = re.search(r'<draw:frame draw:style-name="gr\d+"([^>]*)svg:width="([\d.]+)cm" svg:height="([\d.]+)cm" svg:x="([\d.]+)cm" '
              r'svg:y="([\d.]+)cm"><draw:image xlink:href="([^"]+)"', p)
alto, y = float(m.group(3)), float(m.group(5))
percorso = V / "curve_annotatore.png"
w, h = Image.open(percorso).size
alto = min(alto * 1.25, 15.3 - y)
largo = alto * w / h
nuovo = (f'<draw:frame draw:style-name="{S.IMMAGINE}"{m.group(1)}svg:width="{largo:.2f}cm" svg:height="{alto:.2f}cm" '
         f'svg:x="{(28 - largo) / 2:.2f}cm" svg:y="{y:.2f}cm"><draw:image xlink:href="{S.nome_immagine(percorso)}"')
pagine[i] = p.replace(m.group(0), nuovo)

# 2. i tagli alle quote: commento tolto, slide nuova subito dopo
j = trova("Classificatore o regressore?", "Come si confrontano")
pagine[j] = S.senza_commenti(pagine[j])
pagine.insert(j + 1, con_grafico("Classificatore o regressore?", "Regressore → livello: i tagli alle quote", [
    "Giro 1: nel training il 17% è l5 → nel test è l5 il 17% più alto: **8 su 45**",
    "L'ottava ha previsto 79,6 («sopra 80»): **non è fisso**, negli altri giri 80-87",
    "Con i confini veri sul voto previsto: **mai l5**, quasi mai l1 (barre arancioni)",
], V / "quote_tagli.png"))
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide ->", OUT)
