"""Trentacinquesima revisione (27 novembre) di PaperRivista.odp: come nel paper, l'appendice sui tentativi precedenti va
in fondo e diventa N (l'orientamento diventa M), con la tabella nuova dei 19 tentativi (tab:attempts) divisa in due
slide (tentativi 1-7, 8-13 e 14-19, con l'intestazione ripetuta). La critica si corregge: solo i tentativi 16-19 usano le
226 testate del paper.
Prima: compilare il paper e  python figure/schermate.py . Uso:
    python aggiorna_commenti_35.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "yj"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
AUX = QUI.parents[1] / "Paper" / "Journal" / "main.aux"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
numero = dict(re.findall(r"\\newlabel\{([a-z]+:[^}]+)\}\{\{([^}]+)\}", AUX.read_text()))
assert numero["app:orientation"] == "M" and numero["app:tried"] == "N"

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
tentativi = [i for i, x in enumerate(p) if "Appendice M" in S.titolo_di(x)]
orient = [i for i, x in enumerate(p) if "Appendice N" in S.titolo_di(x)]
assert len(tentativi) == 4 and len(orient) == 6 and orient[0] == tentativi[-1] + 1 and orient[-1] == len(p) - 1


def parti(nome, tagli=(8, 14)):
    """La tabella senza didascalia, divisa in più immagini prima delle righe numero `tagli`; l'intestazione (fino al primo
    numero della prima colonna) si ripete in ogni parte."""
    im = Image.open(SCH / f"{nome}.png").convert("L")
    a = np.array(im) < 128
    righe_piene = a.mean(1) > 0.5                                    # i filetti
    alto = int(np.argmax(righe_piene))                               # il primo filetto: sopra c'è la didascalia
    a, im = a[alto:], im.crop((0, alto, im.width, im.height))
    sinistra = a[:, : int(0.04 * a.shape[1])].any(1) & ~(a.mean(1) > 0.5)  # i numeri della prima colonna
    inizi = [y for y in range(1, len(sinistra)) if sinistra[y] and not sinistra[y - 1]]
    assert len(inizi) == 19, len(inizi)
    testa = im.crop((0, 0, im.width, inizi[0] - 4))
    confini = [inizi[0] - 4] + [inizi[t - 1] - 4 for t in tagli] + [im.height]
    uscite = []
    for k, (y0, y1) in enumerate(zip(confini[:-1], confini[1:]), 1):
        corpo = im.crop((0, y0, im.width, y1))
        img = Image.new("L", (im.width, testa.height + corpo.height), 255)
        img.paste(testa, (0, 0)); img.paste(corpo, (0, testa.height))
        box = Image.eval(img, lambda v: 255 - v).getbbox()
        u = SCH / f"solo_{nome}_{k}.png"
        img.crop((max(box[0] - 6, 0), 0, box[2] + 6, box[3] + 6)).save(u)
        uscite.append(u)
    return uscite


def tabella(sotto, img):
    w, h = Image.open(img).size
    largo = min(25.0, 11.8 * w / h)
    return S.pagina("Appendice N", sotto, [],
                    extra=S.immagine(img, (28 - largo) / 2, 3.5 + (11.8 - largo * h / w) / 2, largo=largo))


def N(sotto, voci):
    return S.pagina("Appendice N", sotto, voci, 24, alto_testo=11.6)


pezzi = parti("tab_attempts")
t = numero["tab:attempts"]
dentro_1 = p[tentativi[1]].replace("Appendice M", "Appendice N").replace("Che cosa abbiamo provato (1/2)",
                                                                          "Dentro la pipeline (1/2)")
dentro_2 = p[tentativi[2]].replace("Appendice M", "Appendice N").replace("Che cosa abbiamo provato (2/2)",
                                                                          "Dentro la pipeline (2/2)")
risposta = p[tentativi[3]].replace("Appendice M", "Appendice N")
assert risposta.count(" in appendice M") == 1
risposta = risposta.replace(" in appendice M", " in appendice N")       # il testo è spezzato dal grassetto
appendice_n = [
    N("I tentativi precedenti: la critica", [
        "«Avete **adattato** le scelte alle 226 testate?»",
        "19 tentativi prima della pipeline",
        "Gli ultimi 4 sulle **stesse** 226 testate, di cui abbiamo visto i risultati",
    ]),
    N("I 19 tentativi", [
        "Ottobre–novembre 2026, uno dopo l'altro",
        "**1–15**: altri dataset (altri scraping, altre etichette), 5 livelli",
        "**16–19**: le 226 testate del paper",
        "Il **20** è la pipeline del paper",
    ]),
    *[tabella(f"Tabella {t} ({k}/{len(pezzi)})", img) for k, img in enumerate(pezzi, 1)],
    dentro_1, dentro_2, risposta,
]
appendice_m = [x.replace("Appendice N", "Appendice M") for x in p[orient[0]:]]
nuove = p[:tentativi[0]] + appendice_m + appendice_n
nuove = [x.replace("(appendice M)", "(appendice N)") for x in nuove]       # il rimando nell'appendice E

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano {len(p)})")
