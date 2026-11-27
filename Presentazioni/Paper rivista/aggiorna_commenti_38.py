"""Trentottesima revisione (27 novembre) di PaperRivista.odp: l'appendice N (che cos'altro abbiamo provato) più breve e
mirata, come nel paper: a che cosa serve, la tabella delle alternative provate sulle 226 testate (tab:tried), il
risultato. Via la tabella dei 19 tentativi e le slide «dentro la pipeline».
Prima: compilare il paper e  python figure/schermate.py . Uso:
    python aggiorna_commenti_38.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "ym"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
AUX = QUI.parents[1] / "Paper" / "Journal" / "main.aux"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
numero = dict(re.findall(r"\\newlabel\{([a-z]+:[^}]+)\}\{\{([^}]+)\}", AUX.read_text()))

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
n = [i for i, x in enumerate(p) if "Appendice N" in S.titolo_di(x)]
assert n and n[-1] == len(p) - 1 and n == list(range(n[0], len(p))), n


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto in giù, tagliata ai lati sul contenuto."""
    a = np.array(Image.open(SCH / f"{nome}.png").convert("L")) < 128

    def tratto(riga):
        d = np.diff(np.concatenate([[0], riga.astype(int), [0]]))
        return (np.where(d == -1)[0] - np.where(d == 1)[0]).max() if riga.any() else 0

    tratti = [tratto(r) for r in a]
    alto = next(y for y, t in enumerate(tratti) if t >= 0.9 * max(tratti))
    sotto = a[alto:]
    col, rig = np.where(sotto.any(0))[0], np.where(sotto.any(1))[0]
    uscita = SCH / f"solo_{nome}.png"
    Image.open(SCH / f"{nome}.png").crop((max(col.min() - 6, 0), alto, col.max() + 6, alto + rig.max() + 6)).save(uscita)
    return uscita


def N(sotto, voci):
    return S.pagina("Appendice N", sotto, voci, 24, alto_testo=11.6)


img = solo_tabella("tab_tried")
w, h = Image.open(img).size
largo = min(25.0, 11.8 * w / h)
appendice_n = [
    N("A che cosa serve", [
        "Abbiamo sviluppato il metodo sulle **stesse** 226 testate dei risultati",
        "Domanda: i risultati dipendono da scelte **adattate** a loro?",
        "L'appendice elenca le **alternative provate** e che fine hanno fatto",
    ]),
    S.pagina("Appendice N", f"Tabella {numero['tab:tried']}", [],
             extra=S.immagine(img, (28 - largo) / 2, 3.5 + (11.8 - largo * h / w) / 2, largo=largo)),
    N("Il risultato", [
        "Il vantaggio dei modelli grandi su TF-IDF **regge** in tutte le alternative",
        "Scelti dentro i fold: solo **penalità** e **layer**",
        "I tentativi su altri dataset hanno fissato il protocollo: gruppi di proprietà, testate politiche, voti 2026",
        "Non è un test su testate **nuove**: lo diciamo",
    ]),
]
nuove = p[:n[0]] + appendice_n

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano {len(p)})")
