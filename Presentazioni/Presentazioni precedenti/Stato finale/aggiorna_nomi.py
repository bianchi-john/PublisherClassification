"""Revisione del 5 novembre (sera, 12) di FinalState.odp: i nomi scelti da John.
- «senza testo»  → «baseline: voto medio»;
- «entro 10 / 20 punti» → «% testate con errore ≤ 10 / 20», con l'esempio (vero 80 → previsto tra 60 e 100).
Slide: «Errore medio, entro 10 / 20 punti» e «Il punto di partenza: senza testo» rifatte (tolti i commenti), la frase
di «Classificatore o regressore?» riscritta; immagini nuove dove il nome compare nel grafico (curve_normale,
curve_confronto, curve_cls_reg). Prima: figure/grafici_passi.py, grafici_curve.py, grafici_finali.py;
dopo: aggiusta_animazioni.py. Uso:  python aggiorna_nomi.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zn"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


pagine[trova("Valutazione", "Errore medio, entro 10 / 20 punti")] = con_grafico("Valutazione", "Errore medio e % testate con errore ≤ 10 / 20", [
    "**Errore medio** (MAE): la media delle 226 distanze",
    ("**% testate con errore ≤ 20**: quante testate hanno il voto previsto a non più di 20 punti dal vero",
     ["es. voto vero 80: va bene un previsto tra 60 e 100"]),
    "Lo stesso con **10 punti**: più severo",
], V / "distanze.png")
pagine[trova("Valutazione", "Il punto di partenza: senza testo")] = con_grafico("Valutazione", "Il punto di partenza: la baseline «voto medio»", [
    "**Baseline: voto medio**: a tutte le testate lo stesso voto, la media del training (~68)",
    "È il «majority» dei regressori",
    "Un modello serve solo se fa **meglio di così**",
], V / "senza_testo.png")

i = trova("Classificatore o regressore?", "")
pagine[i] = S.riscrivi(pagine[i], "entro 20 punti e livelli",
                       "Ordine, % con errore ≤ 20, livelli: il regressore è **pari o meglio**")
pagine[i] = S.cambia_immagine(pagine[i], V / "curve_cls_reg.png")
pagine[trova("Regressori", "Ogni punto: k articoli a caso")] = S.cambia_immagine(pagine[trova("Regressori", "Ogni punto: k articoli a caso")], V / "curve_normale.png")
j = trova("Regressori", "Normale e stirato")
pagine[j] = S.cambia_immagine(pagine[j], V / "curve_confronto.png")

for k, p in enumerate(pagine):
    t = S.testo(re.sub(r"<presentation:notes.*?</presentation:notes>", "", p, flags=re.S))
    assert not re.search(r"senza testo|[Ee]ntro (10|20)", t), (k + 1, t[:200])
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide ->", OUT)
