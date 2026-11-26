"""Revisione del 5 novembre (sera, 6) di FinalState.odp:
- «Il modello come annotatore» e «Classificatore o regressore?» come curve al crescere degli articoli, voto stirato
  (figure/grafici_finali.py); l'accordo NewsGuard ↔ MBFC è una linea;
- slide «Normale e stirato»: frase sullo stiramento corretta (vale per l'errore e per «entro 20 punti», non per l'ordine).
Le slide toccate erano generate e non modificate a mano. Uso:  python aggiorna_finali.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zy"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


pagine[trova("Regressori", "Normale e stirato")] = con_grafico("Regressori", "Normale e stirato", [
    "Stirato (tratteggiato): meglio **solo con molti articoli**; lo Spearman non cambia",
    "Con tutti gli articoli il TF-IDF, più schiacciato, ci guadagna di più: il vantaggio di Mistral sull'errore scende da 1,9 a 1,0 punti, **non più significativo**",
], V / "curve_confronto.png")
pagine[trova("Il modello come annotatore", "Senza classi")] = con_grafico("Il modello come annotatore", "Al crescere degli articoli, voto stirato", [
    "Linea tratteggiata: i due esperti tra loro (NewsGuard ↔ MBFC, **0,71**)",
    "Con NewsGuard Mistral raggiunge gli esperti già con **5 articoli** e li supera con tutti (**0,74**)",
    "Con MBFC un po' meno (0,65): il modello ha imparato NewsGuard. Cerchiati: battono il TF-IDF",
], V / "curve_mbfc.png")
pagine[trova("Classificatore o regressore?", "")] = con_grafico("Classificatore o regressore?", "Pieno: regressore stirato · tratteggiato: classificatore", [
    "Ordine, entro 20 punti e livelli: il regressore è **pari o meglio** quasi sempre",
    "Errore medio: il classificatore è un po' meglio per Mistral (**14,8** contro 15,2 con tutti gli articoli)",
], V / "curve_cls_reg.png")
S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide,", S.contatore[0], "punti nuovi ->", OUT)
