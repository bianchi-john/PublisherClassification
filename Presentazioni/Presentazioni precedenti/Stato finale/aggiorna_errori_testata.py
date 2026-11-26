"""Revisione del 5 novembre (sera, 14) di FinalState.odp: «come sbaglia per testata» (John: «aggiungi»). Tre slide
dopo «Regressori – Normale e stirato»: le stesse testate sono difficili per tutti i modelli; le più sovrastimate e le
più sottostimate; che cosa non conta (tipo di criteri persi, articoli, orientamento). Nessun nome di testata.
Prima: figure/grafici_errori_testata.py; dopo: aggiusta_animazioni.py.
Uso:  python aggiorna_errori_testata.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "ze"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"

(dopo,) = [i for i, p in enumerate(pagine) if "Regressori" in S.titolo_di(p) and "Normale e stirato" in S.testo(p)]
nuove = [
    con_grafico("Regressori", "Come sbaglia: le stesse testate per tutti", [
        "Ogni punto: l'errore di una testata con **Mistral** e con il **TF-IDF**",
        "Quasi sulla diagonale: sbagliano **sulle stesse testate** (correlazione 0,94)",
        "Il difficile sta nelle **testate**, non nel modello",
    ], V / "err_stesse.png"),
    con_grafico("Regressori", "Le testate più sbagliate", [
        "**Sovrastimate**: voto 0-17,5, con **contenuti falsi**; previste 42-69",
        "Il modello **non riconosce il falso**: servono i fatti, non lo stile",
        "**Sottostimate**: voto 70-100, quasi nessun criterio fallito; previste 41-71",
    ], V / "err_estreme.png"),
    con_grafico("Regressori", "Che cosa non conta", [
        "A parità di voto, perdere punti su criteri **non di testo** (proprietà, pubblicità, correzioni…) non fa sbagliare di più",
        "Numero di articoli e orientamento: differenze **piccole**",
    ], V / "err_non_conta.png"),
]
finale = pagine[:dopo + 1] + nuove + pagine[dopo + 1:]
S.scrivi(IN, OUT, c, manifest, finale)
print(len(pagine), "→", len(finale), "slide; nuove dalla", dopo + 2, "->", OUT)
