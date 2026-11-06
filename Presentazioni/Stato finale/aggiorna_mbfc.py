"""Revisione del 6 novembre (2) di FinalState.odp: la parte MBFC più semplice, come chiesto da John.
Le slide da «Un secondo esperto» a «Tre confronti sulle stesse 170 testate» (6) diventano 4 (figure/grafici_mbfc.py):
le etichette delle 170 testate in comune; da 6 classi a 4 e l'abbinamento con NewsGuard; MBFC contro NewsGuard
(Spearman, Kendall); MBFC e il regressore sulle stesse testate. La slide delle curve che segue resta com'è.
Uso:  python aggiorna_mbfc.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zm"
S.animazione = lambda ids: ""          # John ha tolto tutte le animazioni
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


da, a = trova("Media Bias/Fact Check", "Un secondo esperto"), trova("Il modello come annotatore", "Tre confronti")
nuove = [
    con_grafico("Media Bias/Fact Check", "Le etichette delle 170 testate in comune", [
        "MBFC dà a ogni testata una classe di **Factual Reporting** (quanto riporta i fatti correttamente), non un voto",
        "Stesso dominio: **170** delle nostre 226 testate; MBFC non entra mai nell'addestramento",
        "NewsGuard è spostato in alto (57% sopra 75); in MBFC la classe più grande è «Mixed» (38%)",
    ], V / "mbfc_distribuzioni.png"),
    con_grafico("Media Bias/Fact Check", "Da 6 classi a 4, e l'abbinamento con NewsGuard", [
        "Very Low e Very High hanno 1 testata ciascuna: unite a **Low** e **High** → 4 classi",
        "NewsGuard: l4 e l5 insieme (75-100) → 4 fasce",
        "Abbinamento: Low ↔ l1, Mixed ↔ l2, Mostly Factual ↔ l3, High ↔ l4-l5",
    ], V / "mbfc_adattamento.png"),
    con_grafico("NewsGuard e MBFC", "Quanto concordano", [
        "MBFC dà la classe (Low = 1 … High = 4), NewsGuard il voto vero: le nostre misure sull'ordine",
        "Spearman **0,71**, Kendall **0,59**",
        "Stessa fascia solo 84 su 170: «Mostly Factual» in NewsGuard è quasi sempre sopra 75",
    ], V / "mbfc_incrocio.png", largo_max=16),
    con_grafico("NewsGuard e MBFC", "MBFC e il nostro regressore, sulle stesse 170 testate", [
        "Il voto stirato del regressore va nelle stesse 4 fasce: **stessa grana** di MBFC",
        "Mistral concorda con NewsGuard **quanto MBFC** (0,71 e 0,59); il TF-IDF un po' meno; differenze non significative",
        "Con il voto pieno, senza fasce, Mistral arriva a 0,74 di Spearman (slide dopo)",
    ], V / "mbfc_confronto.png", largo_max=18),
]
finale = pagine[:da] + nuove + pagine[a + 1:]
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), f"slide (erano {len(pagine)}): tolte {da + 1}-{a + 1}, inserite {len(nuove)} ->", OUT)
