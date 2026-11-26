"""Revisione del 5 novembre (sera, 4) di FinalState.odp: i risultati dei regressori come curve (misure al crescere
degli articoli letti per testata), senza intervalli. Le 6 slide da «Le prestazioni, voto normale» al DNA diventano 4:
curve del voto normale, «Dove sbagliano» e «Stirare i voti» (le slide di prima, invariate), curve normale e stirato.
Prima: figure/grafici_curve.py. Le altre slide restano. Uso:  python aggiorna_curve.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zw"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 41, len(pagine)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


prima, dove, stirare, ultima = (trova("Regressori", "Le prestazioni, voto normale"), trova("Regressori", "Dove sbagliano"),
                                trova("Regressori", "Stirare i voti"), trova("Il DNA", "normale e stirato"))
assert (dove, stirare, ultima) == (prima + 1, prima + 2, prima + 5)
nuove = [
    con_grafico("Regressori", "Le prestazioni, articolo dopo articolo", [
        "Ogni punto: **k articoli a caso** per testata (20 volte); «tutti» = il risultato finale",
        "Mistral il migliore (Spearman **0,73**, errore **15,7**); dopo ~10 articoli le curve sono **piatte**",
    ], V / "curve_normale.png"),
    S.senza_commenti(pagine[dove]),
    S.senza_commenti(pagine[stirare]),
    con_grafico("Regressori", "Normale e stirato", [
        "Stirato (tratteggiato): meglio **solo con molti articoli**; lo Spearman non cambia",
        "Mistral: con tutti, entro 20 punti **69% → 75%**; con 1 articolo, errore **17,1 → 18,1**",
    ], V / "curve_confronto.png"),
]
finale = pagine[:prima] + nuove + pagine[ultima + 1:]
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide (erano 41),", S.contatore[0], "punti nuovi ->", OUT)
