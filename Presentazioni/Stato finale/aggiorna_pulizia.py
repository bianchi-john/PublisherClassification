"""Revisione del 5 novembre (pomeriggio) di FinalState.odp, sulle slide di scraping e pulizia:
- slide 4: al posto del catalogo NewsGuard intero, solo le 1.293 testate mandate allo scraper (figure/selezione_scraping.png);
- slide 6: i doppioni tolgono quasi solo testate sorelle (stesso gruppo di proprietà);
- slide 7 e una slide nuova dopo: le sorelle passo per passo (figure/sorelle.png, da Dataset/tentativi/20_FromScratch/sorelle.csv).
Tutte le altre slide restano come sono. Prima: figure/genera_grafici.py e figure/grafico_sorelle.py.
Uso:  python aggiorna_pulizia.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

import stile_slide as S
from stile_slide import FIG, con_grafico, pagina

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zs"
z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
c = S.aggiungi_stili(c)
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 54, len(pagine)
t = pd.read_csv(FIG.parents[2] / "Dataset/tentativi/20_FromScratch/sorelle.csv").set_index("passo")
sorelle = t["sorelle (testate - gruppi)"]
assert (t.testate["senza doppioni"], t.testate["dal 2025"], t.testate["almeno 10 articoli"], t.testate["una per gruppo"]) == (361, 531, 287, 226)
tolte_doppioni = t.testate["dal 2025"] - t.testate["senza doppioni"]                         # 170 testate
sorelle_doppioni = sorelle["dal 2025"] - sorelle["senza doppioni"]                           # di cui sorelle


def numero(n, titolo, contiene):
    assert titolo in S.titolo_di(pagine[n - 1]) and contiene in S.testo(pagine[n - 1]), n
    return S.senza_commenti(pagine[n - 1])


# slide 4: le testate mandate allo scraper (via l'immagine del catalogo intero e il commento vuoto)
p4 = numero(4, "I dati selezionati", "")
vecchia = re.search(r'<draw:frame draw:style-name="gr\d+"[^>]*><draw:image xlink:href="Pictures/[^"]+".*?</draw:frame>', p4, flags=re.S).group(0)
pagine[3] = p4.replace(vecchia, S.figura(FIG / "selezione_scraping.png", 3.6))

# slide 6: i doppioni tolgono le sorelle
p6 = numero(6, "Pulizia", "Passo per passo")
p6 = S.riscrivi(p6, "Testate: il taglio più grande sono i", f"Testate: il taglio più grande sono i **doppioni** (531 → 361): **{sorelle_doppioni} su {tolte_doppioni}** sorelle", corpo=20)
p6 = S.riscrivi(p6, "Infine una testata per gruppo", f"Infine una per gruppo di proprietà: da 287 a **226** (le ultime {sorelle['almeno 10 articoli']} sorelle)", corpo=20)
pagine[5] = p6

# slide 7: i proprietari, e una slide nuova con le sorelle passo per passo
numero(7, "Pulizia", "I proprietari")
pagine[6] = pagina("Pulizia", "I proprietari: decisi solo qui", [
    "Allo scraping non guardiamo i proprietari: **prima prendiamo i dati, poi decidiamo cosa togliere**",
    (f"Le 1.293 testate mandate allo scraper stanno in **589 gruppi**: **{sorelle['scaricate (1.293)']}** sono sorelle",
     ["il gruppo più grande è una rete di 186 siti con lo stesso Parent Domain"]),
    "La pulizia toglie **quasi tutte le sorelle da sola**, soprattutto con i **doppioni**",
    f"Restano 287 testate in 226 gruppi: la regola «una per gruppo» toglie solo le **ultime {sorelle['almeno 10 articoli']}**",
])
passo_per_passo = con_grafico("Pulizia", "Le sorelle passo per passo", [
    f"Scraping: da {sorelle['scaricate (1.293)']} a **{sorelle['almeno un articolo']}** sorelle (molti siti delle reti non si scaricano)",
    f"Doppioni: da {sorelle['dal 2025']} a **{sorelle['senza doppioni']}**: delle {tolte_doppioni} testate tolte, **{sorelle_doppioni}** sono sorelle",
    f"Una per gruppo: le ultime **{sorelle['almeno 10 articoli']}**",
], FIG / "sorelle.png")
pagine.insert(7, passo_per_passo)

S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide (erano 54),", S.contatore[0], "punti nuovi ->", OUT)
