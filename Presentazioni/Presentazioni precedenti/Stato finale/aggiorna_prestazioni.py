"""Revisione del 5 novembre (sera, 3) di FinalState.odp: le slide dei regressori (da «Voto vero e voto previsto» al
DNA, 9 slide generate e non toccate a mano) diventano 6: prestazioni del voto normale, dove sbagliano, l'idea dello
stiramento, normale e stirato a confronto, contro il TF-IDF (normale e stirato), il DNA (normale e stirato).
Prima: figure/grafici_regressori.py e figure/grafici_stirare.py. Le altre slide restano.
Uso:  python aggiorna_prestazioni.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zv"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 44, len(pagine)
V = FIG / "valutazione"
prima = next(i for i, p in enumerate(pagine) if "Regressori" in S.titolo_di(p) and "Voto vero e voto previsto" in S.testo(p))
ultima = next(i for i, p in enumerate(pagine) if "Il DNA" in S.titolo_di(p))
assert ultima - prima == 8, (prima, ultima)

nuove = [
    con_grafico("Regressori", "Le prestazioni, voto normale", [
        "Mistral il migliore: errore **15,7** punti, **69%** delle testate entro 20 punti, Spearman **0,73**",
        "Senza testo: errore 22,6, 51% entro 20 punti",
    ], V / "reg_prestazioni.png"),
    con_grafico("Regressori", "Dove sbagliano", [
        "Voti **schiacciati al centro**: le peggiori (0-39) sovrastimate di 31-37 punti, quelle da 100 sottostimate di 19-25",
    ], V / "reg_dove.png"),
    con_grafico("Regressori", "Stirare i voti", [
        "L'idea: **allargare** i voti, come il contrasto di una foto",
        "In ogni giro la testata col voto previsto più basso prende il voto più basso del training, la seconda il secondo, e così via",
    ], V / "stirare_idea.png"),
    con_grafico("Regressori", "Normale e stirato a confronto", [
        "Entro 20 punti: **da +7 a +10 punti %** con lo stirato (TF-IDF, Qwen, Mistral)",
        "Errore medio: da −0,5 a −1,4 punti, ma **dentro il caso**; Spearman uguale; a ModernBERT non serve",
    ], V / "stirare_confronto.png"),
    con_grafico("I modelli contro il TF-IDF", "Normale e stirato", [
        "Normale: Mistral e Qwen sbagliano **1,9 e 1,6 punti in meno** del TF-IDF, e non è il caso",
        "Stirato: il TF-IDF guadagna di più; il vantaggio scende a **1,0 punti** e non è più sicuro",
    ], V / "contro_tfidf2.png"),
    con_grafico("Il DNA", "Quanti articoli servono, normale e stirato", [
        "Mistral: con **5** articoli Spearman 0,71, con tutti 0,73",
        "Stirare conviene **con molti articoli**: con 1 l'errore cresce (17,1 → 18,1), con 50 scende (15,8 → 15,3)",
    ], V / "dna2.png"),
]
finale = pagine[:prima] + nuove + pagine[ultima + 1:]
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide (erano 44),", S.contatore[0], "punti nuovi ->", OUT)
