"""Revisione del 5 novembre (sera, 2) di FinalState.odp: i commenti di John sulle slide 28-42 e due slide nuove:
- «Previsto o stirato?» (figure/valutazione/stirare_confronto.png), dopo il codice dello stiramento;
- «Classificatore o regressore?» (figure/valutazione/cls_vs_reg.png), in fondo.
Le slide commentate erano generate da aggiorna_regressori.py e non toccate a mano: si rigenerano. Le altre restano.
Prima: figure/grafici_stirare.py. Uso:  python aggiorna_stirare.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_codice, con_grafico, figura, immagine, pagina

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zu"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 42, len(pagine)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


# ------------------------------------------------------------------------------------------------ slide rigenerate
nuove = {}
nuove[trova("Valutazione", "Dagli articoli alla testata")] = con_codice("Valutazione", "Dagli articoli alla testata", [
    "Ogni articolo ha un voto previsto; la testata = la **media** dei voti dei suoi articoli",
    ("Si usano **tutti** gli articoli della testata, nessuno escluso", ["da 10 a 201 per testata, in media 113"]),
    "Ogni testata è prevista **una volta sola**, nel giro in cui era nel test",
    "Tutte le misure sono **per testata** (226), non per articolo",
], ["articoli_testata_reg"])
nuove[trova("Valutazione", "Come si valuta un regressore")] = con_grafico("Valutazione", "Come si valuta un regressore", [
    "Nessuna classe: per ogni testata guardiamo **quanto dista** il voto previsto da quello NewsGuard",
    "**Errore medio** (MAE): la media di queste distanze, in punti",
    "**Entro 10 / 20 punti**: quante testate hanno una distanza fino a 10 / 20 punti",
    "**Spearman**: classifica delle testate per voto vero e per voto previsto; 1 = stessa classifica, 0 = nessun legame",
    "**Senza testo**: a tutte lo stesso voto, la media del training (~68); come la classe più frequente per le classi",
], V / "esempio_distanza.png")
nuove[trova("Regressori", "Quante testate sono vicine")] = con_grafico("Regressori", "Quante testate sono vicine al voto vero", [
    "Entro **10 punti**: Mistral 37%, Qwen 35%, TF-IDF 32%; senza testo 18%",
    "Entro **20 punti**: Mistral e Qwen 69%, TF-IDF 61%; senza testo 51%",
    "**Senza testo** = non leggere niente: a ogni testata la media dei voti del training (~68), il «majority» del voto",
], V / "reg_entro.png")
nuove[trova("Regressori", "Stirare i voti")] = con_grafico("Regressori", "Stirare i voti aiuta?", [
    "Il problema: i voti previsti sono **schiacciati al centro**, e agli estremi sbagliamo di più",
    "L'idea: **allargarli**, come si alza il contrasto di una foto",
    "In ogni giro: la testata con il voto previsto più basso prende il voto più basso del training, la seconda il secondo, e così via",
], V / "stirare_dispersione.png")
nuove[trova("Regressori", "Il codice dello stiramento")] = pagina("Regressori", "Come una foto: lo stesso codice, prima e dopo", [],
    extra=figura(FIG / "valutazione" / "stirare_immagine.png", 3.3, basso=11.9) + immagine(FIG / "codice" / "stirare.png", (28 - 16.6) / 2, 12.05, largo=16.6))
nuove[trova("Il modello come annotatore", "Senza classi")] = con_grafico("Il modello come annotatore", "Senza classi: lo Spearman", [
    "Il **voto previsto** dal regressore, confrontato con il voto NewsGuard e con la classe MBFC",
    "Mistral con NewsGuard **0,73**, come i due esperti tra loro (**0,71**)",
    "Con MBFC un po' meno (**0,63**): il modello ha imparato NewsGuard",
    "È il voto **non stirato**; stirato cambia appena (Mistral con MBFC 0,63 → 0,65)",
], V / "mbfc_accordo.png")
i41 = trova("NewsGuard e MBFC", "Quanto concordano")
nuove[i41] = S.senza_commenti(pagine[i41])          # il commento era vuoto

# ------------------------------------------------------------------------------------------------ slide nuove
previsto_o_stirato = con_grafico("Regressori", "Previsto o stirato?", [
    "Entro 20 punti: **da +7 a +10 punti %** con lo stirato (TF-IDF, Qwen, Mistral)",
    "Errore medio: da −0,5 a −1,4 punti, ma **dentro il caso**; Spearman uguale",
    "A ModernBERT non serve",
], V / "stirare_confronto.png")
cls_reg = con_grafico("Classificatore o regressore?", "Nessuno vince nettamente", [
    "Classificatore → voto: **voto atteso**; regressore → livello: **quote** del training",
    "Sul voto: il regressore semplice ~1 punto peggio (TF-IDF, Mistral), **stirato pari**",
    "Sui livelli: F1 **pari**; ordine dei livelli **migliore con il regressore**",
], V / "cls_vs_reg.png")

codice = trova("Regressori", "Il codice dello stiramento")
finale = []
for i, p in enumerate(pagine):
    finale.append(nuove.get(i, p))
    if i == codice:
        finale.append(previsto_o_stirato)
finale.append(cls_reg)
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide (erano 42),", S.contatore[0], "punti nuovi ->", OUT)
