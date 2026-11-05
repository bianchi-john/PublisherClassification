"""Revisione del 5 novembre (sera, 9) di FinalState.odp: risposte ai commenti di John.
- dopo «Modelli lineari – Le penalità provate e quelle scelte»: «Che cosa comporta?» (figure/modelli/penalita_conta.png);
- dopo «ModernBERT – 5 epoche, nessuna validazione»: «Che cosa comporta?» (figure/modelli/bert_comporta.png);
- tolti i due commenti. Prima: figure/grafici_commenti.py; dopo: aggiusta_animazioni.py.
Uso:  python aggiorna_commenti.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zc"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


penalita = con_grafico("Modelli lineari", "Che cosa comporta? La scelta migliora i risultati?", [
    "Una penalità **sbagliata** costa: fino a 0,16 di Spearman per Qwen, 0,09 per Mistral; per il TF-IDF quasi niente (0,02)",
    "Scegliere giro per giro dà quanto la **migliore penalità fissa**: non aggiunge, evita di sbagliare",
    "Il vantaggio di Qwen e Mistral sul TF-IDF **non viene dalla penalità**: il TF-IDF non migliora con nessuna penalità della griglia",
], FIG / "modelli" / "penalita_conta.png")
bert = con_grafico("ModernBERT", "Che cosa comporta?", [
    "Sul training **riconosce le testate**: a ogni articolo dà quasi il voto della sua testata",
    "Su testate nuove sbaglia **come il TF-IDF** e le ordina **peggio**: impara **chi scrive**, non quanto è affidabile",
    "Servirebbero **più testate**, non più epoche: i voti da imparare sono solo ~180, uno per testata",
], FIG / "modelli" / "bert_comporta.png")

i_pen = trova("Modelli lineari", "Le penalità provate e quelle scelte")
i_bert = trova("ModernBERT", "5 epoche, nessuna validazione")
pagine[i_pen] = S.senza_commenti(pagine[i_pen])
pagine[i_bert] = S.senza_commenti(pagine[i_bert])
nuove = {i_pen: penalita, i_bert: bert}
finale = []
for i, p in enumerate(pagine):
    finale.append(p)
    if i in nuove:
        finale.append(nuove[i])
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide; nuove le", i_pen + 2, "e", i_bert + 3, "->", OUT)
