"""Revisione del 5 novembre (sera, 10) di FinalState.odp: i commenti «spiegala meglio con più step, punti con meno
testo» su tre slide; ognuna diventa una piccola serie di slide:
- «ModernBERT – 5 epoche, nessuna validazione»  → il ciclo, le impostazioni, che cosa succede;
- «ModernBERT – Che cosa comporta?»             → perché impara a memoria, errore sul test, ordine, conclusione;
- «Valutazione – Come si valuta un regressore»  → la distanza, errore medio ed entro 10/20, Spearman, senza testo.
Prima: figure/grafici_passi.py; dopo: aggiusta_animazioni.py. Uso:  python aggiorna_passi.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico, pagina

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zq"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
M, V = FIG / "modelli", FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


addestramento = [
    con_grafico("ModernBERT", "Come si addestra: un ciclo", [
        "Si parte da ModernBERT **già pre-addestrato** sull'inglese",
        "Legge un articolo e **prevede** un voto",
        "Lo confronta con il **voto della sua testata**",
        "Si **correggono un poco tutti i pesi**; poi l'articolo dopo",
    ], M / "bert_ciclo.png"),
    con_grafico("ModernBERT", "Le impostazioni", [
        "**5 epoche**: ogni articolo di training è letto 5 volte",
        "Una correzione ogni **32 articoli** (batch)",
        "**Learning rate** 3·10⁻⁵: quanto è grande la correzione",
        "**Gradiente tagliato a 1**: niente correzioni troppo grandi",
        "**Nessuna validazione**: si tiene l'ultima epoca",
    ], M / "bert_lr.png"),
    con_grafico("ModernBERT", "Che cosa succede", [
        "Sul training l'errore per articolo scende a ogni epoca: fino a **~4 punti**",
        "Sul test resta **~25 punti**",
    ], M / "bert_curva.png"),
]
comporta = [
    con_grafico("ModernBERT", "Perché impara a memoria", [
        "Ogni articolo ha come obiettivo il **voto della sua testata**",
        "~180 testate di training: solo **~180 voti diversi**",
        "Basta **riconoscere la testata** per indovinare il voto",
    ], M / "bert_memoria.png"),
    con_grafico("ModernBERT", "Su testate nuove sbaglia come il TF-IDF", [
        "Sul training: **4 punti** per articolo",
        "Sul test: **25 punti**, come il TF-IDF",
        "Mistral, che non si addestra: **22**",
    ], M / "bert_errore.png"),
    con_grafico("ModernBERT", "E ordina le testate peggio", [
        "Spearman **0,61**: il più basso dei quattro",
        "TF-IDF 0,69; Qwen 0,72; Mistral 0,73",
    ], M / "bert_ordine.png"),
    pagina("ModernBERT", "Che cosa comporta", [
        "Impara **chi scrive**, non quanto è affidabile",
        "Più epoche **non servono**",
        "Servirebbero **più testate**, non più articoli",
        "Qwen e Mistral restano **congelati**: si addestra solo il ridge sopra",
    ], corpo=24),
]
valutazione = [
    con_grafico("Valutazione", "Come si valuta un regressore: la distanza", [
        "**Nessuna classe**",
        "Per ogni testata: **voto vero** (NewsGuard) e **voto previsto**",
        "**Distanza** = quanti punti li separano",
    ], V / "esempio_distanza.png"),
    con_grafico("Valutazione", "Errore medio, entro 10 / 20 punti", [
        "**Errore medio** (MAE): la media delle 226 distanze",
        "**Entro 10 / 20 punti**: quante testate hanno una distanza fino a 10 / 20",
    ], V / "distanze.png"),
    con_grafico("Valutazione", "Spearman: l'ordine delle testate", [
        "Due classifiche delle stesse testate: per **voto vero** e per **voto previsto**",
        "Spearman = quanto si somigliano: **1** uguali, **0** nessun legame",
        "Guarda solo l'**ordine**, non le distanze",
    ], V / "classifiche.png"),
    con_grafico("Valutazione", "Il punto di partenza: senza testo", [
        "A tutte le testate lo **stesso voto**: la media del training (~68)",
        "È il «majority» dei regressori",
        "Un modello serve solo se fa **meglio di così**",
    ], V / "senza_testo.png"),
]

sostituzioni = {trova("ModernBERT", "5 epoche, nessuna validazione"): addestramento,
                trova("ModernBERT", "Che cosa comporta?"): comporta,
                trova("Valutazione", "Come si valuta un regressore"): valutazione}
finale = []
for i, p in enumerate(pagine):
    finale.extend(sostituzioni.get(i, [p]))
S.scrivi(IN, OUT, c, manifest, finale)
print(len(pagine), "→", len(finale), "slide; sostituite le", [i + 1 for i in sostituzioni], "->", OUT)
