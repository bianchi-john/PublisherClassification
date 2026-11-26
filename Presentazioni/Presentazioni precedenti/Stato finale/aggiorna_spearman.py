"""Revisione del 5 novembre (sera, 11) di FinalState.odp: commento «spiegalo meglio con altri esempi» sulla slide
«Valutazione – Spearman: l'ordine delle testate». Diventa quattro slide: l'idea (un solo scambio), quattro casi
(1, 0,9, 0, −1), conta solo l'ordine (due modelli inventati), un esempio vero (6 testate, Mistral).
Le slide eliminate da John NON si rimettono: lo script tocca solo questa. Prima: figure/grafici_spearman.py;
dopo: aggiusta_animazioni.py. Uso:  python aggiorna_spearman.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zs"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"

(i,) = [i for i, p in enumerate(pagine) if "Valutazione" in S.titolo_di(p) and "Spearman: l'ordine delle testate" in S.testo(p)]
nuove = [
    con_grafico("Valutazione", "Spearman: due classifiche", [
        "Si mettono le testate **in fila** due volte: per voto **vero** e per voto **previsto**",
        "Si confrontano le **posizioni**, non i voti",
        "Qui C e D si sono scambiate: Spearman **0,9**",
    ], V / "spearman_idea.png"),
    con_grafico("Valutazione", "Spearman: quattro casi", [
        "**1**: stessa classifica",
        "**0**: nessun legame tra le due classifiche",
        "**−1**: classifica rovesciata",
    ], V / "spearman_casi.png"),
    con_grafico("Valutazione", "Spearman guarda solo l'ordine", [
        "Modello 1: voti tutti tra 55 e 65, ma **in ordine**: Spearman 1, errore medio 21,8",
        "Modello 2: voti **vicini**, ma due scambi: Spearman 0,8, errore medio 16,4",
        "Per questo guardiamo **tutte e due** le misure",
    ], V / "spearman_ordine.png"),
    con_grafico("Valutazione", "Spearman: un esempio vero", [
        "6 testate vere, voto previsto da **Mistral**",
        "Linee incrociate = scambi: Spearman **0,77**; su tutte le 226 testate **0,73**",
    ], V / "classifiche.png"),
]
finale = pagine[:i] + nuove + pagine[i + 1:]
S.scrivi(IN, OUT, c, manifest, finale)
print(len(pagine), "→", len(finale), "slide; la", i + 1, "diventa", len(nuove), "->", OUT)
