"""Revisione del 5 novembre (sera, 13) di FinalState.odp: commenti a voce di John.
1. Spearman rispiegato con testate VERE: «Spearman: due classifiche» (esempio inventato) e «Spearman: un esempio vero»
   (6 testate) sono sostituite da tre slide sulle stesse 8 testate vere (voti → posizioni → conto); restano
   «quattro casi» e «guarda solo l'ordine».
2. Prima de «Il modello come annotatore»: come si confronta un voto NewsGuard con una classe MBFC (posizioni, pari
   merito) e i tre confronti sulle stesse 170 testate; nella slide delle curve 0,65 → 0,64 (il valore è 0,645).
3. «Classificatore o regressore?»: classificatore pieno, regressore stirato tratteggiato (come lo stirato altrove).
Prima: figure/grafici_annotatore.py e figure/grafici_finali.py; dopo: aggiusta_animazioni.py.
Uso:  python aggiorna_annotatore.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zm"
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


# 1. Spearman su testate vere
spearman_vero = [
    con_grafico("Valutazione", "Spearman su testate vere: i voti", [
        "8 testate vere del test; voto previsto da **Mistral**",
        "Il voto previsto è un numero qualsiasi (91,3; 51,6…): **non serve** che esista tra i voti NewsGuard",
        "Lo Spearman non confronta i voti: confronta le **posizioni**",
    ], V / "sp_vero_voti.png"),
    con_grafico("Valutazione", "Passo 1: due file, separate", [
        "Fila per voto **NewsGuard**: 1 = la testata più affidabile",
        "**A parte**, fila per voto **previsto**",
        "91,3 conta solo perché è il voto previsto **più alto** delle 8",
    ], V / "sp_vero_posizioni.png"),
    con_grafico("Valutazione", "Passo 2: dalle posizioni al numero", [
        "Per ogni testata: **differenza** tra le due posizioni",
        "Tutte 0 → Spearman **1**; qui → **0,76**; su tutte le 226 testate **0,73**",
        "Pari merito (39 testate hanno 100 in NewsGuard): prendono la **posizione media**",
    ], V / "sp_vero_conto.png"),
]
i_toy = trova("Valutazione", "Spearman: due classifiche")
i_vero = trova("Valutazione", "Spearman: un esempio vero")
pagine = pagine[:i_toy] + spearman_vero + [p for k, p in enumerate(pagine) if k > i_toy and k != i_vero]

# 2. NewsGuard, MBFC e il modello
mbfc = [
    con_grafico("NewsGuard e MBFC", "Un voto contro una classe: come si confrontano", [
        "Nessuna conversione: anche qui contano solo le **posizioni**",
        "Le classi MBFC sono in ordine: Low < Mixed < Mostly Factual < High",
        "Stessa classe = **pari merito**: le due «High» sono entrambe 1,5",
        "Su tutte le 170 testate in comune: **0,71**",
    ], V / "mbfc_posizioni.png"),
    con_grafico("Il modello come annotatore", "Tre confronti sulle stesse 170 testate", [
        "Il voto del modello è quello del **test**: testate mai viste in addestramento",
        "Mistral con NewsGuard **0,74**: come i due esperti tra loro (0,71), anzi un po' di più",
        "Con MBFC **0,64**: il modello ha imparato NewsGuard, non MBFC",
    ], V / "mbfc_triangolo.png"),
]
i_curve = trova("Il modello come annotatore", "Linea tratteggiata")
pagine[i_curve] = pagine[i_curve].replace("Con MBFC un po&apos; meno (0,65)", "Con MBFC un po&apos; meno (0,64)")
assert "(0,64)" in pagine[i_curve]
pagine = pagine[:i_curve] + mbfc + pagine[i_curve:]

# 3. classificatore pieno, regressore tratteggiato
i = trova("Classificatore o regressore?", "")
vecchio = "Pieno: regressore stirato · tratteggiato: classificatore"
assert vecchio in pagine[i]
pagine[i] = pagine[i].replace(vecchio, "Pieno: classificatore · tratteggiato: regressore stirato")
pagine[i] = S.cambia_immagine(pagine[i], V / "curve_cls_reg.png")

S.scrivi(IN, OUT, c, manifest, pagine)
print(len(pagine), "slide ->", OUT)
