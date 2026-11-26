"""Revisione del 6 novembre di FinalState.odp: solo slide NUOVE, inserite dopo slide di John (che restano com'erano):
- dopo «Spearman guarda solo l'ordine»: Kendall, le coppie (kendall_coppie.png) e il numero (kendall_mistral.png);
- dopo «Come si stira»: lo stiramento passo per passo (stirare_passi.png) e training/test (stirare_distribuzioni.png);
- dopo «Normale e stirato»: solo gli stirati (curve_stirato.png);
- dopo «Media Bias/Fact Check»: come abbiniamo le testate (mbfc_abbinamento.png);
- dopo «Un voto contro una classe»: dove i due esperti concordano e dove no (mbfc_fasce.png).
Prima: figure/grafici_spiegazioni.py e figure/grafici_curve.py. Uso:
python aggiorna_spiegazioni.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_grafico

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zz"
S.animazione = lambda ids: ""          # John ha tolto tutte le animazioni: anche le slide nuove senza
z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
V = FIG / "valutazione"


def trova(titolo, contiene):
    (i,) = [i for i, p in enumerate(pagine) if titolo in S.titolo_di(p) and contiene in S.testo(p)]
    return i


dopo = {
    trova("Valutazione", "Spearman guarda solo l'ordine"): [
        con_grafico("Valutazione", "Kendall: contare le coppie", [
            "Per ogni **coppia** di testate: il modello le mette nello stesso ordine di NewsGuard?",
            "8 testate → 28 coppie: **22** nello stesso ordine (✓), **6** invertite (✗)",
        ], V / "kendall_coppie.png"),
        con_grafico("Valutazione", "Kendall: il numero", [
            "Kendall = (coppie giuste − coppie invertite) / tutte le coppie = (22 − 6) / 28 = **0,57**",
            "Sullo stesso esempio lo Spearman è 0,76: il Kendall è sempre più basso, è **un'altra scala**",
            "Si legge così: (1 + Kendall) / 2 ≈ coppie nell'ordine giusto. Mistral 0,54 → **3 coppie su 4**",
            "Le coppie con lo stesso voto NewsGuard non contano (Kendall «b»)",
        ], V / "kendall_mistral.png"),
    ],
    trova("Regressori", "Come si stira"): [
        con_grafico("Regressori", "Stirare passo per passo", [
            "Le 45 testate di test si mettono **in fila** per voto previsto",
            "Ognuna prende il voto vero del training **alla stessa posizione %**",
            "L'ordine non cambia: C resta sbagliata (prevista 63, vera 17,5)",
        ], V / "stirare_passi.png"),
        con_grafico("Regressori", "Il training e il test", [
            "Prima i previsti stanno tra 45 e 90; dopo, gli stirati hanno **la forma dei voti veri del training**",
            "Si usa solo il training: il voto vero delle testate di test non si guarda mai",
        ], V / "stirare_distribuzioni.png"),
    ],
    trova("Regressori", "Normale e stirato"): [
        con_grafico("Regressori", "Solo stirati", [
            "Tutti i modelli stirati, anche il TF-IDF; cerchiati: battono il TF-IDF stirato",
            "Mistral e Qwen vincono con pochi articoli; con tutti gli articoli il TF-IDF li raggiunge",
        ], V / "curve_stirato.png"),
    ],
    trova("Media Bias/Fact Check", "Un secondo esperto"): [
        con_grafico("Media Bias/Fact Check", "Come abbiniamo le testate", [
            "Stesso dominio (senza «www.»): **170** testate su 226 hanno una scheda MBFC",
            "Le altre 56 MBFC non le valuta: sono più spesso poco affidabili (l1-l2: 46% contro 31%)",
        ], V / "mbfc_abbinamento.png"),
    ],
    trova("NewsGuard e MBFC", "Un voto contro una classe"): [
        con_grafico("NewsGuard e MBFC", "Dove concordano e dove no", [
            "In generale sono d'accordo, ma «Mixed» copre tutta la scala NewsGuard",
            "Un solo disaccordo forte: una testata «High» per MBFC con voto NewsGuard tra 40 e 59",
        ], V / "mbfc_fasce.png", largo_max=17),
    ],
}
finale = []
for i, p in enumerate(pagine):
    finale += [p] + dopo.get(i, [])
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), f"slide (erano {len(pagine)}),", S.contatore[0], "punti nuovi ->", OUT)
