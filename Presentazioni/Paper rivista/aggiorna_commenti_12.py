"""Dodicesima revisione del 26 novembre di PaperRivista.odp (54 slide): l'appendice B spiegata partendo dalla Tabella 4
(colonne, parentesi, asterisco), poi il bootstrap, il p (che non è ρ) e Holm, con i numeri di Mistral contro TF-IDF
(Dataset/tentativi/20_FromScratch/paper_main_table.csv: p di ρ 0,0038, corretto 0,0646 = 0,0038 × 17).
Prima: python figure/schermate.py (tab_main.png). Uso:
    python aggiorna_commenti_12.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zn"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 54, len(p)
b = [i for i, x in enumerate(p) if "Appendice B" in S.titolo_di(x)]
assert b == [26], b


def solo_tabella(nome):
    """La schermata di una tabella senza didascalia: dal primo filetto (una riga quasi tutta scura) in giù."""
    im = Image.open(SCH / f"{nome}.png").convert("L")
    w, h = im.size
    px = im.load()
    riga = next(y for y in range(h) if sum(px[x, y] < 128 for x in range(w)) > 0.85 * w)
    uscita = SCH / f"solo_{nome}.png"
    im.crop((0, riga, w, h)).save(uscita)
    return uscita


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


appendice_b = [
    S.con_grafico("Appendice B", "Partiamo dalla Tabella 4", [
        "**MAE**: errore medio in punti. **Spearman ρ** e **Kendall τ**: quanto è giusto l'**ordine** delle testate. "
        "**Entro 20**: % di testate sbagliate di meno di 20 punti",
        "Tra **parentesi** l'intervallo al 95%; **asterisco** (*) = significativamente migliore di TF-IDF **dopo la "
        "correzione di Holm**; **grassetto** = il migliore della colonna",
    ], solo_tabella("tab_main"), 20, largo_max=26.0),
    B("Da dove vengono parentesi e asterisco: il bootstrap", [
        "Ripeschiamo le 226 testate 10.000 volte (con ripetizione) e ogni volta ricalcoliamo **tutte le colonne**, per "
        "**tutti** i modelli",
        "**Parentesi**: il 95% centrale dei 10.000 valori. Mistral, MAE 15,8 [14,3; 17,4]",
        "**Confronto con TF-IDF**: sugli stessi ripescaggi, la differenza. MAE di Mistral − MAE di TF-IDF = −2,0 "
        "[−2,9; −1,2]",
        "Da ogni confronto esce un **p**: non è scritto nella tabella, serve a decidere l'asterisco",
    ]),
    B("Il p non è ρ", [
        "**ρ** (rho) = correlazione di Spearman: una **colonna** della tabella, misura l'ordine delle testate",
        "**p** = p-value: dice se una **differenza** con TF-IDF può essere un **caso**",
        "p = quota dei 10.000 ripescaggi in cui vince TF-IDF, × 2 (contiamo anche la differenza opposta)",
        ("Esempi, Mistral contro TF-IDF:", [
            "MAE −2,0: TF-IDF non vince in **nessun** ripescaggio → p < 0,001",
            "ρ 0,735 contro 0,674, cioè +0,06: TF-IDF vince nello 0,2% dei ripescaggi → p = **0,004**",
        ]),
        "Con p < 0,05 una differenza è «significativa»",
    ]),
    B("Holm: perché su ρ l'asterisco non c'è", [
        "Nella tabella i confronti con TF-IDF sono **24** (6 modelli × 4 colonne): con tanti p, qualcuno sotto 0,05 può "
        "uscire **per caso**",
        "**Holm** corregge i p: dal più piccolo al più grande, ogni p è moltiplicato per il numero di confronti ancora in "
        "gioco",
        "Mistral, MAE: p < 0,001 → corretto < 0,001 → **asterisco**",
        "Mistral, ρ: p 0,0038 × 17 confronti ancora in gioco = **0,065** → sopra 0,05 → **niente asterisco**: è in "
        "grassetto perché è il migliore, ma non è significativamente meglio di TF-IDF",
        "Risultato: l'**errore** più basso dei modelli grandi è significativo; l'**ordine** con tutti gli articoli no "
        "(con 1 o 5 articoli sì), e lo scriviamo",
    ]),
]
nuove = p[:26] + appendice_b + p[27:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 54)")
