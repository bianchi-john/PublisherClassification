"""Tredicesima revisione del 26 novembre di PaperRivista.odp (57 slide): l'appendice B in tre parti dette chiaramente.
1. Nel paper: la Tabella 4 (colonne, parentesi, asterisco), il bootstrap, il p (che non è ρ).
2. La critica possibile: con decine di p, qualche «significativo» può essere un caso.
3. La risposta, cioè che cosa c'è nell'appendice B: la correzione di Holm, i tre gruppi di test, l'esempio di ρ e
   l'esito (errore significativo; ordine con tutti gli articoli no, con 1 o 5 sì).
Prima: aggiorna_commenti_12.py ha già prodotto figure/schermate/solo_tab_main.png. Uso:
    python aggiorna_commenti_13.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zo"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 57, len(p)
b = [i for i, x in enumerate(p) if "Appendice B" in S.titolo_di(x)]
assert b == [26, 27, 28, 29], b


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


appendice_b = [
    S.con_grafico("Appendice B", "1. Nel paper: la Tabella 4", [
        "**MAE**: errore medio in punti. **Spearman ρ** e **Kendall τ**: quanto è giusto l'**ordine** delle testate. "
        "**Entro 20**: % di testate sbagliate di meno di 20 punti",
        "Tra **parentesi** l'intervallo al 95%; **asterisco** (*) = significativamente migliore di TF-IDF; "
        "**grassetto** = il migliore della colonna",
    ], SCH / "solo_tab_main.png", 20, largo_max=26.0),
    B("1. Nel paper: il bootstrap", [
        "Ripeschiamo le 226 testate 10.000 volte (con ripetizione) e ogni volta ricalcoliamo **tutte le colonne**, per "
        "**tutti** i modelli",
        "**Parentesi**: il 95% centrale dei 10.000 valori. Mistral, MAE 15,8 [14,3; 17,4]",
        "**Confronto con TF-IDF**: sugli stessi ripescaggi, la differenza. MAE di Mistral − MAE di TF-IDF = −2,0 "
        "[−2,9; −1,2]",
        "Da ogni confronto esce un **p**: non è scritto nella tabella, serve a decidere l'asterisco",
    ]),
    B("1. Nel paper: il p (che non è ρ)", [
        "**ρ** (rho) = correlazione di Spearman: una **colonna** della tabella, misura l'ordine delle testate",
        "**p** = p-value: dice se una **differenza** con TF-IDF può essere un **caso**",
        "p = quota dei 10.000 ripescaggi in cui vince TF-IDF, × 2 (contiamo anche la differenza opposta)",
        ("Esempi, Mistral contro TF-IDF:", [
            "MAE −2,0: TF-IDF non vince in **nessun** ripescaggio → p < 0,001",
            "ρ 0,735 contro 0,674, cioè +0,06: TF-IDF vince nello 0,2% dei ripescaggi → p = **0,004**",
        ]),
        "Con p < 0,05 una differenza è «significativa»",
    ]),
    B("2. La critica possibile", [
        "Confrontiamo con TF-IDF 6 modelli su 4 colonne (24 p), più le curve per numero di articoli e altri confronti "
        "nel testo: **decine** di p",
        "Ogni p sotto 0,05 ha 1 probabilità su 20 di essere un **caso**: con decine di p, qualcuno lo è quasi "
        "sicuramente",
        "La critica: «i vostri asterischi e i vostri "
        "“significativamente” potrebbero essere dei casi»",
    ]),
    B("3. La risposta, in appendice B", [
        "Tutti i p sono corretti con **Holm**: dal più piccolo al più grande, ogni p è moltiplicato per il numero di "
        "confronti ancora in gioco",
        "In **tre gruppi**: la Tabella 4 (24 p), le curve (16 p per modello), gli altri confronti del testo (24 p). Gli "
        "intervalli della Tabella 7 non sono corretti, e lo diciamo",
        "L'asterisco e la parola «significativo» nel paper valgono **dopo** la correzione",
        "Esempio: ρ di Mistral, p 0,0038 × 17 confronti ancora in gioco = **0,065** → sopra 0,05 → niente asterisco",
        "Esito: la critica ha ragione sull'**ordine** con tutti gli articoli (non significativo, e lo scriviamo); non "
        "sull'**errore**, che resta significativo, né sull'ordine con 1 o 5 articoli",
    ]),
]
nuove = p[:26] + appendice_b + p[30:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 57)")
