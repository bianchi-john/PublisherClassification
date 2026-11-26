"""Quattordicesima revisione del 26 novembre di PaperRivista.odp (58 slide): le tre slide centrali dell'appendice B come un
filo unico. Il p di ρ di Mistral sembra significativo (0,004) → ma è uno di 24 p (la critica) → la correzione di Holm
glielo toglie (0,0038 è l'8° p più piccolo: × 17 = 0,065), per questo nella Tabella 4 non ha l'asterisco.
Uso:
    python aggiorna_commenti_14.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zq"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 58, len(p)
for i, atteso in [(28, "1. Nel paper: il p"), (29, "2. La critica possibile"), (30, "3. La risposta, in appendice B")]:
    assert "Appendice B" in S.titolo_di(p[i]) and atteso in S.testo(p[i]), (i, S.testo(p[i])[:80])


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


p[28] = B("1. Nel paper: il p (che non è ρ)", [
    "**ρ** (rho) = Spearman, una **colonna** della tabella; **p** = p-value, dice se una **differenza** con TF-IDF può "
    "essere un caso",
    "p = quota dei ripescaggi in cui vince TF-IDF × 2; se p < 0,05 la differenza è «significativa»",
    ("Mistral contro TF-IDF:", [
        "MAE −2,0: TF-IDF non vince mai → p < 0,001 → significativo",
        "ρ +0,06: TF-IDF vince nello 0,2% dei ripescaggi → p = 0,004 → **anche questo sembra significativo**",
    ]),
    "Ma nella Tabella 4 il ρ di Mistral **non ha l'asterisco**. Perché? Lo spiegano le prossime due slide",
])
p[29] = B("2. La critica possibile: tanti p", [
    "Il p di ρ non è l'unico: nella Tabella 4 confrontiamo con TF-IDF **ogni** modello (6) su **ogni** colonna (4): "
    "**24 p**",
    "Ogni p ha 1 probabilità su 20 di finire sotto 0,05 **per caso**, anche quando la differenza non c'è",
    "Come con i dadi: con un dado fare 6 è difficile; tirandone 24, almeno un 6 esce quasi sempre",
    "Con 24 p, anche se nessun modello fosse davvero diverso da TF-IDF, ci sarebbe il **71%** di probabilità di vedere "
    "almeno un «significativo»",
    "La critica: «un p di 0,004 trovato fra 24 vale meno di un p di 0,004 da un confronto solo»",
])
p[30] = B("3. La risposta, in appendice B: la correzione", [
    "Alziamo l'asticella con **Holm**: si mettono i 24 p in ordine dal più piccolo, e ognuno si moltiplica per quanti "
    "confronti restano in gioco (24, 23, 22, …)",
    "Mistral, MAE: p < 0,001, anche × 24 resta sotto 0,05 → **significativo** → asterisco",
    "Mistral, ρ: 0,0038 è l'8° p più piccolo, restano 17 confronti → 0,0038 × 17 = **0,065** → sopra 0,05 → **non** "
    "significativo → niente asterisco",
    "Lo stesso per le curve per numero di articoli e per i confronti citati nel testo",
    "Esito: l'**errore** più basso dei modelli grandi è significativo; l'**ordine** con tutti gli articoli no, e lo "
    "scriviamo; con 1 o 5 articoli sì",
])

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
p = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
               'draw:text-style-name="P2"') for x in p]
S.scrivi(IN, OUT, c, manifest, p)
print(f"{OUT.name}: {len(p)} slide")
