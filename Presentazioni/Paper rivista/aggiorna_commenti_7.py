"""Settima revisione del 26 novembre di PaperRivista.odp (58 slide), sul file di John:
- appendice A: altri esempi per l'RMSE; l'R² rifatto in tre slide, con l'esempio di John (una testata con voto 40 e
  tre articoli) e il passaggio da articoli a testate;
- appendice B: Holm spiegato passo per passo (che cos'è un p-value, perché tanti confronti sono un problema,
  Bonferroni, Holm, due esempi, il p corretto, le nostre famiglie), poi il risultato.
Uso:
    python aggiorna_commenti_7.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zi"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 58, len(p)


def controlla(i, titolo, sottotitolo):
    assert titolo in S.titolo_di(p[i]) and sottotitolo in S.testo(p[i]), (i, S.titolo_di(p[i]))


def A(sotto, voci):
    return S.pagina("Appendice A", sotto, voci, 20, alto_testo=11.6)


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


controlla(24, "Appendice A", "Un errore di 40 pesa come")
controlla(25, "Appendice A", "Quanta parte delle differenze")
controlla(28, "Appendice B", "Confronti multipli: la critica")
controlla(29, "Appendice B", "Confronti multipli: come la affrontiamo")
controlla(30, "Appendice B", "Confronti multipli: il risultato")

rmse = A("Metriche in più: RMSE, altri esempi", [
    "Modello C: sbaglia di 5, 5, 5, 25 → MAE **10**, RMSE **13,2**: un solo errore grande basta ad alzarlo",
    ("Un caso come i nostri: voti veri 10, 50, 75, 100; previsti 45, 55, 70, 85 (voti schiacciati al centro)", [
        "errori 35, 5, 5, 15 → MAE **15**, RMSE **19,4**",
        "la testata con voto 10, sovrastimata di 35, alza l'RMSE molto più del MAE",
    ]),
    "Nei dati Mistral fa proprio così: MAE **15,8**, RMSE **19,6**",
    "Come il MAE, si calcola sui voti delle **testate** (la media dei loro articoli)",
])

r2 = [
    A("Metriche in più: R², che cosa misura", [
        "Le testate hanno voti **diversi**: l'R² dice quanta parte di queste differenze il modello **ritrova**",
        "Il riferimento è chi risponde **sempre la media** dei voti",
        "R² = 1 − (errori al quadrato del modello) / (errori al quadrato di chi dà sempre la media)",
        "**1** = perfetto; **0** = come dare sempre la media; **sotto 0** = peggio della media",
        "Su **una testata sola** non si calcola: i suoi articoli hanno tutti lo stesso voto, non ci sono differenze da "
        "spiegare",
    ]),
    A("Metriche in più: R², da articoli a testate", [
        "Testata A, voto **40**: tre articoli, previsti 30, 50, 40. Testata B, voto **80**: previsti 70, 60, 80",
        ("**Per articolo**: 6 articoli, ognuno col voto della sua testata (media 60)", [
            "chi dà sempre 60 sbaglia di 20 su tutti → errori al quadrato 6 × 400 = **2.400**",
            "il modello sbaglia di 10, 10, 0, 10, 20, 0 → **700** → R² = 1 − 700/2.400 = **0,71**",
        ]),
        ("**Per testata**: prima la **media** degli articoli → A **40**, B **70**", [
            "chi dà sempre 60: 400 + 400 = **800**; il modello: 0 + 100 = **100** → R² = 1 − 100/800 = **0,88**",
        ]),
        "Nella media gli errori dei singoli articoli in parte si **compensano**: per testata l'R² sale",
    ]),
    A("Metriche in più: R², tre modelli a confronto", [
        "Tre testate con voti 20, 50, 80; chi dà sempre 50 → errori al quadrato 900 + 0 + 900 = **1.800**",
        "Modello buono, prevede 30, 50, 70 → 200 → R² **0,89**",
        "Modello che **comprime**, prevede 45, 50, 55 → 1.250 → R² **0,31**, anche se l'ordine è perfetto",
        "Modello rovesciato, prevede 80, 50, 20 → 7.200 → R² **−3**, molto peggio della media",
        "Nei dati: Mistral **0,48**, TF-IDF **0,34**: spiegano circa metà e un terzo delle differenze tra le testate; "
        "comprimono i voti, per questo l'R² non è alto. La mediana costante: **−0,06**",
    ]),
]

holm = [
    B("Confronti multipli: che cos'è un p-value", [
        "Sulle stesse 226 testate Mistral sbaglia **2,0** punti meno di TF-IDF",
        "Domanda: se i due modelli fossero **uguali**, quanto spesso vedremmo per caso una differenza così?",
        "**Bootstrap**: ripeschiamo le 226 testate 10.000 volte (con ripetizione) e ricalcoliamo la differenza",
        "p = 2 × la quota di ripescaggi in cui la differenza va **dall'altra parte**",
        "Diciamo «significativo» se p < **0,05**: quando la differenza non c'è, sbagliamo **1 volta su 20**",
    ]),
    B("Confronti multipli: perché sono un problema", [
        "Con **un** confronto, sbagliare 1 volta su 20 va bene",
        "Nella Tabella 4 i confronti sono **24**: 6 modelli contro TF-IDF × 4 metriche",
        "Anche se nessun modello fosse diverso da TF-IDF: in media **1,2** «significativi» per caso, e circa **71%** di "
        "probabilità di averne **almeno uno**",
        "Come comprare 24 biglietti di una lotteria in cui vince 1 biglietto su 20: è più facile vincere che no",
        "Serve una soglia più severa quando i confronti sono tanti",
    ]),
    B("Confronti multipli: la prima idea, Bonferroni", [
        "Si divide la soglia per il numero di confronti: con 24, significativo solo se p < 0,05/24 ≈ **0,002**",
        "Così la probabilità di **almeno un** falso «significativo» resta sotto il **5%**",
        "Difetto: è **troppo severa**, tratta ogni confronto come il più sospetto e perde differenze vere",
    ]),
    B("Confronti multipli: Holm, passo per passo", [
        "**1.** Si ordinano i p-value dal più piccolo al più grande",
        "**2.** Il primo deve stare sotto 0,05/m (m = numero di confronti), il secondo sotto 0,05/(m−1), il terzo sotto "
        "0,05/(m−2), …, l'ultimo sotto 0,05",
        "**3.** Ci si ferma al **primo** che non passa: lui e tutti quelli dopo **non** sono significativi",
        "Il primo gradino è Bonferroni; poi la soglia si **allenta**, perché i confronti già superati non sono più in "
        "gioco",
        "Stessa garanzia di Bonferroni (al massimo 5% di avere anche un solo falso «significativo»), ma trova più "
        "differenze vere",
    ]),
    B("Confronti multipli: Holm, esempio 1", [
        ("Quattro confronti, p = 0,001 · 0,012 · 0,020 · 0,040", [
            "0,001 ≤ 0,05/4 = 0,0125 ✓",
            "0,012 ≤ 0,05/3 = 0,0167 ✓",
            "0,020 ≤ 0,05/2 = 0,025 ✓",
            "0,040 ≤ 0,05/1 = 0,05 ✓",
        ]),
        "Holm: **4** significativi",
        "Bonferroni (tutti sotto 0,0125): solo **2**",
    ]),
    B("Confronti multipli: Holm, esempio 2", [
        ("Quattro confronti, p = 0,004 · 0,030 · 0,035 · 0,045", [
            "0,004 ≤ 0,05/4 = 0,0125 ✓",
            "0,030 > 0,05/3 = 0,0167 ✗ → **ci si ferma**",
            "0,035 e 0,045: **non** significativi, anche se sono sotto 0,05",
        ]),
        "Senza correzione: 4 «significativi»; con Holm: **1**",
    ]),
    B("Confronti multipli: il p corretto, quello del paper", [
        "Invece di cambiare la soglia si **corregge il p**, e lo si confronta sempre con 0,05",
        "p corretto = p × numero di confronti ancora in gioco (m, poi m−1, …); non può scendere rispetto al precedente",
        ("Esempio 2:", [
            "0,004 × 4 = **0,016** ✓;  0,030 × 3 = **0,090** ✗",
            "0,035 × 2 = 0,070, ma non può scendere → **0,090** ✗;  0,045 × 1 → **0,090** ✗",
        ]),
        "Nel paper «Holm p = 0,07» vuol dire: p corretto 0,07 > 0,05 → **non** significativo",
    ]),
    B("Confronti multipli: le nostre tre famiglie", [
        "Correggiamo **dentro** gruppi di domande diverse (famiglie), non tutto insieme",
        "**1.** Ogni modello contro TF-IDF sulle 4 metriche della Tabella 4: 6 × 4 = **24** test",
        "**2.** Le curve per numero di articoli: 8 punti × 2 metriche = **16** test per modello",
        "**3.** Gli altri confronti citati nel testo: **24** test",
        "Gli intervalli della Tabella 7 (i protocolli) **non** sono corretti, e lo scriviamo",
    ]),
    B("Confronti multipli: il risultato", [
        "Mistral contro TF-IDF, errore: **−2,0** punti [−2,9; −1,2] → **significativo** dopo Holm",
        "Mistral contro TF-IDF, ρ: +0,060 [0,020; 0,104]. L'intervallo esclude lo zero, ma il p corretto è **0,07** → "
        "**non** significativo",
        "Quota entro 20 punti: +6,6, p corretto 0,18 → non significativo",
        "Qwen3-Embedding-8B contro TF-IDF: errore −1,5 ed entro 20 punti +8,4 significativi; ρ +0,037 no",
        "Nel testo: il vantaggio **solido** è sull'**errore**; sull'ordine delle testate è probabile ma non dimostrato",
    ]),
]

nuove = p[:25] + [rmse] + r2 + p[26:29] + holm + p[31:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 58)")
