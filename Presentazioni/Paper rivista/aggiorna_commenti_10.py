"""Decima revisione del 26 novembre di PaperRivista.odp (64 slide), sul file di John: l'appendice B in quattro parti.
1. Che cosa c'è già nel paper principale (Sezione 4 e Tabella 4): «significativo», il bootstrap (uno solo), il
   p-value, l'asterisco della Tabella 4, che è già dopo la correzione di Holm.
2. La critica dei tanti confronti.
3. Come rispondiamo: Bonferroni, Holm e il resto (le slide di prima, invariate).
4. Chi ha ragione, con i p prima e dopo la correzione (Dataset/tentativi/20_FromScratch/paper_main_table.csv e
   paper_curves.csv): sull'errore noi, sull'ordine con tutti gli articoli la critica, con pochi articoli noi.
Uso:
    python aggiorna_commenti_10.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zl"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 64, len(p)
for i, atteso in [(26, "Che cosa vuol dire «significativo» nel paper"), (27, "Il bootstrap: uno solo"),
                  (28, "Il p-value"), (29, "Confronti multipli: la critica"), (30, "Bonferroni"),
                  (35, "le nostre tre famiglie"), (36, "Confronti multipli: il risultato")]:
    assert "Appendice B" in S.titolo_di(p[i]) and atteso in S.testo(p[i]), (i, S.testo(p[i])[:80])


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


gia_nel_paper = [
    B("1. Già nel paper: che cosa vuol dire «significativo»", [
        "Sezione 4, «Statistical testing»: intervalli bootstrap al 95%, confronti sugli stessi ripescaggi, p corretti "
        "con Holm; «significativo» = lo resta **dopo** la correzione, al 5%",
        "La domanda: una differenza tra due modelli è **vera**, o dipende da **quali** 226 testate ci sono capitate?",
        "Con altre testate politiche, due modelli **ugualmente bravi** vincerebbero ora l'uno ora l'altro: una "
        "differenza piccola può venire solo da questo",
    ]),
    B("1. Già nel paper: il bootstrap, uno solo", [
        "Ripeschiamo le 226 testate 10.000 volte (con ripetizione) e su ogni ripescaggio calcoliamo le metriche di "
        "**tutti** i modelli: un calcolo solo, insieme alla Tabella 4",
        ("Da quei 10.000 valori escono due cose:", [
            "gli **intervalli** al 95% tra parentesi nelle tabelle, es. MAE di Mistral 15,8 [14,3; 17,4]",
            "i **confronti** con TF-IDF: sugli stessi ripescaggi, la differenza di MAE di Mistral − TF-IDF, −2,0 "
            "[−2,9; −1,2], con il suo p",
        ]),
        "Se Mistral è meglio in **quasi tutti** i ripescaggi, la differenza non dipende dalle testate capitate",
    ]),
    B("1. Già nel paper: il p-value", [
        "Sugli **stessi** 10.000 ripescaggi contiamo quelli in cui la differenza va **dall'altra parte**, cioè TF-IDF "
        "meglio di Mistral",
        "Esempio: TF-IDF vince in 150 ripescaggi su 10.000 → **1,5%**",
        "Lo moltiplichiamo per **2** perché avremmo segnalato anche la differenza opposta: contiamo i due lati → "
        "p = **0,03**",
        "**Significativo** se p < **0,05**: se i due modelli fossero ugualmente bravi, un risultato così uscirebbe meno "
        "di 1 volta su 20",
    ]),
    B("1. Già nel paper: l'asterisco della Tabella 4", [
        "L'asterisco (*) segna un modello significativamente migliore di TF-IDF **dopo la correzione di Holm** (lo dice "
        "la didascalia della Tabella 4)",
        "La correzione è quindi **già** nei risultati principali; l'appendice B dice solo **come** è fatta",
        "Asterischi: Mistral, MAE 15,8*; Qwen3-Embedding-8B, MAE 16,3* ed entro 20 punti 69,9*",
        "**Nessun** asterisco su Spearman e Kendall",
    ]),
]

critica = B("2. La critica: tanti confronti", [
    "Nella Tabella 4 confrontiamo con TF-IDF **6 modelli** su **4 metriche** (MAE, Spearman, Kendall, entro 20 punti): "
    "**24** confronti, ognuno col suo p",
    "Ognuno ha **1 probabilità su 20** di sembrare significativo per caso, anche se i modelli fossero ugualmente bravi",
    "Se nessun modello fosse davvero diverso da TF-IDF, su 24 confronti ci aspetteremmo 24 × 0,05 ≈ **1** falso "
    "«significativo» (in media, ripetendo lo studio tante volte), e circa **71%** di probabilità di averne almeno uno",
    "La critica: «qualcuno dei vostri risultati significativi potrebbe essere un caso»",
])


def rinumera(pag, vecchio, nuovo):
    assert pag.count(vecchio) == 1, vecchio
    return pag.replace(vecchio, nuovo)


risposta = [rinumera(p[30], "Confronti multipli: la prima idea, Bonferroni", "3. La risposta: la prima idea, Bonferroni")]
for i in range(31, 36):
    risposta.append(rinumera(p[i], "Confronti multipli:", "3. La risposta:"))
# il p corretto di Mistral su Spearman è 0,0646 (paper_main_table.csv): 0,065, non 0,07 (corretto anche nel paper)
risposta[4] = S.riscrivi(risposta[4], "Holm p = 0,07", "Nel paper «Holm p = 0,065» vuol dire: p corretto 0,065 > 0,05 → "
                         "**non** significativo", 20)

verdetto = [
    B("4. Chi ha ragione? I numeri", [
        "Mistral, errore (MAE) **−2,0**: p < 0,001, corretto < 0,001 → **resta significativo**",
        "Qwen3-Embedding-8B, errore **−1,5** ed entro 20 punti **+8,4**: corretti 0,004 → **restano significativi**",
        "Mistral, Spearman e Kendall **+0,06**: p 0,004, corretti **0,065** → **non** più significativi",
        "Mistral, entro 20 punti **+6,6**: p 0,013, corretto 0,18 → non significativo",
        "Con **1 o 5 articoli** per testata Mistral è significativamente meglio di TF-IDF anche nell'ordine (ρ +0,22 e "
        "+0,12) e nell'errore (−3,2 e −2,2), dopo Holm",
    ]),
    B("4. Chi ha ragione? Il verdetto", [
        "Sull'**errore** abbiamo ragione noi: il vantaggio dei modelli grandi resta dopo la correzione",
        "Sull'**ordine** delle testate, con tutti gli articoli, ha ragione la critica: +0,06 potrebbe essere un caso, e "
        "nel paper lo scriviamo",
        "Con **pochi articoli** abbiamo ragione noi anche sull'ordine",
        "Quindi i risultati significativi ci sono: il principale (errore più basso) lo è; quello sull'ordine no, e non lo "
        "presentiamo come tale",
        "Senza correzione anche l'ordine sarebbe «significativo» (p 0,004): la correzione ci toglie un risultato, segno "
        "che la applichiamo davvero",
    ]),
]

nuove = p[:26] + gia_nel_paper + [critica] + risposta + verdetto + p[37:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 64)")
