"""Ottava revisione del 26 novembre di PaperRivista.odp (65 slide), sul file di John:
- appendice A: R² e AUC come la slide RMSE di John, una slide ciascuna (che cos'è in breve, poi esempi al volo);
- appendice B: prima che cosa vuol dire «significativo» nel paper, il bootstrap (lo stesso degli intervalli), il
  p-value e il perché del ×2, e solo dopo la critica dei tanti confronti; Bonferroni, Holm e il resto restano.
Uso:
    python aggiorna_commenti_8.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zj"
S.animazione = lambda ids: ""                       # nessuna animazione
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
p = [S.senza_commenti(x) for x in re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)]
assert len(p) == 65, len(p)


def controlla(i, titolo, contiene):
    assert titolo in S.titolo_di(p[i]) and contiene in S.testo(p[i]), (i, S.titolo_di(p[i]))


def A(sotto, voci):
    return S.pagina("Appendice A", sotto, voci, 20, alto_testo=11.6)


def B(sotto, voci):
    return S.pagina("Appendice B", sotto, voci, 20, alto_testo=11.6)


controlla(22, "Appendice A", "il modello A sbaglia di 10, 10, 10, 10")       # la slide RMSE di John: resta così
controlla(23, "Appendice A", "R², che cosa misura")
controlla(25, "Appendice A", "R², tre modelli a confronto")
controlla(26, "Appendice A", "Metriche in più: AUC")
controlla(28, "Appendice B", "Confronti multipli: la critica")
controlla(29, "Appendice B", "che cos'è un p-value")
controlla(30, "Appendice B", "perché sono un problema")
controlla(31, "Appendice B", "Bonferroni")

r2 = A("R²", [
    "Quanta parte delle **differenze tra i voti** delle testate il modello ritrova, rispetto a chi risponde **sempre "
    "la media** (1 = perfetto, 0 = come la media, sotto 0 = peggio)",
    ("Esempio, 3 testate con voto 20, 50, 80 (chi dà sempre 50 ha R² 0):", [
        "il modello A prevede 30, 50, 70 → R² **0,89**",
        "il modello B prevede 45, 50, 55 → R² **0,31**: ordine giusto, ma voti schiacciati al centro",
        "il modello C prevede 80, 50, 20 → R² **−3**: peggio che dare sempre la media",
    ]),
])
auc = A("AUC", [
    "Affidabile = voto ≥ 60. Su tutte le coppie (una testata affidabile, una no), quante volte il modello dà il voto "
    "più alto a quella affidabile (1 = sempre, 0,5 = a caso)",
    ("Esempio, 4 testate: due affidabili (voto 90 e 70), due no (voto 50 e 20):", [
        "il modello A prevede 80, 65, 55, 40 → 4 coppie giuste su 4 → AUC **1**",
        "il modello B prevede 75, 55, 60, 45 → la testata da 70 finisce sotto quella da 50 → 3 su 4 → AUC **0,75**",
        "il modello C dà 65 a tutte → non distingue nessuna coppia → AUC **0,5**, come tirare a caso",
    ]),
])

holm_inizio = [
    B("Che cosa vuol dire «significativo» nel paper", [
        "Nella Tabella 4 l'asterisco (*) segna i modelli **significativamente** migliori di TF-IDF; nel testo: «Mistral "
        "abbassa l'errore di TF-IDF di 2,0 punti, in modo significativo»",
        "La domanda: la differenza è **vera**, o dipende da **quali** 226 testate ci sono capitate?",
        "Con altre testate politiche, due modelli **ugualmente bravi** vincerebbero ora l'uno ora l'altro: una "
        "differenza piccola può venire solo da questo",
    ]),
    B("Il bootstrap, lo stesso degli intervalli", [
        "È lo stesso bootstrap degli intervalli al 95% di tutte le tabelle: ripeschiamo le 226 testate **10.000** volte, "
        "con ripetizione",
        "Per confrontare due modelli usiamo gli **stessi** ripescaggi per tutti e due, e ogni volta calcoliamo la "
        "differenza: MAE di Mistral − MAE di TF-IDF",
        "Se Mistral è meglio in **quasi tutti** i ripescaggi, la differenza non dipende dalle testate capitate",
    ]),
    B("Il p-value", [
        "Contiamo i ripescaggi in cui la differenza va **dall'altra parte**, cioè TF-IDF meglio di Mistral",
        "Esempio: su 10.000 ripescaggi, TF-IDF vince in 150 → **1,5%**",
        "Lo moltiplichiamo per **2** perché avremmo segnalato anche la differenza opposta (TF-IDF meglio di Mistral): "
        "contiamo i due lati → p = **0,03**",
        "**Significativo** se p < **0,05**: se i due modelli fossero ugualmente bravi, un risultato così uscirebbe meno "
        "di 1 volta su 20",
    ]),
    B("Confronti multipli: la critica", [
        "Nella Tabella 4 confrontiamo con TF-IDF **6 modelli** su **4 metriche** (MAE, Spearman, Kendall, entro 20 "
        "punti): **24** confronti, ognuno col suo p",
        "Ognuno ha **1 probabilità su 20** di sembrare significativo per caso, anche se i modelli fossero ugualmente "
        "bravi",
        "Se nessun modello fosse davvero diverso da TF-IDF, su 24 confronti ci aspetteremmo 24 × 0,05 ≈ **1** falso "
        "«significativo» (in media, ripetendo lo studio tante volte), e circa **71%** di probabilità di averne almeno uno",
        "La critica: «qualcuno dei vostri asterischi potrebbe essere un caso»",
    ]),
]

nuove = p[:23] + [r2, auc] + p[27:28] + holm_inizio + p[31:]

# nel file di John, salvato da LibreOffice, gli stili automatici sono rinumerati: P5 ha uno sfondo, P2 no
assert '<style:style style:name="P2" style:family="paragraph"><loext:graphic-properties draw:fill="none"/>' in c
nuove = [x.replace('presentation:style-name="pr4" draw:text-style-name="P5"', 'presentation:style-name="pr4" '
                   'draw:text-style-name="P2"') for x in nuove]
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 65)")
