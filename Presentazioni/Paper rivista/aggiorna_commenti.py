"""Revisione del 26 novembre di PaperRivista.odp: risolve i commenti di John lasciati nelle slide. Lavora sul file di
John (tiene le sue modifiche): cambia le immagini delle figure rifatte nel paper (2, 3, 5, 8), riscrive o aggiunge le
slide chieste (deduplicazione, Kendall, baseline, testate non valutate, protocolli, orientamento) e rifà le appendici
come critica → come la affrontiamo → esperimento. Toglie i commenti risolti.
Prima: compilare il paper e  python figure/schermate.py  (più la figura 3 senza didascalia, vedi sotto). Uso:
    python aggiorna_commenti.py <file di John.odp> <uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import stile_slide as S  # noqa: E402

S.PREFISSO[0] = "zc"
S.animazione = lambda ids: ""                       # nessuna animazione
SCH = QUI / "figure" / "schermate"
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])

z = zipfile.ZipFile(IN)
c = S.aggiungi_stili(z.read("content.xml").decode())
manifest = z.read("META-INF/manifest.xml").decode()
vecchie = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(vecchie) == 44, len(vecchie)
p = [S.senza_commenti(x) for x in vecchie]          # tutti i commenti di John sono risolti qui sotto


def titolo(i, atteso):
    assert atteso in S.titolo_di(p[i]), (i, S.titolo_di(p[i]))
    return p[i]


def sch(nome):
    return SCH / f"{nome}.png"


def testo(tit, sotto, voci, corpo=20):
    return S.pagina(tit, sotto, voci, corpo, alto_testo=11.6)


def grafico(tit, sotto, nome, voci=(), corpo=20):
    return S.con_grafico(tit, sotto, list(voci), sch(nome), corpo)


def figura_poi_testo(tit, sotto, img, voci, corpo=20):
    """La figura in alto, il testo in fondo (al posto della didascalia)."""
    alto_testo = S.altezza_testo(voci, corpo) + 0.3
    y_testo = 15.4 - alto_testo
    w, h = Image.open(img).size
    largo = min(25.0, (y_testo - 0.2 - 3.4) * w / h)
    extra = S.immagine(img, (28 - largo) / 2, 3.4, largo=largo)
    pag = S.pagina(tit, sotto, voci, corpo, alto_testo=alto_testo, extra=extra)
    return pag.replace('svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline"',
                       f'svg:x="1.471cm" svg:y="{y_testo:.2f}cm" presentation:class="outline"')


def figura_a_sinistra(tit, sotto, img, voci, corpo=20):
    """Una figura quasi quadrata alta quanto la slide, a sinistra; il testo a destra."""
    extra = S.immagine(img, 1.2, 2.9, alto=12.7)
    pag = S.pagina(tit, sotto, voci, corpo, alto_testo=10.0, extra=extra)
    vecchio = 'svg:width="25.199cm" svg:height="10.00cm" svg:x="1.471cm" svg:y="3.4cm" presentation:class="outline"'
    assert pag.count(vecchio) == 1
    return pag.replace(vecchio, 'svg:width="13.6cm" svg:height="10.00cm" svg:x="13.6cm" svg:y="4.2cm" presentation:class="outline"')


def numera(pag, vecchio, nuovo):
    assert pag.count(vecchio) == 1, vecchio
    return pag.replace(vecchio, nuovo)


nuove = []
# ------------------------------------------------------------------ contributi + deduplicazione
nuove += [titolo(0, "Contributi del paper (1/2)"),
          testo("Deduplicazione (1/2)", "Che cosa vuol dire", [
              "Molti siti pubblicano **lo stesso articolo**, copiato o ripreso, spesso siti dello **stesso proprietario**",
              "Se una copia è in training e l'altra in test, il modello **riconosce l'articolo** invece di giudicarlo",
              "Quindi teniamo **una sola copia** di ogni articolo (la più vecchia), confrontando **tutte le testate** fra loro",
              "Toglierle solo dentro ogni testata non basta: la stessa copia resterebbe in train e test sotto due nomi",
          ]),
          testo("Deduplicazione (2/2)", "Come la facciamo, con un esempio", [
              "**Copie esatte**: lo stesso testo",
              ("**Quasi-copie**: confrontiamo le sequenze di **5 parole consecutive** dei due articoli", [
                  "Jaccard = sequenze in comune / sequenze totali dei due; se ≥ **0,5** è una copia",
                  "Es.: due articoli di 500 parole uguali tranne titolo e ultima riga → Jaccard ≈ 0,95 → copia",
                  "Due articoli diversi sulla stessa notizia → Jaccard ≈ 0,05 → non è una copia",
              ]),
              "Prima: il **19,4%** degli articoli ha una copia in una testata dello **stesso proprietario**, il **2,7%** "
              "in una di un altro proprietario; dopo: nessuna",
          ]),
          titolo(1, "Contributi del paper (2/2)"), titolo(2, "Tabella 1"), titolo(3, "Il dataset")]

# ------------------------------------------------------------------ figure 2 e 8 rifatte, anonimizzazione semplificata
mappa = S.cambia_immagine(titolo(4, "Gruppi di proprietà"), sch("fig_network"))
nuove.append(mappa)
anon = titolo(5, "Anonimizzazione")
anon = S.riscrivi(anon, "Testate e giornalisti", "Testate (**compresa quella che pubblica**) e giornalisti → **[OUTLET]**, "
                  "**[AUTHOR]**", 24)
anon = S.riscrivi(anon, "Più due regex", "Link ed e-mail: due espressioni regolari → **[URL]**, **[EMAIL]**", 24)
nuove += [anon, S.cambia_immagine(titolo(6, "Anonimizzazione"), sch("fig_anonymization"))]

# ------------------------------------------------------------------ figura 3: orientamenti e articoli per testata
titolo(7, "Il dataset finale")
nuove.append(figura_poi_testo("Il dataset finale", "Figura 3", SCH / "solo_dataset.png", [
    "(a) Orientamento: NewsGuard dà solo **Left**, **Right** o campo vuoto (**none**): **nessun accorpamento** "
    "(le cinque classi erano nell'esportazione del 2022)",
    "(b) Articoli per testata in ordine crescente: minimo **10**, mediana 112, massimo 201",
]))
nuove += [titolo(8, "La pipeline"), titolo(9, "Setup sperimentale"), titolo(10, "Risultati: articoli"),
          titolo(11, "Risultati: testate")]

# ------------------------------------------------------------------ Spearman e Kendall
nuove += [
    testo("Spearman e Kendall (1/2)", "Che cosa misurano", [
        "Non guardano **quanto** il voto è vicino, ma se l'**ordine** delle testate è giusto",
        "Servono per l'uso pratico: decidere **quali testate guardare prima**",
        "**Spearman ρ**: correlazione tra le **posizioni in classifica** (1 = stesso ordine, 0 = nessun legame)",
        "**Kendall τ**: conta le **coppie** di testate messe nell'ordine giusto",
    ]),
    testo("Spearman e Kendall (2/2)", "Un esempio con 4 testate", [
        "Voto vero: A **20**, B **50**, C **70**, D **90**; previsto: A **40**, B **65**, C **60**, D **80**",
        ("6 coppie: A-B ✓, A-C ✓, A-D ✓, **B-C ✗** (B previsto più alto di C), B-D ✓, C-D ✓", [
            "5 coppie giuste, 1 sbagliata → τ = (5 − 1) / 6 = **0,67**",
            "Spearman sullo stesso esempio: **0,80**; MAE: 13,8 punti (ordine ed errore sono due cose diverse)",
        ]),
        "**τ_b** tiene conto dei **pari merito**: le 226 testate hanno solo 45 voti diversi",
        "Nei dati: Mistral τ_b **0,54**, TF-IDF **0,48**",
    ]),
]

# ------------------------------------------------------------------ baseline con esempi
titolo(12, "Le baseline senza testo")
nuove += [
    testo("Le baseline senza testo (1/3)", "Un esempio da seguire", [
        "Testate di **training**: A 20 (destra), B 60 (destra), C 75 (sinistra), D 80 (nessuna), E 100 (sinistra)",
        "Testate di **test**: X 30 (destra), Y 90 (sinistra), W 80 (nessuna)",
        "Le baseline **non leggono il testo**: usano solo i voti del training (e, l'ultima, l'orientamento)",
    ]),
    testo("Le baseline senza testo (2/3)", "Le due costanti", [
        ("**Training mean**: a ogni testata di test la **media** del training, (20+60+75+80+100)/5 = **67**", [
            "errori: X 37, Y 23, W 13 → MAE **24,3**",
        ]),
        ("**Training median**: la **mediana** del training, **75**", [
            "errori: X 45, Y 15, W 5 → MAE **21,7**: la mediana è la costante migliore per il MAE",
        ]),
        "Nei dati veri: media → MAE **22,6**; mediana → MAE **22,1**",
    ]),
    testo("Le baseline senza testo (3/3)", "Solo l'orientamento", [
        ("**Orientation only**: la media del training **per orientamento**", [
            "destra (A, B) → **40**; sinistra (C, E) → **87,5**; nessuna (D) → **80**",
            "X (destra) → 40, errore 10; Y (sinistra) → 87,5, errore 2,5; W → 80, errore 0",
        ]),
        "Misura quanto del voto si indovina **conoscendo solo la parte politica**",
        "Nei dati veri (voto medio: destra 54, sinistra 80) → MAE **18,5**, ρ **0,36**",
        "I modelli col testo fanno meglio: Mistral MAE **15,8**, ρ **0,74**",
    ]),
]

# ------------------------------------------------------------------ figura 5 rifatta
titolo(13, "Voto previsto e voto vero")
nuove += [figura_a_sinistra("Voto previsto e voto vero", "Figura 5", SCH / "solo_scatter.png", [
    "Un punto per **testata**: la media delle previsioni sui suoi articoli",
    "In alto la distribuzione dei **voti veri** (uguale nei quattro grafici), a destra quella dei **previsti**",
    "I previsti sono più **raccolti** dei veri: i modelli **comprimono** la scala e sovrastimano i voti bassi",
]),
          titolo(14, "Contro TF-IDF")]

# ------------------------------------------------------------------ testate non valutate, passo per passo
titolo(15, "Testate non valutate (1/3)"); titolo(16, "Testate non valutate (2/3)")
nuove += [
    testo("Testate non valutate (1/5)", "Il problema", [
        "Vogliamo usare il modello su testate **mai valutate** da NewsGuard",
        "Quelle che non siamo riusciti a raccogliere sono molto meno affidabili: voto medio **41** contro **69**",
        "Nel livello più basso (0–39): **43%** tra le non raccolte, **15%** nel nostro dataset",
    ]),
    testo("Testate non valutate (2/5)", "Perché conta", [
        "Mistral sbaglia di **31** punti nel livello più basso (0–39) e di **8,5** nel livello 60–74",
        "Una popolazione con più testate del livello basso ha un errore medio **più alto**",
        "→ l'errore misurato sulle 226 testate è **ottimista** per le testate non valutate",
    ]),
    testo("Testate non valutate (3/5)", "Come lo stimiamo: il calcolo", [
        "Errore di Mistral **per livello**: 31,0 · 11,5 · 8,5 · 12,7 · 18,9",
        "Quota di ogni livello tra le testate **non raccolte**: 43,3% · 35,1% · 5,0% · 13,2% · 3,4%",
        "Media pesata: 0,433×31,0 + 0,351×11,5 + 0,050×8,5 + 0,132×12,7 + 0,034×18,9 = **20,2**",
        "Con le quote del **nostro** dataset (15% · 19% · 13% · 35% · 17%) si ritrova **15,8**",
    ]),
    testo("Testate non valutate (4/5)", "Le altre righe della Tabella 6", [
        "**Ricalibrato**: una retta stimata sul training «allarga» i voti compressi → errore più basso per tutti",
        "**Senza le 16 testate** col marchio rimasto nel testo: il vantaggio di Mistral resta (**1,8** punti)",
    ]),
    numera(titolo(17, "Testate non valutate (3/3)"), "(3/3)", "(5/5)"),
]

# ------------------------------------------------------------------ il protocollo, più chiaro
nuove += [
    numera(titolo(18, "Quanto conta il protocollo (1/4)"), "(1/4)", "(1/6)"),
    testo("Quanto conta il protocollo (2/6)", "Una protezione tolta alla volta: test separati", [
        "Partiamo dal **protocollo completo** e togliamo **una sola** protezione per volta",
        "**Article split**: fold sugli articoli → la **stessa testata** è in training e in test",
        "**Sisters**: testate di test **mai viste**, ma in training aggiungiamo le **89 sorelle** tolte dalla pulizia "
        "(testo originale: si confronta con «original text»)",
        "**Original text**: testate separate, testo **non anonimizzato**",
        "**Rules only**: testate separate, anonimizzazione con **sole regole**",
    ]),
    testo("Quanto conta il protocollo (3/6)", "Stesse testate in training e in test?", [
        "**Article split** → **sì**: è proprio la protezione che togliamo",
        "**Sisters** → **no**: le testate di test non sono mai in training, ci sono solo testate dello **stesso proprietario**",
        "**Original text**, **Rules only**, protocollo completo → **no**",
        "Ogni riga della Tabella 7 si confronta con quella che ha **una protezione in più**: le sorelle con il testo "
        "originale, le altre con il protocollo completo",
    ]),
    testo("Quanto conta il protocollo (4/6)", "«Rules only»: che cos'è e perché", [
        "Togliamo link, e-mail e il **nome della testata** (che conosciamo dal dominio) con espressioni regolari, "
        "**senza GLiNER**",
        "**Non** togliamo i nomi delle **altre** testate né dei giornalisti",
        "Perché: è l'anonimizzazione **più semplice e più usata**; ci dice **quanto aggiunge GLiNER**",
        "Risultato: l'errore cambia di **al massimo 0,4** punti",
    ]),
    grafico("Quanto conta il protocollo (5/6)", "Tabella 7", "tab_ladder", [
        "Stesse testate in training e in test **solo** nell'article split",
    ]),
    numera(titolo(21, "Quanto conta il protocollo (4/4)"), "(4/4)", "(6/6)"),
    titolo(22, "Quanti articoli servono"), titolo(23, "Quanti articoli servono"), titolo(24, "Orientamento politico"),
]
titolo(19, "Quanto conta il protocollo (2/4)"); titolo(20, "Quanto conta il protocollo (3/4)")

# ------------------------------------------------------------------ orientamento: esempio e risposta
nuove += [
    testo("Orientamento politico", "«A parità di voto»: un esempio", [
        "Due testate con lo **stesso voto NewsGuard, 75**: una di sinistra, una di destra",
        "Il modello prevede **80** per quella di sinistra e **70** per quella di destra → divario di **10** punti",
        "Ma le testate di destra hanno in media voti più bassi, e i voti bassi sono **sovrastimati** da tutti i modelli",
        "Per confrontarle **a parità di voto** togliamo dall'errore la parte dovuta al voto vero (una retta), poi "
        "confrontiamo sinistra e destra",
    ]),
    testo("Orientamento politico", "Il risultato", [
        "A parità di voto Mistral **sovrastima** le testate di sinistra (**+5,2**) e **sottostima** quelle di destra "
        "(**−4,3**): divario **9,6** (8,7–8,8 con altri due controlli)",
        "Senza tener conto del voto, invece, le testate di destra risultano **sovrastimate**: sono più spesso nei "
        "livelli bassi",
        "Il zero-shot, mai addestrato sui voti, **non ha divario** (0,1)",
    ]),
    grafico("Orientamento politico", "Figura 7", "fig_errors",
            ["(b) a parità di voto: sinistra **sovrastimata**, destra **sottostimata**"]),
]
titolo(25, "Orientamento politico")
nuove += [titolo(26, "Il tool"), titolo(27, "Il tool"), titolo(28, "Le appendici")]


# ------------------------------------------------------------------ appendici: critica → come → esperimento
def appendice(lettera, nome, critica, come, risultato, schermata=None):
    """Tre slide: la critica spiegata, come la affrontiamo passo per passo, l'esperimento con i risultati evidenziati."""
    out = [testo(f"Appendice {lettera}", f"{nome}: la critica", critica),
           testo(f"Appendice {lettera}", f"{nome}: come la affrontiamo", come)]
    out.append(grafico(f"Appendice {lettera}", f"{nome}: il risultato", schermata, risultato) if schermata
               else testo(f"Appendice {lettera}", f"{nome}: il risultato", risultato))
    return out


nuove += appendice("A", "Metriche in più", [
    "«Perché solo MAE, Spearman e Kendall? Altri lavori usano R², RMSE o AUC»",
    "Un revisore vuole vedere che il risultato **non dipende dalla metrica** scelta",
    "Chi usa il modello per decidere «affidabile sì/no» guarda l'**AUC**, non il MAE",
], [
    "**RMSE**: come il MAE ma punisce di più gli errori grandi; errori 5, 5, 30 → MAE 13,3, RMSE **17,8**",
    "**R²**: quota della varianza dei voti veri spiegata; 0 = dare sempre la media, 1 = perfetto",
    "**AUC**: presa a caso una testata affidabile (≥ 60) e una no, probabilità che il modello dia il voto più alto "
    "alla prima; 0,5 = a caso",
    "Calcolate **per testata** (media dei suoi articoli), come la Tabella 4: il compito è il voto della **testata**",
], [
    "Stesso ordine del MAE: Mistral R² **0,48**, RMSE **19,6** (TF-IDF 0,34 e 22,0)",
    "AUC: Mistral **0,88**, TF-IDF **0,86**: nel solo affidabile sì/no il vantaggio è piccolo",
    "Ha senso tenerle **in appendice**: confermano il MAE e anticipano la critica",
], "tab_more")

nuove += appendice("B", "Confronti multipli", [
    "«Avete fatto **decine di confronti**: qualcuno risulta significativo **per caso**»",
    "Con 24 confronti al 5%, in media ~1 risulta «significativo» anche se non c'è nessuna differenza",
], [
    "**Holm**: ordino i p-value dal più piccolo e li confronto con soglie via via meno severe",
    ("Esempio, 3 confronti, α = 0,05: p = 0,010 · 0,020 · 0,040", [
        "0,010 ≤ 0,05/3 = 0,017 ✓;  0,020 ≤ 0,05/2 = 0,025 ✓;  0,040 ≤ 0,05/1 ✓ → tutti significativi",
        "se il primo fosse 0,020 (> 0,017) ci si ferma subito: nessuno significativo",
    ]),
    "Noi: tre famiglie (ogni modello contro TF-IDF in Tabella 4: 24; le curve: 16 per modello; gli altri confronti "
    "del testo: 24)",
], [
    "Mistral contro TF-IDF, errore: **−2,0** punti, **significativo** dopo Holm",
    "Mistral contro TF-IDF, ρ: +0,06, **non** significativo dopo Holm (p = 0,07)",
    "Lo diciamo nel testo: il vantaggio solido è sull'**errore**",
])

nuove += appendice("D", "Chi resta fuori", [
    "«Il vostro dataset è **rappresentativo** delle testate politiche?»",
    "Se la pulizia toglie soprattutto un certo tipo di testate, i risultati valgono solo per le altre",
    "Ed è proprio su testate **non valutate** che il modello andrebbe usato",
], [
    "Seguiamo **ognuna** delle 1.293 testate politiche nei passi della pulizia",
    "Per ogni passo: quante escono, voto medio, livelli, orientamento, quota USA",
], [
    "Non raccolte: voto medio **41** contro **69**; **43%** nel livello più basso",
    "Tolte coi duplicati: quasi tutte la **rete da 164 siti**",
    "Non è rappresentativo: lo diciamo, e stimiamo l'errore su quella popolazione (Tabella 6)",
], "tab_selection")

nuove += appendice("E, F", "Impostazioni e prompt", [
    "«Avete scelto le impostazioni guardando il test?»",
    "«Lo zero-shot va male perché il prompt è fatto male?»",
    "Nel primo caso i risultati sarebbero **ottimisti**; nel secondo il confronto con lo zero-shot sarebbe **ingiusto**",
], [
    "**Penalità della ridge** e **layer di Mistral** (27 o 40) scelti su **3 fold interni** delle sole testate di "
    "training, mai sul test",
    "ModernBERT: impostazioni fisse e dichiarate (5 epoche, learning rate 3·10⁻⁵)",
    "Il prompt descrive i **nove criteri** NewsGuard con i loro pesi e i segnaposto; è riportato per intero",
], [
    "Tutte le impostazioni sono nell'appendice E: lo studio si può **riprodurre**",
    "Il zero-shot risponde quasi sempre **65 o 45** (85% delle risposte)",
    "Lo stesso Mistral con una testa lineare è il **migliore**: il limite è chiedere il voto, non il prompt",
])

nuove += appendice("G", "«Solo il nome»", [
    "«I modelli grandi **conoscono già** le testate dal pretraining: riconoscono la testata, non leggono il testo»",
    "Le testate famose sono su Wikipedia e nei siti di fact-checking, che i modelli hanno letto",
    "Se fosse così il modello **non servirebbe** per le testate piccole e sconosciute",
], [
    "Diamo al modello **solo il nome di dominio** (es. example.com), nessun articolo",
    "Stesso vettore congelato, stessa ridge, stessi fold",
    "Se il modello «sa» già il voto, dal solo nome dovrebbe andare quasi come dal testo",
], [
    "Dal solo nome: ρ **0,39** (Mistral) e **0,35** (Qwen); MAE 20,4 e 21,7",
    "Dal testo: ρ **0,74** e **0,71**; MAE 15,8 e 16,3",
    "Il solo nome vale quanto **l'orientamento** (ρ 0,36): il grosso viene dalla lettura degli articoli",
    "E il 96% degli articoli è stato pubblicato **dopo** il rilascio dei modelli",
])

nuove += appendice("H", "Altri split", [
    "«Lo split in 5 fold è stato **fortunato**?»",
    "Con 226 testate (circa 45 per fold) poche testate facili o difficili possono spostare il confronto",
    "Il vantaggio di Mistral su TF-IDF è di **2 punti**: serve sapere se regge con un'altra divisione",
], [
    "Altri **4 split** fatti allo stesso modo, con seed diversi",
    "Stessi modelli e stesse impostazioni; confrontiamo errore e ρ",
], [
    "Mistral batte TF-IDF di **2,0–2,3** punti in **ogni** split",
    "Lo split del paper è quello col vantaggio **più piccolo**",
    "L'errore di ogni modello varia di meno di mezzo punto",
], "tab_splits")

nuove += appendice("I", "Testate sorelle", [
    "«Le testate dello stesso proprietario sono davvero un problema? Si somigliano davvero?»",
    "Togliere le sorelle costa: **704** testate in meno; se non si somigliano, toglierle è inutile",
], [
    "Le **copie** prima e dopo la deduplicazione",
    "I **voti** delle sorelle",
    "Quanto aiutano le sorelle in training e se dipende dalla **somiglianza** dei testi",
], [
    "**98%** delle coppie di testate con articoli in comune ha lo stesso proprietario",
    "**23** testate su 31 hanno lo stesso voto di tutte le sorelle; una testata dista dalle sorelle **3,8** punti, "
    "due testate a caso **29,6**",
    "Con le sorelle in training la somiglianza col training sale da **0,19** a **0,60** e l'errore crolla (Tabella 7)",
])

nuove += appendice("J", "Anonimizzazione a regole", [
    "«I risultati **dipendono** dall'anonimizzazione scelta?»",
    "GLiNER sbaglia qualche nome e un'altra scelta toglierebbe nomi diversi",
    "Se i numeri cambiassero molto, il confronto fra modelli dipenderebbe da un dettaglio tecnico",
], [
    "Rifatta **tutta la pipeline** (anche ModernBERT) con 6 regole al posto di GLiNER",
    "Le regole tolgono solo la **testata** e il suo staff: firma, link, e-mail, account, autori, nome",
], [
    "I modelli linguistici cambiano di **meno di 0,25** punti",
    "Solo TF-IDF perde un po' (ρ −0,017): usa i nomi come parole",
], "tab_anon")

nuove += appendice("K", "Boilerplate", [
    "«I modelli riconoscono la testata dalle **righe fisse** (newsletter, donazioni, disclaimer) invece di leggere?»",
    "Sarebbe un altro modo di **riconoscere** la testata, come i nomi: l'anonimizzazione non lo blocca",
], [
    "Boilerplate = riga ripetuta in ≥ 3 articoli e ≥ 5% della testata: **3%** delle parole",
    "TF-IDF **riaddestrato** senza quelle righe",
    "1.000 articoli **con e senza** quelle righe, modelli salvati",
    "Mistral **scomposto token per token** (la previsione è una somma di contributi)",
], [
    "TF-IDF senza boilerplate: **17,5** contro 17,8",
    "Le righe fisse spostano Mistral di **~2** punti (deviazione standard delle previsioni 17,5)",
    "Il voto viene dal **corpo** degli articoli",
], "tab_reading")

nuove += appendice("L, N", "Il design e le stesse testate", [
    "«Regole, split e modelli sono stati **adattati** alle 226 testate su cui riportate i risultati?»",
    "Prima di questa pipeline ci sono stati 19 tentativi sulle **stesse testate**, di cui abbiamo visto i risultati",
    "Il rischio: aver tenuto le scelte che vanno bene **su queste** testate",
], [
    "Elenchiamo **tutto** quello che abbiamo provato (classificazione, Gemma, DeBERTa, 14 regressori, quantili)",
    "Le uniche scelte fatte dentro i fold sono **penalità** e **layer**",
    "Errori per livello e orientamento, modello per modello",
], [
    "Risultati stabili sui **5 split**; l'anonimizzazione rifatta cambia **meno di mezzo punto**",
    "Non è una replica su testate nuove: lo diciamo **chiaramente nei limiti**",
])

titoli_appendici = [S.titolo_di(x) for x in p[29:44]]
assert all("Appendice" in t for t in titoli_appendici), titoli_appendici
S.scrivi(IN, OUT, c, manifest, nuove)
print(f"{OUT.name}: {len(nuove)} slide (erano 44), {len(S.immagini)} immagini nuove")
