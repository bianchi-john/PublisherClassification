"""Revisione del 5 novembre di FinalState.odp (il file di John con i suoi commenti): da qui in poi SOLO i regressori.
Il voto previsto si valuta con la distanza dal voto NewsGuard vero; le classi compaiono solo nel confronto con Media
Bias/Fact Check, trasformando il voto in classe MBFC in tre modi.

Le slide di John (1-17 e le slide di sezione) restano com'erano, con poche modifiche puntuali chieste nei commenti; le
slide 18-49 erano state generate da aggiorna_slide.py e non toccate a mano (solo commentate), quindi si rigenerano.
Prima di eseguire: figure/genera_grafici.py, figure/grafici_modelli.py, figure/grafici_valutazione.py, figure/schermate_codice.py.
Uso:  python aggiorna_regressori.py <file di John.odp> <file in uscita.odp>"""
import re
import sys
import zipfile
from pathlib import Path

import stile_slide as S
from stile_slide import FIG, con_codice, con_grafico, pagina

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S.PREFISSO[0] = "zr"
z = zipfile.ZipFile(IN)
c = z.read("content.xml").decode()
manifest = z.read("META-INF/manifest.xml").decode()
c = S.aggiungi_stili(c)
pagine = re.findall(r"<draw:page .*?</draw:page>", c, flags=re.S)
assert len(pagine) == 49, len(pagine)
titoli = [S.titolo_di(p) for p in pagine]
V, M = FIG / "valutazione", FIG / "modelli"


def numero(n, titolo):
    """La slide n (da 1) del file di John, controllando il titolo."""
    assert titolo in titoli[n - 1], (n, titoli[n - 1])
    return S.senza_commenti(pagine[n - 1])


# ------------------------------------------------------------------------------------------------ le slide di John, ritoccate
prima = [numero(i, titoli[i - 1]) for i in range(1, 13)]
anonimizzazione = S.cambia_immagine(numero(13, "Anonimizzazione"), FIG / "anonimizzazione.png")       # senza i titoli
split = S.togli_punto(numero(14, "split"), "Lo stesso split per classificazione e regressione")
lunghezza = S.togli_punto(numero(16, "Gestione lunghezza articoli"), "DeBERTa")
scelta = numero(17, "Scelta dei modelli")
for vecchio, nuovo in [("TF-IDF + regressione logistica / ridge", "TF-IDF + regressione ridge"),
                       ("Classificazione: i 5 livelli. Regressione:", "Un solo compito: il **voto NewsGuard esatto** (0-100), cioè una regressione"),
                       ("Dati salvati: logit / punteggio", "Dati salvati: il voto previsto di ogni articolo per ciascun fold, oltre al modello relativo a ogni fold")]:
    scelta = S.riscrivi(scelta, vecchio, nuovo)
tfidf_3 = numero(20, "TF-IDF")
assert "Contare le parole" in numero(18, "TF-IDF")          # rigenerata: non toccata a mano, il terzo punto finiva sotto il grafico
tfidf_1 = con_grafico("TF-IDF", "Contare le parole, pesare quelle rare", [
    "Ogni articolo diventa un vettore: un peso per ogni parola e coppia di parole (**100.000** in tutto)",
    "Peso alto = parola frequente nell'articolo ma **rara negli altri**",
    "Vocabolario e pesi calcolati **solo sugli articoli di training**",
], M / "tfidf_esempio.png")
sezione_addestramento, giri = numero(25, "Addestramento"), numero(26, "Addestramento")
sezione_valutazione = numero(31, "Valutazione")

# ------------------------------------------------------------------------------------------------ lo split: il confronto visivo
split_perche = con_grafico("Lo split", "Per voto o per livello: quanto cambia", [
    ("Il regressore impara il **voto esatto**: ogni fold deve coprire **tutta la scala**", ["con i fold per livello, dentro un livello il voto era lasciato al caso"]),
    "Le stesse 226 testate divise nei due modi: cambia poco, **solo dentro i livelli**",
], FIG / "split_confronto.png")

# ------------------------------------------------------------------------------------------------ i modelli
modelli = [
    tfidf_1,
    con_codice("TF-IDF", "Sopra, una regressione ridge", [
        "Ogni articolo ha come obiettivo il **voto della sua testata**",
        "**Ridge**: voto = somma di (peso × TF-IDF) su tutte le parole, più una costante",
        "Impara i pesi che rendono minimo l'**errore al quadrato**, più una **penalità** sui pesi grandi",
    ], ["tfidf_matrici", "ridge"]),
    tfidf_3,
    con_codice("Qwen3-Embedding-8B", "Un modello di embedding, congelato", [
        ("8 miliardi di parametri, trasforma un testo in un vettore di **4.096 numeri**", ["4.096 è la lunghezza del vettore, non dei token"]),
        "Legge un'istruzione + l'articolo intero; il vettore è l'**ultimo token**",
        ("Il limite di 4.096 token è solo un **tetto**: mediana ~760, massimo 3.000", ["nessun posto vuoto: articoli di lunghezza simile insieme, il riempimento (padding) è ignorato"]),
        "**Non lo addestriamo**: sopra, lo stesso ridge del TF-IDF",
    ], ["qwen"]),
    con_codice("Mistral-Small-24B", "Un modello linguistico grande, congelato", [
        "24 miliardi di parametri, versione **Base**; caricato a **4 bit** per entrare nella GPU (32 GB)",
        "Vettore = **media** degli stati interni di tutti i token, strato 27 di 40 (5.120 numeri)",
        "Anche qui **non lo addestriamo**: sopra, il ridge",
    ], ["mistral"]),
    con_codice("ModernBERT", "Un encoder piccolo, addestrato tutto", [
        "149 milioni di parametri; lo **addestriamo tutto** sui nostri articoli",
        "Potrebbe leggere **8.192** token: lo fermiamo a **4.096** come gli altri (l'articolo più lungo ne ha 2.635)",
        "Una sola uscita, il voto / 100; impara rendendo minimo l'**errore al quadrato**",
    ], ["modernbert_reg"]),
    pagina("Un solo compito: il voto", "Perché niente classi", [
        "NewsGuard dà un **voto 0-100**: lo prevediamo direttamente (**regressione**)",
        ("Le classi sono confini scelti da noi", ["59 e 60 finiscono in classi diverse, 60 e 74 nella stessa"]),
        "Ogni articolo riceve un voto; la testata = la **media** dei voti dei suoi articoli",
        "Valutiamo la **distanza** dal voto vero; le classi solo per il confronto con **MBFC**",
    ]),
]

# ------------------------------------------------------------------------------------------------ l'addestramento
addestramento = [
    sezione_addestramento, giri,
    con_grafico("Modelli lineari", "La penalità si sceglie solo con il training", [
        "La **penalità** dice quanto il ridge tiene piccoli i pesi: bassa impara a memoria, alta non impara",
        "Si sceglie **senza guardare il test**: con le sole testate di training, in 3 prove",
    ], M / "cv_interna.png"),
    con_codice("Modelli lineari", "Perché 3 prove interne", [
        "In training ~180 testate: con 3 gruppi restano **~60 testate** per valutare ogni prova",
        "Con 5 o 10 gruppi ne resterebbero 36 o 18: lo Spearman su così poche testate **oscilla** troppo",
        "Costo: TF-IDF, 7 penalità × 3 prove = **21 modelli** in ogni giro",
        "Come lo split esterno, i gruppi sono **divisi per voto**",
    ], ["griglie_reg", "fold_lineare"]),
    con_grafico("Modelli lineari", "Le penalità provate e quelle scelte", [
        "Ogni curva: lo Spearman medio delle 3 prove per ogni penalità; il cerchio è la scelta",
        "TF-IDF: curva quasi **piatta**, la penalità conta poco; Qwen e Mistral: un massimo chiaro",
        "Sul bordo della griglia solo TF-IDF (giro 2) e Qwen (giro 1), dove la curva è piatta",
    ], M / "penalita.png"),
    con_grafico("ModernBERT", "5 epoche, nessuna validazione", [
        "Learning rate 3·10⁻⁵, batch **32** (4 × 8 accumulati); warm-up 6%, poi decrescita lineare; si tiene l'ultima epoca",
        "**Gradiente tagliato a 1**: se un passo è troppo lungo (norma > 1) lo si accorcia a 1; evita salti che rovinano il modello",
        "Sul training l'errore per articolo scende a **~4 punti**, sul test resta **~25**: impara le testate **a memoria**",
    ], M / "bert_curva.png"),
    con_codice("ModernBERT", "Il ciclo di addestramento", [
        "clip_grad_norm_ è il taglio del gradiente; ogni 8 batch da 4 un passo dell'ottimizzatore",
    ], ["addestramento_bert_reg"]),
    con_codice("Riproducibilità", "Ogni cosa ha un seme", [
        "Seme **42** per tutto; lo split ha seme 1; ridge con random_state fisso",
        ("ModernBERT: seme 42 + numero del giro", ["fissa i pesi iniziali dell'uscita, il dropout e l'ordine dei batch; algoritmi deterministici sulla GPU"]),
        "Qwen e Mistral non hanno niente di casuale: **leggono soltanto** (niente addestramento, niente dropout)",
        "Salviamo il voto di **ogni articolo di ogni fold** e il modello di ogni fold",
    ], ["riproducibile", "seme_bert"]),
]

# ------------------------------------------------------------------------------------------------ la valutazione: distanza dal voto vero
valutazione = [
    sezione_valutazione,
    con_codice("Valutazione", "Dagli articoli alla testata", [
        "Ogni articolo ha un voto previsto; la testata = la **media** dei voti dei suoi articoli",
        "Ogni testata è prevista **una volta sola**, nel giro in cui era nel test",
        "Tutte le misure sono **per testata** (226), non per articolo",
    ], ["articoli_testata_reg"]),
    con_grafico("Valutazione", "Come si valuta un regressore", [
        "Nessuna classe: per ogni testata la **distanza** tra voto previsto e voto NewsGuard",
        "**Errore medio** (MAE, punti) · **entro 10 / 20 punti** (% di testate) · **Spearman** (l'ordine delle testate)",
        "Riferimento **senza testo**: a tutte la media del training (errore 22,6)",
    ], V / "esempio_distanza.png"),
    con_grafico("Valutazione", "Intervalli al 95%: il bootstrap", [
        "Le 226 testate sono un **campione**: con altre testate i numeri cambierebbero un po'",
        "Si ripescano 226 testate **a caso con ripetizione** e si ricalcola la misura, **2.000 volte**",
        "Intervallo al 95% = dove cadono 95 valori su 100",
        "Due modelli si confrontano sulle **stesse testate ripescate**: conta l'intervallo della differenza",
    ], V / "bootstrap.png"),
    con_codice("Valutazione", "Il codice del bootstrap", [
        "Le coppie (voto vero, voto previsto) restano fisse: si ripescano solo le testate",
    ], ["bootstrap"]),
    con_grafico("Regressori", "Voto vero e voto previsto", [
        "Le previsioni sono **schiacciate verso il centro**: TF-IDF, Qwen e Mistral non scendono sotto ~35",
    ], V / "reg_dispersione.png"),
    con_grafico("Regressori", "Errore e ordine", [
        "Errore medio: Mistral **15,7**, Qwen 16,1, TF-IDF 17,7, ModernBERT 17,7; senza testo 22,6",
        ("Spearman e errore misurano cose diverse: l'**ordine** e la **distanza**", [
            "l'ordine è buono (0,73), ma i voti sono schiacciati: agli estremi la distanza resta grande"]),
    ], V / "reg_mae_spearman.png"),
    con_grafico("Regressori", "Quante testate sono vicine al voto vero", [
        "Entro **10 punti**: Mistral 37%, Qwen 35%, TF-IDF 32%; senza testo 18%",
        "Entro **20 punti**: Mistral e Qwen 69%, TF-IDF 61%; senza testo 51%",
    ], V / "reg_entro.png"),
    con_grafico("Regressori", "Dove sbagliano", [
        "Le testate peggiori (0-39) sono **sovrastimate** di 31-37 punti, quelle da 100 **sottostimate** di 19-25",
        "In mezzo (60-74) l'errore è piccolo: è l'effetto dello schiacciamento",
    ], V / "reg_per_fascia.png"),
    con_grafico("Regressori", "Stirare i voti", [
        "In ogni giro il voto previsto diventa il voto del training **allo stesso posto in classifica**",
        "L'ordine non cambia, le distanze sì: Mistral **15,7 → 15,2**, TF-IDF 17,7 → 16,3",
        "Se l'ordine è sbagliato, stirare allarga anche gli errori: ModernBERT 17,7 → 18,1",
    ], V / "reg_stirati.png"),
    con_codice("Regressori", "Il codice dello stiramento", [
        "Come stirare un'immagine: la distribuzione dei voti previsti diventa quella dei voti veri del training",
    ], ["stirare"]),
    con_grafico("I modelli contro il TF-IDF", "Differenza sulle stesse testate", [
        "Mistral e Qwen sbagliano **1,9 e 1,6 punti in meno** del TF-IDF: l'intervallo non tocca lo 0",
        "Entro 20 punti: Qwen **+8** e Mistral **+7,5** punti %; ModernBERT pari sull'errore, peggio sull'ordine",
    ], V / "contro_tfidf.png"),
    con_grafico("Il DNA", "Quanti articoli servono", [
        "Mistral: con **1** articolo Spearman 0,63 ed errore 17,1; con **5** 0,71 e 16,0; con tutti 0,73 e 15,7",
        "Il voto della testata si legge già in **pochi articoli**",
    ], V / "dna.png"),
]

# ------------------------------------------------------------------------------------------------ Media Bias/Fact Check: l'unico posto con le classi
mbfc = [
    pagina("Media Bias/Fact Check", "Un secondo esperto", [
        ("MBFC valuta il **Factual Reporting** di una testata", ["Very Low, Low, Mixed, Mostly Factual, High, Very High",
                                                                "Very Low e Very High: 1 testata ciascuna → unite a Low e High: **4 classi**"]),
        "Testate in comune (stesso dominio): **170 su 226**",
        "MBFC non entra mai nell'addestramento: serve solo come **riferimento**",
        "Due domande: i due esperti quanto concordano? E il modello con loro?",
    ]),
    con_grafico("NewsGuard e MBFC", "Quanto concordano gli esperti", [
        "Spearman **0,71**: più alta la classe MBFC, più alto il voto NewsGuard, ma con molte eccezioni",
        "Le «Mixed» di MBFC vanno da 0 a 100 in NewsGuard",
    ], V / "mbfc_esperti.png", largo_max=19),
    con_grafico("Il modello come annotatore", "Senza classi: lo Spearman", [
        "Il **voto previsto** dal regressore, confrontato con il voto NewsGuard e con la classe MBFC",
        "Mistral con NewsGuard **0,73**, come i due esperti tra loro (**0,71**)",
        "Con MBFC un po' meno (**0,63**): il modello ha imparato NewsGuard",
    ], V / "mbfc_accordo.png"),
    pagina("Dal voto alle classi MBFC", "Perché e come", [
        "MBFC dà una **classe**, non un voto: per contare le testate «nella classe giusta» il voto va trasformato",
        "Lo stesso modo per il **voto vero di NewsGuard** (l'esperto) e per il **voto previsto** (il modello)",
        "Si impara sulle testate degli **altri 4 fold** e si applica al fold di test",
        ("Tre modi:", ["**quote**: in ordine di voto, si taglia con le proporzioni delle classi MBFC",
                       "**mediane**: ogni classe ha il suo voto tipico; si va alla classe col voto tipico più vicino",
                       "**soglie**: i 3 tagli del voto che sbagliano meno classi"]),
    ], corpo=20),
    con_grafico("Dal voto alle classi MBFC", "Tre modi di tagliare il voto", [
        "Sul voto NewsGuard delle 170 testate: i tagli cambiano molto da un modo all'altro",
    ], V / "mbfc_metodi.png"),
    con_codice("Dal voto alle classi MBFC", "Il codice: quote e mediane", [
        "by_fold: la trasformazione si impara sugli altri fold e si applica al fold di test",
    ], ["mbfc_quote_mediane", "mbfc_fold"]),
    con_codice("Dal voto alle classi MBFC", "Il codice: le soglie", [
        "Programmazione dinamica: trova **in modo esatto** i tagli che sbagliano meno",
    ], ["mbfc_soglie"]),
    con_grafico("Il modello come annotatore", "Con le classi MBFC", [
        "NewsGuard (l'esperto): classe giusta **49-53%**, kappa 0,62-0,70",
        "Mistral: **45-49%**, kappa 0,58-0,62: poco sotto l'esperto; sempre «Mixed»: 38%",
    ], V / "mbfc_classi.png"),
    con_grafico("Il modello come annotatore", "Dove sbagliano", [
        "Con le quote: NewsGuard **90** testate giuste su 170, Mistral **83**",
        "Errori quasi sempre tra classi **vicine**; mai Low ↔ High",
    ], V / "mbfc_incrocio.png", largo_max=21),
    pagina("In breve", "", [
        "Dal **solo testo** anonimo si prevede il voto NewsGuard di testate mai viste: errore **15,7** punti (senza testo 22,6)",
        "I modelli grandi congelati sono i migliori; il TF-IDF segue; ModernBERT impara a memoria",
        "I voti previsti sono **schiacciati**: l'ordine è buono (Spearman 0,73), gli estremi no",
        "Mistral concorda con NewsGuard **quanto MBFC**; in classi MBFC, poco sotto l'esperto",
    ]),
]

finale = prima + [anonimizzazione, split, split_perche, lunghezza, scelta] + modelli + addestramento + valutazione + mbfc
S.scrivi(IN, OUT, c, manifest, finale)
print(len(finale), "slide (erano 49),", S.contatore[0], "punti nuovi ->", OUT)
