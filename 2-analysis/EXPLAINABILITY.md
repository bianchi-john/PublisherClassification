# Perché i transformer non battono un modello TF-IDF?

Analisi del 5 ottobre 2026, condotta **dopo** la versione del paper inviata a ECIR.
I numeri vengono dagli script `run_tfidf_lr_baseline.py`, `explain_tfidf.py`,
`explain_clustering.py`, `explain_publishers.py` ed `explain_network_cv.py`;
tutti gli output sono in `Results/derived/explain/`.

## In breve

Gran parte di ciò che i modelli misurano è **un'impronta di produzione**:
template, CMS, nomi di testate, stile da comunicato stampa. Riconoscere un
template è un compito lessicale, quindi una regressione logistica sulle parole
basta, e la comprensione semantica dei transformer non ha niente da aggiungere.

Queste impronte sono condivise da **reti di siti**. La più grande conta 130 dei
152 editori l1, tutti con lo stesso punteggio NewsGuard (32), e ha siti in ogni
fold. La separazione per editore impedisce di memorizzare un sito, ma non
impedisce di riconoscere i suoi fratelli. Tenendo ogni rete in un solo fold,
l'accuracy della TF-IDF sugli editori scende **da 0.68 a 0.57**.

Fuori dalle reti, dove le impronte non aiutano, tutti i modelli sono deboli
(0.40–0.44 di accuracy sugli editori). I transformer lì hanno un vantaggio a
livello di articolo, fino a +8 punti di F1 macro, ma non sopravvive
all'aggregazione.

Gli errori sono una proprietà delle testate, non dei singoli articoli. Intere
testate "sembrano" di un'altra classe, e tutti i modelli sbagliano le stesse.
Le predizioni separano grandi gruppi ma non seguono il punteggio dentro una
classe.

## 1. Le reti di siti

Ho cercato articoli di editori diversi che iniziano con lo stesso testo (le
prime 12 parole, con i numeri mascherati) e ho collegato gli editori che ne
condividono almeno due.

Che cosa hanno in comune i siti di una rete, cioè che cosa un modello può
imparare da un sito e ritrovare nei suoi fratelli. Le percentuali dicono in
quale quota degli articoli compare la parola, **nella rete contro il resto del
corpus**.

| Rete (esempi) | Siti | Classe e punteggio NewsGuard | Che cosa condividono | Parole tipiche: nella rete vs resto |
|---|---|---|---|---|
| Siti "locali" automatizzati (aroostooknews.com, boisecitywire.com) | 130 | tutti l1, tutti **32** | **Template e articoli generati.** Il 71% degli articoli apre con un testo presente in altri siti della rete, per esempio la didascalia *"Federal Energy Regulatory Commission (FERC) Commissioner Mark C. Christie \| www.ferc.gov"* in 105 siti. Molti pezzi sono statistiche per contea scritte da un modello fisso. | *county* 67% vs 11%; *short term* 52% vs 2%; *seasonal* 49% vs 1% |
| Giornali locali (andalusiastarnews.com, atmoreadvance.com) | 34 | tutti l3, tutti **72.5** | **Comunicati stampa ripubblicati e lo stesso CMS.** Gli stessi comunicati PR Newswire / GlobeNewswire escono su più siti, e la pagina ha lo stesso lettore audio. | *trinity audio player ready* 46% vs 2%; *published* 62% vs 11% |
| Altri giornali locali (alabamanow.com, salisburypost.com, brewtonstandard.com) | 6 + 4 + 3 | tutti l3, tutti **72.5** | Gli stessi comunicati del 21 marzo 2025 e lo stesso lettore audio: con ogni probabilità **la stessa catena della riga sopra**, che l'euristica divide in quattro reti. | *march 21* 38–46% vs 5%; *trinity audio* 42% vs 5% |
| Testate conservatrici (americanlookout.com, christianheadlines.com, dailysignal.com) | 18 | miste, da l1 a l5 | **Pezzi ripubblicati** su Trump e incorporamenti da Twitter. Il legame è debole: solo il 17% delle aperture è condiviso. | *trump* 63% vs 21%; *pic twitter* 22% vs 4% |
| Sottodomini HuffPost (data., editorial., elections.huffingtonpost.com) | 8 | tutti l4, tutti **87.5** | **La stessa pagina:** la scritta *advertisement* dei banner e i blocchi fissi del sito HuffPost. | *advertisement* 84% vs 4%; *huffpost* 49% vs 0% |
| Courier Newsroom (cardinalpine.com, coppercourier.com) | 7 | tutti l2, tutti **57** | **Lo stesso appello in fondo** (*"Support Our Cause…"*) e pezzi ripubblicati. Il legame lessicale è debole (15% delle aperture). | *photo* 33% vs 5% |
| Star News Network (georgiastarnews.com, tennesseestar.com) | 5 | tutti l2 (40 e 47.5) | **Gli stessi articoli firmati**, per esempio *"by Brett Rowland…"* in tutti e 5 i siti, e lo stesso blocco redazionale. | *reporter* 53% vs 3%; *email protected* 38% vs 1% |
| Western Journal (westernjournal.com, w-j.co) | 4 | tutti l4, tutti **82** | **Lo stesso modulo** di sondaggio e iscrizione in fondo a ogni articolo. | *western journal* 95% vs 0%; *poll entitles* 86% vs 0% |
| Siti "patriot" (gopdailybrief.com, patriotjournal.org) | 4 | 3 l2, 1 l3 | **Gli stessi titoli e pezzi** ripubblicati. | *joe biden* 42% vs 5%; *fox news* 38% vs 3% |

Il punto è nella terza colonna: **dentro quasi ogni rete il punteggio NewsGuard
è identico.** Il segnale "appartengo a questa rete" corrisponde quindi
esattamente all'etichetta, e un modello che riconosce la rete indovina la
classe senza dover valutare l'affidabilità.

- In reti di almeno 3 siti stanno **223 editori su 439** e **8.752 articoli su 19.476** (45%).
- Le parole tipiche sono calcolate su unigrammi e bigrammi senza stopword, come differenza tra la quota di articoli della rete che contiene la parola e la quota nel resto del corpus.

**Accuratezza sugli editori, a parità di classe** (stessi modelli del paper):

| Classe | Nelle reti | Fuori dalle reti |
|---|---|---|
| l1 | 0.98–0.99 (132) | 0.25–0.40 (20) |
| l2 | 0.73–0.86 (22) | 0.47–0.74 (47) |
| l3 | 0.89 (53) | 0.21–0.40 (42) |
| l4 | 0.79–0.86 (14) | 0.46–0.62 (74) |

Nel complesso: 0.92–0.93 nelle reti, 0.40–0.44 fuori. Lo 0.69 del paper è la
media di questi due mondi.

**Una conseguenza scomoda:** i siti inaffidabili *indipendenti* (l1 fuori dalle
reti) vengono riconosciuti solo nel 25–40% dei casi.

**Separazione per rete (solo TF-IDF; i transformer andrebbero riaddestrati su GPU):**

| | Separato per editore | Separato per rete |
|---|---|---|
| Accuracy articoli | 0.576 | 0.472 |
| F1 macro articoli | 0.527 | 0.432 |
| Accuracy editori | 0.681 | 0.569 |
| Tolleranza (α=1.0) editori | 0.879 | 0.777 |
| Accuracy editori fuori dalle reti | 0.421 | 0.417 |

Anche senza aver visto nessun sito fratello, il 77% della rete l1 viene ancora
classificato l1, e l'89% della rete l3 come l3: parte del segnale si trasferisce
davvero. Crollano invece le reti piccole: l4 da 0.86 a 0.14, l2 da 0.86 a 0.45.

## 2. Che cosa impara la TF-IDF

I termini con il peso più alto per classe (modello addestrato su tutto il corpus,
usato solo per la lettura e mai per valutare):

- **l1:** *official headshot, original source can be found here, congressman, rep, county, district, emphasized, highlighted*: lo stile da comunicato stampa della rete di 130 siti.
- **l2:** *Epoch Times, Project Veritas, RedState, NTD, advertisement, Nigeria, Saudi*: nomi di testate e argomenti.
- **l3:** *trinity audio player ready, email newsletter signup, published, March 21 2025*: il CMS dei giornali locali.
- **l4:** *read more, Getty, Sky News, GB News, Labour, all rights reserved*: testate britanniche e widget di pagina.
- **l5:** *FP, IrishCentral, Spotlight PA, view image in fullscreen*: nomi di testate, ognuno legato a un solo editore.

Tra i 200 termini migliori di ogni classe, la quota dominata da un solo editore
(più del 50% dei suoi articoli) va dal 16% (l1) al 33% (l4).

## 3. Che cosa non spiega la parità

| Esperimento (TF-IDF) | Acc. articoli | Acc. editori |
|---|---|---|
| Testo intero (riferimento) | 0.576 | 0.681 |
| Prime 190 parole (≈ 256 token, come BERT/RoBERTa/Llama) | 0.557 | 0.677 |
| Prime 750 parole (≈ 1.024 token, come Mistral) | 0.568 | 0.677 |
| Solo parole funzionali (stile) | 0.472 | 0.615 |
| Senza parole funzionali (contenuto) | 0.570 | 0.672 |
| Senza i 100 / 1.000 / 10.000 n-grammi più predittivi | 0.563 / 0.553 / 0.540 | 0.663 / 0.663 / 0.661 |
| Senza paragrafi di boilerplate (28% degli articoli toccati) | 0.573 | 0.681 |

- **La lunghezza del testo no:** troncando come i transformer si perdono 2 punti sugli articoli e 0.4 sugli editori.
- **Il boilerplate di pagina no** (newsletter, donazioni, pubblicità): nessun effetto.
- **Poche parole chiave no:** il segnale è diffuso e ridondante, e resiste anche alla rimozione di 10.000 termini, come ci si aspetta da un template ripetuto ovunque.
- **Stile e contenuto portano segnale in modo ridondante:** le sole parole funzionali arrivano già a 0.62 sugli editori.

## 4. Struttura degli errori

- **TF-IDF e transformer imparano cose simili.** La concordanza delle predizioni (kappa di Cohen) tra TF-IDF e transformer vale 0.55–0.58; tra transformer vale 0.55–0.64. Tra il 72% e il 76% degli errori della TF-IDF sono errori anche di ciascun transformer.
- **Il tetto è comune.** Il 21% degli articoli è sbagliato da tutti e cinque i modelli, il 42% è giusto per tutti. Gli ensemble aggiungono circa 1 punto sugli articoli e niente sugli editori.
- **Gli errori sono un fatto di testata.** La correlazione intra-editore degli errori (ICC) vale 0.59–0.67, contro 0.12–0.16 attesi permutando gli errori dentro la stessa classe. Circa un quarto degli editori ha più dell'80% di articoli sbagliati, circa metà meno del 20%. Gli articoli "anomali" (sbagliati da tutti dentro editori giusti per tutti) sono solo lo 0.3%: il rumore delle etichette sui singoli articoli non è il problema.
- **82 editori sono sbagliati da tutti e cinque i modelli:** 23 dei 35 l5, 25 l3, 14 l4. In 58 casi su 82 di un solo livello.
- **Le predizioni non seguono il punteggio dentro le classi.** La correlazione di Spearman tra livello medio predetto e punteggio NewsGuard è 0.74–0.81 nel complesso, ma quasi nulla dentro le classi: l1 da −0.26 a −0.31, l3 da −0.21 a 0.11, l4 da −0.15 a 0.14. Quella negativa in l1 si spiega con la rete: i 130 siti hanno punteggio 32 e sono predetti bene, mentre gli l1 indipendenti hanno punteggi più bassi e sono predetti più in alto.
- **La vicinanza alle soglie tra classi non spiega gli errori:** il tasso d'errore per distanza dalla soglia non è monotòno.

## 5. Risposta alla domanda

I transformer non battono la TF-IDF perché **la parte del compito che si riesce a
risolvere è soprattutto riconoscimento della fonte**, e per riconoscere una fonte
bastano le parole. Dove le impronte non aiutano, il compito torna difficile per
tutti.

Un'ipotesi non verificata qui: il testo degli articoli contiene poco di ciò che
NewsGuard valuta. Diversi dei nove criteri riguardano il sito, non gli articoli
(proprietà e finanziamento, politica di correzione, etichettatura della
pubblicità, chi è responsabile).

## 6. Che cosa ne segue

**Per la valutazione:** la separazione per editore è necessaria ma non basta.
Serve una separazione per rete o per proprietario. È la stessa tesi del paper,
spinta un livello più in alto, e un'estensione naturale per la rivista.

**Per i modelli**, nell'ordine in cui li proverei:
1. **Togliere le scorciatoie:** deduplicare il testo condiviso tra siti, mascherare nomi di testate e template, valutare a parte le testate indipendenti.
2. **Aggiungere segnali a livello di sito, recuperati con tecniche di IR:** pagine "About", "Corrections", "Ownership" e masthead, cioè proprio i criteri di NewsGuard assenti dagli articoli. È anche un taglio più IR del problema.
3. **Addestrare a livello di editore**, con modelli su insiemi di articoli (multi-instance learning), visto che l'errore è un fatto di testata.
4. **Riaddestrare almeno un transformer con la separazione per rete**, per vedere se il suo vantaggio fuori dalle reti cresce quando le scorciatoie spariscono.

## Limiti di questa analisi

- **Le reti sono rilevate con un'euristica** (testo iniziale condiviso). Può mancare reti dello stesso proprietario senza testo comune, e può unire siti che si limitano a ripubblicare contenuti (il gruppo misto da 18).
- **Solo la TF-IDF è stata riaddestrata con la separazione per rete;** per i transformer ho solo l'analisi dentro/fuori le reti sulle predizioni esistenti.
- **Con la separazione per rete un fold contiene tutti i 130 siti l1,** quindi in quel fold il training ha pochi l1. I pesi di classe bilanciati attenuano lo squilibrio, ma non lo eliminano.
- **I file con le predizioni per articolo contengono le etichette NewsGuard:** restano nel repo privato e non vanno pubblicati.
