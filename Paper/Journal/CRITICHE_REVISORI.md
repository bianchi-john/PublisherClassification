# Critiche possibili dei revisori (bozza del 17/11/2026)

Due revisori indipendenti hanno letto la bozza rivista (`main.tex` dopo il commit 6d290f9). Il primo ha guardato metodo,
statistica e coerenza con il codice; il secondo contributo, novità, affermazioni ed etica. Ogni critica qui sotto è stata
**ricontrollata sui dati, sul codice o sulle fonti**. Le critiche generiche o sbagliate sono state scartate. Le cifre
vengono dai `paper_*.csv` o da ricalcoli sulle previsioni salvate.

**Verdetto.** Nessun difetto fatale: il risultato centrale regge a ogni prova. Mistral congelato + ridge ha un MAE più
basso di TF-IDF su testate mai viste, in tutte e cinque le divisioni, sulle testate piccole e anche dopo una
ricalibrazione. Ma diverse affermazioni di punta vanno oltre ciò che gli esperimenti isolano. Tre sono errori introdotti
oggi nella revisione: la lettura dell'orientamento come compressione, i «28 test» e le «89 sorelle tolte dall'ultima
regola».

Gravità: 🔴 grave (un revisore la userebbe per chiedere una revisione maggiore) · 🟠 media · 🟡 minore.
Costo: **T** solo testo · **C** analisi su CPU (minuti od ore) · **G** GPU.

## Stato (17/11 sera): correzioni di solo testo fatte in `main.tex`

- **2, orientamento.** Riscritto: il divario viene da segnali legati all'orientamento, non dalla compressione; sulla
  calibrazione «nessuna prova», non «uguaglianza»; abstract, discussione e lavori futuri allineati.
- **4, sorelle.** Corretti 89/61; il gradino misura copie e proprietà insieme (scritto); split per articolo sulle stesse
  31 testate (11,7 / 12,8 / 9,6 contro 8,3 / 11,6 / 9,7: regge); merito a Zhou et al. per la somiglianza.
- **5, anonimizzazione.** Tolto «every explicit trace»; misura indipendente con il Brand Name (2,1%), handle (9,5%) e
  alias dichiarati; scritto che i modelli leggono solo il corpo.
- **6–7, dimensione e fine-tuning.** «I modelli più grandi», le dimensioni dei vettori e i limiti del confronto con
  ModernBERT dichiarati; corretta la frase sugli iperparametri (quelli di ModernBERT sono fissi); confronto della
  memorizzazione tra grandezze omogenee.
- **8, statistica.** 24 test; il confronto con tutti gli articoli è riportato dalla famiglia principale; una sola
  convenzione per «significativo»; le curve dicono «da 1 a 50 articoli».
- **9.** Nuova appendice «What else was tried on these outlets».
- **10.** Citati Burdisso 2024, CrediBench 2026 ed Eldifrawi 2026; novità ristretta a «solo testo anonimizzato». Gli
  altri lavori segnalati dal revisore vanno verificati prima di citarli.
- **12, etica.** Corrette le due frasi false; frase su robots.txt e una richiesta al secondo; `\todo` sulla licenza
  NewsGuard (decisione tua). La modalità pubblica del tool è ancora da decidere.
- **15, minori.** Data dell'estrazione NewsGuard, aggregazione, appendice A, curve di Qwen 0,6B, rilascio dei quattro
  modelli, intervallo del tool, pesi di addestramento, determinismo, «cheapest», uso per l'ordinamento, legami di
  proprietà mancati, frase sull'ecosistema mediatico; abstract ridotto da 429 a 298 parole.

**Restano:**
- le analisi: 1 (ricalibrazione), 3 (sonda «solo nome»), 4 (gradino senza copie), 5 (senza le 17 testate con il
  nome), 8 (TOST), 11 (punteggio di Lin), 13 (teste non lineari), 14 (MAE ripesato);
- le decisioni: notorietà, ricalibrazione, licenza, metafora del DNA, modalità pubblica del tool.

## Stato (17/11, 17:30): piano «conciso» in corso

- **Concisione.** Testo principale da circa 10.400 a circa 8.000 parole, da 31 a 22 pagine. Le appendici sono in
  `supplementary.tex` (7 pagine, rimandi con `xr-hyper`); tolte la figura a barre dei risultati e la metafora del DNA;
  la figura dell'anonimizzazione è nel supplementare.
- **Passo 17** (`robustness.py`), fatto, una frase nel paper per ogni controllo:
  - **1, ricalibrazione** annidata: MAE Mistral 14,7, TF-IDF 16,2, zero-shot 21,2; Mistral − TF-IDF −1,5 [−2,7; −0,4],
    Qwen non più significativo;
  - **14, MAE ripesato** sulle testate non raccolte: Mistral 20,2, TF-IDF 22,9, costante 34,9;
  - **5, senza le 16 testate** con il nome rimasto: Mistral − TF-IDF −1,8.
- **3, sonda «solo nome»**, fatta (18/11, 01:10): dal solo nome di dominio ρ 0,39 (Mistral) e 0,35 (Qwen), MAE 20,4 e
  21,7, quanto il solo orientamento e molto meno del testo (0,735); una frase nella scala e una nei limiti.
- **Non fatte, per scelta** (il testo dichiara il limite): sorelle senza copie, Qwen 4B, semi di ModernBERT, Lin et al.,
  TF-IDF non lineare, testate nuove, TOST.

## Stato (18/11): 10, 11 e 12 trattati solo nel testo (decisione di John)

- **10, posizionamento.** Eldifrawi et al. 2026 letto per intero: 3 classi da MBFC su 300 domini, Llama-3.3-70B con
  ricerca web e link, domini separati in un solo esperimento, nessun mascheramento; aggiunto alla tabella di confronto.
  Citati e verificati Kiesel et al. 2019, Potthast et al. 2018, Gruppi et al. 2021, Buckmann et al. 2025. McElroy 2025
  non citato (dall'abstract non si capisce come divide i dati).
- **11, un solo valutatore.** Nei limiti: valutatore unico e proprietario; una replica pubblica potrebbe usare i voti
  aperti di Lin et al., che includono NewsGuard. Nessuna analisi.
- **12, etica.** Tolta dal supplementare la frase sulla modalità pubblica del tool. Restano a John: la licenza NewsGuard
  per modelli e previsioni (`\todo` nel testo) e se tenere la modalità pubblica nel tool.

---

## 🔴 1. Il MAE misura soprattutto la calibrazione

**Il problema.**
- Tutti i modelli comprimono le previsioni. La compressione nasce dall'addestramento sul singolo articolo, non dalla
  media: la pendenza della previsione sul voto vero è la stessa per articolo e per testata (Mistral 0,366 contro
  0,374; TF-IDF 0,244 contro 0,241).
- La frase di §5.6 «the mean of noisy article predictions is itself pulled towards the centre» è quindi sbagliata.
- Anche «a structural property of the task» non è dimostrata: è una proprietà del nostro disegno.

**La prova.** Una ricalibrazione lineare per testata (voto ~ a + b·previsione, stimata sulle testate degli altri fold)
cambia parecchio i numeri:

| | MAE ora | ricalibrato |
|---|---|---|
| TF-IDF | 17,8 | 16,4 |
| Qwen3-8B | 16,3 | 15,5 |
| Mistral-24B | 15,8 | 14,9 |
| zero-shot | 24,4 | 21,2 (meglio della costante, 22,1) |

- Mistral − TF-IDF scende da −2,0 a −1,55 [−2,68; −0,44]: resta significativo.
- Qwen − TF-IDF scende a −0,94 [−2,05; +0,14]: non è più significativo.
- «Lo zero-shot fa peggio di una costante» è quindi un effetto della calibrazione.

**Un'aggravante.** La penalità della ridge si sceglie con Spearman, che ignora la scala, mentre la misura principale è il
MAE.

**Cosa fare.**
- Aggiungere una ricalibrazione lineare per testata, stimata dentro i fold di addestramento (annidata), e riportare il
  MAE prima e dopo. **C**, meno di un'ora.
- Oppure scegliere la penalità con il MAE.
- In ogni caso correggere le frasi di §5.6 e lo zero-shot nell'abstract. **T**

**Nota.** Lo «stiramento» dei voti era stato escluso dal paper: i revisori lo chiederebbero comunque. Consiglio la
ricalibrazione lineare, che è la versione più semplice.

## 🔴 2. Orientamento: la lettura «è la compressione» è sbagliata (l'ho scritta io oggi)

**Il problema.**
- Il predittore di riferimento (12,6 punti) non comprime verso la media generale ma verso la media dell'orientamento di
  ciascuna testata. Comprimere verso la media generale dà un divario esattamente zero (verificato).
- Il divario di 9,6 punti dice quindi che le previsioni contengono informazione legata all'orientamento oltre al voto
  vero. Non è un effetto meccanico della compressione.
- Ricalibrando, il divario **aumenta**: Mistral da 9,6 a 13,3, TF-IDF da 6,7 a 11,7. Le frasi «the gap would only shrink
  with less compression» (§5.7) e «reduce the compression … and the orientation gap that follows from it» (lavori futuri)
  dicono il contrario dei dati.
- «Equal predictions correspond to equal true scores» si basa su una differenza non significativa con un intervallo di
  16 punti (0,8 [−7,5; 8,9]). È assenza di prova, non prova di uguaglianza; per TF-IDF la stima puntuale è 7,1.
- Kleinberg et al. riguarda esiti binari: per un punteggio continuo va adattato o detto esplicitamente.

**Cosa fare** (**T**, più un test di equivalenza **C**).
- Riscrivere: i modelli usano segnali legati all'orientamento, che è il comportamento atteso di un predittore calibrato
  che sfrutta tratti informativi, e questo produce un divario a parità di voto.
- Sulla calibrazione dire «nessuna prova di cattiva calibrazione, intervallo largo», con un test di equivalenza (TOST)
  a margine dichiarato.
- Togliere dall'abstract «no more than the compression alone».
- Correggere i lavori futuri.

## 🔴 3. «Leggere, non riconoscere»: la conoscenza delle testate dal pre-addestramento non è esclusa

**Il problema.**
- Le testate sono separate tra addestramento e test, ma gli LLM da 8 e 24 miliardi di parametri hanno probabilmente letto
  per anni queste testate e quello che si dice di loro.
- Possono riconoscerle dallo stile, o dai nomi rimasti (punto 5), e portarsi dietro la loro reputazione. TF-IDF non può
  farlo.
- Il limite scritto nel paper (gli articoli sono più recenti dei modelli) non risponde: il rischio riguarda la conoscenza
  delle testate, non degli articoli.
- Yang e Menczer, che citiamo, mostrano che gli LLM valutano le testate dal solo nome di dominio.

**Un indizio nei nostri dati.** Era l'analisi della notorietà, poi tolta. Il guadagno di Mistral era −3,0 punti nella metà
più nota delle testate (significativo) e −1,1 nella meno nota (non significativo). Entrambi i revisori, senza sapere
niente di quell'analisi, lo indicano come il rischio principale.

**Cosa fare.**
- **Sonda «solo nome».** Si calcola il vettore congelato del solo nome di dominio o del nome della testata, con la stessa
  CV e la stessa ridge. Se predice il voto di testate mai viste, la rappresentazione contiene la loro reputazione. **G**,
  pochi minuti.
- **Zero-shot dal solo nome di dominio.** Si guarda se gli errori del modello sugli articoli lo seguono. **G**, pochi
  minuti.
- **Notorietà:** riportare la divisione per notorietà già calcolata, o simile. **C**
- Comunque ammorbidire «reads, not recognizes» (abstract, introduzione, conclusione). **T**

## 🔴 4. La scala delle sorelle mescola copie e proprietà, e i conteggi sono sbagliati

**Il problema.**
- Le 89 sorelle non sono «quelle che solo l'ultima regola ha tolto». Sono le testate con almeno 10 articoli prima della
  deduplicazione. L'ultima regola ne toglie 61 (Tabella 2), le altre 28 le ha tolte la deduplicazione. Da qui anche 31
  contro 20 testate con una sorella.
- Il gradino aggiunge tutti i loro articoli, copie comprese: il 50% degli articoli di test ha una quasi-copia tra quelli
  delle sorelle. Misura quindi copie e proprietà insieme, mentre §4.3 dice che misura il residuo dopo la deduplicazione,
  e la conclusione chiama «necessaria» la regola di una testata per gruppo.
- Il confronto «al livello dello split per articolo» mette 31 testate contro 226. Sulle stesse testate regge (verificato
  dal revisore), ma il paper non lo mostra.
- **Novità:** «conta la somiglianza dei testi» l'ha già mostrato Zhou et al. 2021 (EACL), nella sezione «Performance
  Variance and Site Similarity Analysis» (verificato nel PDF). Lo citiamo solo per lo split per articolo.

**Cosa fare.**
- Aggiungere il gradino «sorelle senza copie», cioè senza gli articoli delle sorelle che sono quasi-copie di articoli del
  paper. **C**, circa 40 minuti, i vettori ci sono già.
- Riportare lo split per articolo sulle stesse 31 testate. **C**
- Correggere 89/61 e 31/20. **T**
- Dare il merito a Zhou per la somiglianza; il nuovo è la rete di copie tra testate dello stesso proprietario e il fatto
  che queste condividono il voto. **T**

## 🔴 5. L'anonimizzazione è sopravvalutata, e la sua misura è circolare

**Il problema** (tutto verificato).
- Gli **handle** (@FoxNews, @nomegiornalista) non sono mascherati: il 9,5% degli articoli ne contiene almeno uno. La
  prima esecuzione a regole li mascherava, quella finale no.
- Il **nome commerciale** della testata (campo «Brand Name» di NewsGuard) resta nel 2,1% degli articoli, in 17 testate.
  In alcune è quasi sempre presente: Council on Foreign Relations nell'82% degli articoli, Human Rights Watch nel 76%,
  Checks & Balances Project nel 100%.
- Restano anche **alias**: «Bulwark» nell'89% degli articoli di thebulwark.com, «Judicial Watch»/«Fitton» nel 64%,
  «Gingrich» nel 62%.
- L'1,6% di nome residuo citato nel paper si misura con lo stesso confronto che maschera, quindi non può vedere ciò che
  sfugge.
- I modelli leggono **solo il corpo**, non il titolo: il paper non lo dice.

**Effetto sui numeri: probabilmente piccolo.** Testo originale e testo anonimizzato differiscono di al massimo 0,3 punti.
Il problema è l'affermazione «every explicit trace removed» e il contributo 2.

**Cosa fare.**
- Ammorbidire il testo, dichiarare che i modelli leggono solo il corpo, riportare la misura indipendente (Brand Name).
  **T**
- Prova di robustezza: rifare il confronto principale senza le 17 testate con il nome rimasto. **C**, minuti.
- Facoltativo: aggiungere Brand Name e handle all'anonimizzazione e rifare tutto. **G**, circa 8 ore più tutta la
  pipeline.

## 🟠 6. «Conta la dimensione» poggia su un solo confronto

**Il problema.**
- L'unico effetto significativo della dimensione è Qwen 8B contro 0,6B, e i due vettori hanno dimensioni diverse (4.096
  contro 1.024 componenti): non «differiscono solo per dimensione», come scrive §5.1.
- Mistral 24B contro Qwen 8B non è significativo.
- ModernBERT congelato (150M) ha un MAE più basso di Qwen 0,6B: l'andamento non è monotono.

**Cosa fare.**
- Aggiungere Qwen3-Embedding-4B. **G**, circa 1 ora.
- Aggiungere un controllo a parità di dimensione: il vettore 8B proiettato a 1.024 componenti (PCA sui fold di
  addestramento). **C**
- Oppure scrivere solo «i modelli più grandi». **T**

## 🟠 7. Il confronto con l'addestramento (fine-tuning) è debole

**Il problema.**
- ModernBERT ha un solo seme e nessuna scelta di iperparametri sui fold interni. Epoche e learning rate vengono dal
  tentativo 16, scelti guardando le curve di test.
- Il §5.1 dice invece che gli iperparametri si scelgono dentro i fold di addestramento: per ModernBERT è falso.
- L'argomento della memorizzazione confronta grandezze diverse: un errore sugli articoli di addestramento (circa 4 punti)
  con un MAE sulle testate di test (17,5).
- «Il limite è la rappresentazione, non la memorizzazione» non segue da una sola coppia di modelli.

**Cosa fare.**
- Correggere le frasi e ammorbidire. **T**
- Facoltativo: 3 semi e una scelta degli iperparametri sui fold interni. **G**, molte ore.

## 🟠 8. La statistica si contraddice

**Il problema** (verificato).
- La famiglia della tabella principale ha **24** test, non 28: errore mio.
- Lo stesso confronto (Mistral − TF-IDF sulla ρ, con tutti gli articoli) è «non significativo» in §5.3 (Holm p 0,065) e
  «significativo per ogni k» in §5.4 (Holm p 0,005), perché la famiglia delle curve include «tutti gli articoli».
- «Significativo» è definito con Holm, ma nella scala (†) e nell'appendice C è non corretto.
- «Non cambia» richiede un test di equivalenza, non una differenza non significativa.
- Un revisore che ha visto le versioni precedenti (classificazione, F1, ρ) può leggere il MAE scelto ora come misura
  principale come un cambio di obiettivo dopo i risultati.

**Cosa fare** (**T**, più TOST **C**).
- Dichiarare un'ipotesi principale: Mistral contro TF-IDF sul MAE.
- Mettere ogni altro confronto in una sola famiglia (togliere «tutti» dalle curve).
- Usare una sola convenzione per «significativo».
- TOST per l'anonimizzazione.
- Giustificare il MAE in partenza: il bersaglio è il voto.

## 🟠 9. Molti modelli provati sugli stessi dati, senza dirlo

**Il problema.** Le stesse 226 testate sono state usate nei tentativi 16–19 e nei passi 6, 10, 11 e 12, con i risultati di
test visibili. In quei passi abbiamo provato:
- Gemma;
- DeBERTa;
- 14 regressori su 5 rappresentazioni;
- i meta-articoli;
- classificazione contro regressione;
- lo stiramento dei voti.

Il paper dice solo che penalità e strato si scelgono sui fold interni. La stabilità sulle cinque divisioni limita il
danno, ma la trasparenza manca.

**Cosa fare.**
- Un'appendice con tutto ciò che è stato provato su queste testate, e quali scelte precedono i risultati di test. **T**
- L'ideale sarebbe una conferma su testate nuove (valutate o raccolte dopo ottobre 2026): costoso.

## 🟠 10. Posizionamento: lavori mancanti e novità più stretta di come la scriviamo

**Il problema.** «All of them predict classes» vale solo dentro la nostra tabella. Lavori verificati:
- **Burdisso et al. 2024** (NAACL, già citato): stima un **grado continuo** di affidabilità, ma dal grafo dei link, non
  dal testo (ρ 0,80 con i giudizi dei giornalisti).
- **Kondrup et al., CrediBench** (KDD 2026, arXiv 2509.23340): credibilità dei domini **anche come regressione**, da grafo,
  testo e tempo.
- **Eldifrawi, Wang, Trabelsi 2026** (arXiv 2607.04560): credibilità del dominio **dai suoi articoli, senza conoscere prima
  il dominio**. È il concetto più vicino al nostro: va letto e citato.
- **Zhou et al. 2021:** la somiglianza tra siti (punto 4).
- **McElroy et al. 2025** (arXiv 2511.16416): qualità percepita con le etichette di Lin et al. Il revisore dice che la CV è
  per articoli, ma dall'abstract non risulta: da controllare.

Da controllare prima di citarli (indicati dal revisore, esistenza plausibile):
- Kiesel et al. 2019 (SemEval, testate separate);
- Potthast et al. 2018 (lo stile separa l'iperpartigianeria, meno le fake news);
- Gruppi et al. 2021 (reti di condivisione come segnale);
- Buckmann et al. 2025 (una testa lineare sugli stati interni batte la risposta del modello);
- Tang et al. 2024 (embedding degli LLM per la regressione).

**Cosa fare.** Citarli e riscrivere la novità come: solo testo, testate e proprietari separati, punteggio continuo, e
misura di quanto conta ogni protezione. Che un LLM congelato batta TF-IDF, da solo, è atteso. **T**

## 🟠 11. Un solo valutatore, proprietario

**Il problema.** Le etichette non si possono rilasciare, quindi la replica richiede di comprare NewsGuard. Esiste un
punteggio aperto e continuo che include NewsGuard: Lin et al. 2023, che citiamo, con 11.520 domini su GitHub.

**Cosa fare.** Rifare la tabella principale con il punteggio di Lin. **C**: i vettori ci sono già, cambiano solo le
etichette. Va prima controllato quante delle 226 testate copre. Risponde insieme alla critica sul valutatore unico e alla
riproducibilità.

## 🟠 12. Etica: frasi non vere e una licenza da chiarire

**Il problema.**
- «The released files contain no names of people» è falso: gli URL degli articoli contengono nomi. Il revisore ne conta
  circa 4.000 tra le righe delle previsioni, con nomi di personaggi pubblici e possibilmente di privati.
- «Names of journalists are masked before any model reads them» è contraddetto dai gradini della scala su testo
  originale.
- **Licenza:** rilasciamo modelli e previsioni addestrati sulle etichette NewsGuard, che seguono il voto con ρ 0,73 su 226
  domini con nome. Va verificato che la licenza NewsGuard lo permetta, e scritto esplicitamente. **Decisione di John.**
- La modalità pubblica del tool mostra previsioni per testata con nome, in contrasto con «never label an outlet without
  human review».
- Mancano una frase su robots.txt e sul limite di richieste (lo scraper li rispetta) e una frase su revisione etica o
  protezione dei dati.

**Cosa fare.** **T**, più la verifica della licenza.

## 🟠 13. Solo teste lineari (critica di John, fondata)

**Il problema.**
- Nel paper ogni rappresentazione è letta da una regressione lineare (ridge). Un revisore chiederà se un modello non
  lineare cambia le conclusioni: o in meglio per gli LLM, o in un confronto più equo per TF-IDF.
- La risposta l'abbiamo già in parte, ma non è nel paper. Il passo 10 ha provato 14 teste sui vettori di Qwen e Mistral:
  kernel ridge RBF, una piccola rete neurale, attenzione sugli articoli (multiple-instance learning), modelli per
  testata.
- Risultato del passo 10: nessun salto. Il «tetto» sui fold interni è circa 0,75 di ρ. La piccola rete neurale abbassa il
  MAE di Mistral da 15,7 a 14,8 (significativo sul voto predetto), e il vantaggio su TF-IDF non dipende dal regressore.
- **Due limiti:**
  - quei numeri usano i vecchi modelli Mistral, il voto «stirato» e 1.000 ricampionamenti, quindi vanno ricalcolati con
    il protocollo del paper;
  - **TF-IDF non ha mai avuto un modello non lineare**, quindi il confronto non lineare non è simmetrico.

**Cosa fare.**
- Un'appendice con le teste non lineari del passo 10 ricalcolate come il paper (voto predetto, 10.000 ricampionamenti,
  modelli attuali). **C**
- Una baseline non lineare su TF-IDF, per esempio gradient boosting o una piccola rete neurale su TF-IDF ridotto con SVD.
  **C**, circa un'ora.
- Una frase nel testo: la testa lineare è una scelta di semplicità, non un limite che cambia le conclusioni.

## 🟠 14. Il test ha la stessa distribuzione dei voti dell'addestramento (critica di John, fondata in parte)

**Che cosa non è.**
- Non è una fuga di informazione. I fold sono stratificati sul voto esatto, quindi addestramento e test hanno la stessa
  distribuzione (media del test circa 68 in ogni fold), ma nessuna informazione sulle singole testate passa dal training
  al test.
- Non avvantaggia i modelli rispetto alle baseline, anzi: la costante (mediana di addestramento) è quella che ne
  beneficia di più.

**Il problema vero: il numero assoluto è ottimistico per l'uso reale.**
- Il caso d'uso del paper sono le testate non ancora valutate, che probabilmente sono meno affidabili.
- Le 619 testate che non siamo riusciti a raccogliere sono per il 43% nel livello più basso, contro il 15% del dataset.
- I modelli sbagliano molto proprio lì: Mistral ha un MAE di 31 punti nel livello 0–39.

Ripesando l'errore per livello con quella distribuzione (calcolo fatto):

| | MAE sul dataset | con il mix delle testate non raccolte | con il mix di tutte le 1.293 testate politiche |
|---|---|---|---|
| Mistral-24B | 15,8 | 20,2 | 19,7 |
| TF-IDF | 17,8 | 22,9 | 22,4 |
| costante (mediana) | 22,1 | 34,9 | 32,6 |

Il vantaggio dei modelli sulla costante e su TF-IDF cresce, ma il MAE atteso nell'uso reale è circa 20 punti, non 16.
Lo stesso vale per la compressione verso la media di addestramento: con testate in media meno affidabili, la sovrastima
cresce.

**Cosa fare.**
- Riportare il MAE ripesato, con una frase su questo spostamento della distribuzione nei limiti e nella discussione
  dell'uso pratico. **C**, minuti.
- Facoltativo: ripetere il passo 16 con divisioni non stratificate, per mostrare che la stratificazione non cambia il
  confronto tra modelli. **C**, circa un'ora.

## 🟡 15. Correzioni minori (tutte verificate o verificabili)

- **Aggregazione (§5.3):** «1,5–3 punti» mescola le unità. Da articolo a testata nella Tabella 2→3 sono 2,0–3,1; il test
  confronta testate con 1 articolo e con tutti.
- **Appendice A:** «the ranking of the models is the same as with the MAE» è falso per l'AUC (Qwen > Mistral).
- **§5.4:** «il vantaggio è massimo con pochi articoli» vale per la ρ, non per il MAE di Qwen.
- **Qwen3-0,6B** non è solo «non migliore di TF-IDF»: sul MAE è significativamente peggiore in alcune curve.
- **§6:** «the trained models of Table 4» → i quattro modelli principali.
- **Data dell'estrazione NewsGuard:** 23/10/2026. Il 24% dei voti è aggiornato prima del 2025.
- **Scraper:** descrivere feed, sitemap e home, i più recenti prima, massimo 200, robots.txt, una richiesta al secondo.
- **Determinismo:** «deterministic» è troppo forte (il codice usa `warn_only`).
- **Tool:** l'intervallo mostrato è il bootstrap della media degli articoli, largo pochi punti, mentre l'errore reale è
  circa 16 punti. Serve un intervallo di previsione dai residui.
- **Pesi diversi:** in addestramento le testate pesano per numero di articoli, in valutazione tutte uguali; dirlo.
- **Legame di proprietà mancato:** spectator.co.uk e thespectator.com (voti 100 e 80, fold diversi) non sono nello stesso
  gruppo, ma sono la coppia più simile tra le 226. Controllo da fare.
- **Inversione nello split per articolo:** l'inversione TF-IDF/Qwen è fragile (Δρ 0,020 [0,000; 0,041], penalità al bordo
  della griglia). Meglio dire che il vantaggio di Qwen sul MAE sparisce.
- **«What matters is the similarity»:** è una correlazione su 31 testate, significativa solo per TF-IDF. Va scritto meno
  in termini causali.
- **«A property of the media ecosystem rather than of our sampling» (§4.5):** l'appendice D lo contraddice.
- **«The cheapest option»:** è falso, TF-IDF costa meno.
- **Uso per l'ordinamento:** la conclusione lo raccomanda, ma lì il vantaggio di Mistral non è significativo.
- **Metafora del DNA:** il DNA di un capello è proprio il modo classico di **identificare** una persona, l'opposto di
  «tratti, non identità». Da rivedere (decisione di John).
- **Abstract:** circa 400 parole, troppo lungo per molte riviste (150–250).
- **Appendice F (tool):** è documentazione software; meglio nel materiale supplementare.
- **Terminologia:** outlet/publisher/source mescolati; «paper» come nome di un protocollo nella Tabella 5.

---

## Controllato e a posto

Sono controlli fatti dai revisori sui dati e sul codice; i primi due e l'ultimo li ho ricontrollati anche io.

- Nessuna fuga in `linear_fold`: vocabolario TF-IDF e standardizzazione sono stimati sul solo addestramento, e i fold
  interni contengono solo testate di addestramento.
- Nessun gruppo di proprietà attraversa i fold, nemmeno con i gruppi calcolati su tutte le 1.293 testate.
- Nessun confondente temporale (voto ed errori non dipendono dalle date degli articoli).
- Le testate più simili tra loro non condividono il voto: un predittore «testata più vicina» fa peggio della costante.
- Il vantaggio di Mistral regge sulle testate piccole (10–39 articoli: 15,0 contro 17,1) e dopo la ricalibrazione.
- Le tabelle 3, 5, 7 e 8 e i numeri di §5.3–5.4 coincidono con i CSV, tranne «28».
- Lo strato 27 di Mistral era scelto a priori.

## Cosa consiglio, in ordine

1. **Prima di tutto, solo testo** (mezza giornata):
   - punti 2 e 8;
   - le correzioni di 4 e 5;
   - punti 6, 7, 10 e 12;
   - le minori del 15;
   - l'appendice del punto 9.

   Molti di questi errori li abbiamo introdotti noi e vanno tolti comunque.
2. **Analisi economiche, che rispondono alle critiche più forti:**
   - ricalibrazione annidata (punto 1, **C**);
   - sonda «solo nome» e zero-shot dal nome (punto 3, **G**, minuti);
   - gradino «sorelle senza copie» e split per articolo sulle stesse testate (punto 4, **C**);
   - punteggio di Lin et al. (punto 11, **C**, se la copertura è buona);
   - teste non lineari del passo 10 ricalcolate e una baseline non lineare su TF-IDF (punto 13, **C**);
   - MAE ripesato sulla distribuzione delle testate non raccolte (punto 14, **C**, minuti).
3. **Facoltative e più costose:**
   - Qwen3-Embedding-4B (punto 6, **G** circa 1 ora);
   - semi di ModernBERT (punto 7);
   - nuova anonimizzazione (punto 5);
   - testate nuove (punto 9).

**Decisioni tue:**
- riprendere la notorietà, cioè il punto 3, oggi la critica più forte;
- la ricalibrazione al posto dello stiramento, escluso in passato;
- la licenza NewsGuard per i modelli rilasciati;
- la metafora del DNA.
