# Tentativo 09 — Cambio di paradigma: testo, link e pagina

> **Tentativo autonomo**: `0-scraping` → `1-data-cleaning` → `2-segnali` → `3-classificazione`, dati in `Dataset/`
> (fuori da git), risultati in `Results/`. `0-scraping` e `1-data-cleaning` sono copie di quelli del tentativo 06: lo
> scraping non è stato rifatto, i passi 01–02 della pulizia non sono stati rieseguiti (le loro uscite, deterministiche,
> sono riusate); il passo 03 è stato rieseguito con i gruppi nuovi. Il percorso di tutti i tentativi è nel README di
> `Code/WebScience2027`.

**La domanda.** Dai tentativi 05–08: sul solo testo un TF-IDF è vicino al massimo (F1 ≈ 0,40) e i transformer non
aggiungono nulla, perché il segnale è lessicale e segue il genere della testata. Qui si cambia paradigma: non solo
*che cosa scrive* una testata, ma **chi cita** e **com'è fatto il suo sito**. Quanto vale ogni tipo di informazione,
e quanto si arriva in alto combinandoli, con una valutazione che tiene conto delle reti?

## Il disegno (fissato il 22 ottobre 2026, prima dei risultati)

- **Dataset**: quello della Versione 2 (pulizia, testate di solo training con il loro peso), con i **gruppi** = reti
  trovate ∪ stesso owner NewsGuard ∪ **stesso Parent Domain** NewsGuard. Il Parent Domain copre le reti che
  NewsGuard non attribuisce a un owner (Natural News: owner "Unknown", Parent Domain `naturalnews.com`); nelle
  Versioni 1–4 due testate di Natural News stavano nel test in fold diversi dal resto della rete. Test: **301
  testate** (almeno 20 articoli, al più 5 per gruppo), 5 fold per gruppo.
- **Ingressi** (solo cose osservabili per qualunque sito; dall'HTML grezzo di ogni articolo, salvato dallo scraping):
  - **testo**: la media dei vettori TF-IDF degli articoli della testata;
  - **link**: i domini citati nel testo degli articoli (escluso il sito stesso e i siti dello stesso gruppo), le
    quote per tipo (social, video, video alternativi, istituzionali, Wikipedia, testate NewsGuard) e l'affidabilità
    dei siti citati, calcolata **solo con le etichette delle testate di training** di ogni fold;
  - **pagina**: servizi di terze parti per categoria (lista di Disconnect), reti pubblicitarie, link alle pagine di
    trasparenza (chi siamo, rettifiche, etica, redazione, proprietà, pubblicità, donazioni, newsletter), firma, data,
    dati strutturati, opinione dichiarata, paywall, commenti, cookie, contenuti incorporati, dimensioni.
- **Non sono ingressi** i criteri e il punteggio NewsGuard: il punteggio è esattamente la somma pesata dei 9 criteri
  per tutte le 9.222 testate NewsGuard (un criterio che NewsGuard non applica, per esempio la pubblicità in un sito
  senza pubblicità, vale i suoi punti), e per un sito nuovo non si avrebbero. Sono solo **obiettivi**: per il modello
  "alla NewsGuard" l'etichetta di un criterio è "la testata prende i punti" (sì, oppure non applicato).
  Non si usano nemmeno Wikipedia, Twitter, traffico web e forma dell'URL (le famiglie di Baly et al., EMNLP 2018).
- **Modelli** (regressione logistica ovunque, pesi per testata come nelle Versioni 2–4, C scelto in una validazione
  interna per gruppi sul solo training):
  1. un modello per famiglia;
  2. la **combinazione** (stacking sulle probabilità fuori fold delle famiglie), anche a coppie;
  3. **"alla NewsGuard"**: 9 classificatori sì/no, uno per criterio, poi il punteggio con i pesi pubblici di NewsGuard
     (22, 18, 12,5, 12,5, 10, 7,5, 7,5, 5, 5) e il livello.
- **Riferimento**: il TF-IDF per articolo delle Versioni 1–4, ricalcolato sullo stesso test.
- **ModernBERT come famiglia del testo** (aggiunto la sera del 22 ottobre, prima dei suoi risultati): ModernBERT-base
  sul solo testo, ricetta delle Versioni 2–4 (1024 token, 2 epoche fisse); per entrare nello stacking servono le sue
  probabilità fuori fold, quindi per ogni fold 3 addestramenti interni (le stesse parti di `02_modelli`) e uno
  finale; la testata è la media delle probabilità di al più 100 suoi articoli. Semi 42, 1, 2, mediati. Combinazioni:
  ModernBERT al posto del TF-IDF, accanto al TF-IDF, e con tutte le famiglie.
- **Metriche**: F1 macro a 5 livelli e binario per testata, bootstrap per gruppo.

## File

```
1-data-cleaning/03_dataset_finale.ipynb   gruppi reti ∪ owner ∪ Parent Domain, testate valutate, fold
2-segnali/comune.py                        percorsi, dominio registrato (tldextract, senza rete), categorie di Disconnect
2-segnali/01_estrai_segnali.ipynb          dall'HTML di ogni articolo: link citati e segnali della pagina
2-segnali/02_segnali_testata.ipynb         una riga per testata (senza etichette) e "c'è segnale?" per livello
3-classificazione/comune.py                dati, pesi, metriche (dalla Versione 2), criteri e pesi NewsGuard
3-classificazione/01_riferimento_tfidf.ipynb  il TF-IDF per articolo delle Versioni 1–4, sullo stesso test
3-classificazione/02_modelli.ipynb         famiglie, combinazioni, modello "alla NewsGuard"
3-classificazione/03_confronto.ipynb       intervalli, differenze con il riferimento, segnali più forti, criteri
3-classificazione/04_modernbert.ipynb      ModernBERT come famiglia del testo (GPU): 4 addestramenti per fold, un seme
3-classificazione/05_combinazione_bert.ipynb  le combinazioni con ModernBERT (media dei semi finiti)
esegui_tutto.sh                            la catena in ordine (salta i passi già fatti)
coda_notte.sh                              ModernBERT semi 42, 1, 2 sulla GPU, e dopo ogni seme 05 + 03
stato.sh, controlla_server.sh              a che punto siamo (sul server; dal PC)
```

## Rieseguire

Tutto gira in locale (CPU): `./esegui_tutto.sh ~/.cache/irgood-venv/bin/python irgood-venv` dalla cartella del
tentativo (serve `Dataset/` con lo scraping e l'HTML grezzo, `newsguard.csv`, e le uscite della pulizia 01–02 in
`Dataset/derived` e `Results/1-data-cleaning`). L'estrazione dall'HTML richiede circa un'ora e un quarto con 6
processi; il resto pochi minuti, tranne il riferimento TF-IDF.

## Sul server (22 ottobre)

In `~/John/Tentativo09` (solo quello che serve: niente HTML grezzo). Due screen:
- `t09`: `N_JOBS=5 ./esegui_tutto.sh ~/John/johnVenv/bin/python johnvenv` (CPU, al più 20 core, `nice`), dalle 16:55;
- `t09_notte`: `./coda_notte.sh` (GPU, un addestramento alla volta, circa 4 ore per seme; prima di ogni addestramento
  aspetta che sulla GPU condivisa ci siano almeno 6,5 GB liberi), dalle 17:41.

Per controllare dal PC: `./controlla_server.sh` (lo stato) e `./controlla_server.sh risultati` (le tabelle del confronto).

## Risultati (23 ottobre)

Per testata, 301 testate valutate, intervalli al 95% con bootstrap per gruppo (`Results/3-classificazione/confronto/`).
Maggioranza: F1 0,096, binario 0,561; a caso: F1 0,176.

| modello | F1 a 5 livelli | Δ dal riferimento [IC 95%] | binario |
|---|---|---|---|
| riferimento: TF-IDF per articolo | 0,340 | — | 0,684 |
| testo (TF-IDF per testata) | 0,339 | −0,00 [−0,05; +0,04] | 0,684 |
| ModernBERT (media di 3 semi) | 0,325 | −0,02 [−0,07; +0,04] | 0,708 |
| link | 0,247 | −0,09 [−0,15; −0,03] | 0,641 |
| pagina | 0,241 | −0,10 [−0,16; −0,03] | 0,615 |
| combinato testo + pagina | 0,354 | +0,01 [−0,06; +0,09] | 0,711 |
| combinato tutte | 0,347 | +0,01 [−0,07; +0,09] | 0,688 |
| combinato tutte + ModernBERT | 0,318 | −0,02 [−0,10; +0,05] | 0,714 |
| alla NewsGuard | 0,319 | −0,02 [−0,10; +0,06] | 0,718 |

- **Nessun modello batte il testo** in modo significativo: tutto arriva a F1 circa 0,34 e binario circa 0,70. Link e
  pagina hanno segnale (molto sopra il caso) ma meno del testo, e non gli aggiungono nulla di dimostrabile.
- **Criteri** (accuratezza bilanciata del modello alla NewsGuard): informazione responsabile 0,77, titoli 0,72,
  rettifiche 0,70, contenuti falsi 0,67, firma 0,63, chi comanda 0,62, notizie/opinioni 0,61, proprietà 0,56,
  pubblicità 0,54. Si stimano quelli di contenuto, non quelli di pagina.
- **Segnali più forti**: l5 cita bbc.com, apple.news, siti governativi, e ha redazione, rettifiche, paywall,
  newsletter; l1 cita t.me, gab.com, rt-english.com, pravda.ru e incorpora video da piattaforme alternative.
- **Variabilità dei fold**: sulle 284 testate comuni con la Versione 2 il riferimento passa da 0,357 a 0,340 solo per
  i gruppi nuovi (Parent Domain), e 118 di quelle testate cambiano previsione. Con 301 testate in 218 gruppi non si
  distinguono differenze sotto i 6–7 punti.
