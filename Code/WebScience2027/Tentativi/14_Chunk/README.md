# Tentativo 14 — ModernBERT a pezzi (chunk), in un solo notebook

> **Tentativo autonomo**, tutto in `chunk.ipynb` (le funzioni lunghe in `funzioni.py`): dai dati grezzi dello scraping
> ai risultati. Dati in `Dataset/` (fuori da git, sul server), risultati e notebook eseguito in `Results/`.

**La domanda** (proposta di Fabio, 26 ottobre): ModernBERT legge al più 1024 token e circa il 40% degli articoli è più
lungo. Spezzando ogni articolo in pezzi, valutando ogni pezzo da solo e mettendo insieme i voti, il transformer legge
tutto il testo: va meglio del troncato, e del classificatore a feature? E quanto testo resta fuori oggi?

## I due notebook finali (26 ottobre, per la presentazione)

1. **`01_preparazione_dataset.ipynb`**: il racconto dai dati grezzi al dataset, passo per passo, con tabelle e grafici:
   - NewsGuard e scraping, poi la pulizia degli articoli e del testo;
   - copie tra testate e reti di siti, poi il filtro solo politici;
   - i gruppi costruiti nel notebook: nostre reti + owner + Parent Domain, con i grafici di confronto reti ↔ Parent Domain e alcuni gruppi disegnati;
   - testate valutate e fold, il dataset finale per livello.

   Torna identico al tentativo 13 (✓ su testate, valutate, etichette e fold). Salva `Results/dataset/articoli.csv.gz` e
   `testate.csv` (fuori da git: etichette NewsGuard).
2. **`02_classificazione.ipynb`**:
   - quanto testo resta fuori dai 1024 token e come si spezza un articolo;
   - i tre modelli (feature, ModernBERT troncato, ModernBERT a pezzi) e i sette modi di mettere insieme i pezzi;
   - il confronto con intervalli da bootstrap per gruppo e le conclusioni.

   Le previsioni dei modelli sono quelle del run completo di `chunk.ipynb` (stesso dataset, controllato nel notebook),
   salvate in `Results/cache/` (fuori da git); senza cache il notebook le ricalcola.

Entrambi eseguiti sul server (`~/John/Tentativo14`, circa 11 + 1 minuti) e copiati qui con l'output delle celle;
grafici in PNG in `Results/dataset/grafici/` e `Results/classificazione/grafici/`. Per una prova veloce sul PC:
`papermill -k irgood-venv -p N_JOBS 8 -p TESTATE 800 01_preparazione_dataset.ipynb prova.ipynb` (circa 4 minuti).

**Risultato**:

| Modello | F1 |
|---|---|
| feature (voto) | 0,390 |
| ModernBERT troncato | 0,326 |
| ModernBERT a pezzi | 0,38–0,40 (+0,05–0,08 sul troncato, significativo in 5 modi su 7) |

I pezzi pareggiano il feature ma non lo battono. Il guadagno viene soprattutto dall'addestrare sui pezzi: con solo il primo pezzo in test si ha già 0,376.

## Il notebook completo `chunk.ipynb`, sezione per sezione

1. **Dati grezzi**: i testi dei tre giri di scraping (ottobre + terzo giro), più di 100 parole, niente doppioni, inglese.
2. **Pulizia** (regole dei tentativi 06–13): copie identiche, copie tra testate (MinHash), dal 2025, frasi ripetute
   dalla testata, titolo e data in testa, link e tweet, doppioni nella testata, almeno 100 parole.
3. **Reti di siti** (copie, codici di account, titolare legale, dominio/redirect, autori).
4. **Dataset solo politici** con le etichette NewsGuard 2026: gruppi reti ∪ owner ∪ Parent Domain, copie tra gruppi,
   testate valutate (almeno 20 articoli, al più 5 per gruppo), solo training con almeno 10 articoli, 5 fold per gruppo;
   il notebook controlla che torni uguale al tentativo 13 (323 testate, 254 valutate).
5. **Quanto testo resta fuori** dai 1024 token, per livello.
6. **Classificatore a feature** (TF-IDF + regressione logistica, C in validazione interna): voto e media per testata.
7. **ModernBERT troncato**: i primi 1024 token; 2 epoche, a ogni epoca al più 5 testate per gruppo e 20 articoli per
   testata, seme 42.
8. **ModernBERT a pezzi**: pezzi da 512 token con 64 di sovrapposizione, addestrato e valutato sui pezzi, stesso numero
   di token per testata a ogni epoca (40 pezzi); sette modi di mettere insieme i pezzi: maggioranza, media, media
   geometrica, media pesata per lunghezza, pezzo più sicuro, solo il primo pezzo, tutti i pezzi della testata insieme.
9. **Confronto**: F1 macro a 5 livelli, accuratezza, tolerant ±1 e ½, binario e sua F1, intervalli al 95% da un
   bootstrap per gruppo, differenze con il feature e con il troncato; F1 per fold; matrici di confusione.

Ogni passo lungo salva il risultato in `Results/cache/`: rilanciando il notebook si riparte da lì.

## Sul server

`~/John/Tentativo14`: dati grezzi collegati (hardlink) dalla copia del tentativo 12; `esegui.sh` (screen `t14`) esegue
il notebook con papermill e salva **`Results/chunk_eseguito.ipynb`**, con l'output di tutte le celle (fino a 3
tentativi). Prova breve (`PROVA = True`: 120 testate, un fold, un'epoca) superata il 26 ottobre:
`Results/prova/chunk_prova.ipynb`. Dal PC: `./controlla_server.sh`.

Dalla prova (solo un assaggio, 120 testate): tra il 35% e il 43% degli articoli supera i 1024 token, e si perde il 31%
del testo; con pezzi da 512 un articolo ne ha in mediana 2 (in media 3).
