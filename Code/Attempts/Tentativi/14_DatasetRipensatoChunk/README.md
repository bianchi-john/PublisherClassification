# Tentativo 14: dataset ripensato e ModernBERT a pezzi

> **Tentativo autonomo** con due notebook:
> - `01_preparazione_dataset.ipynb`: dal grezzo al dataset, con regole semplici;
> - `02_classificazione.ipynb`: feature, ModernBERT troncato e ModernBERT a pezzi, l'idea di Fabio.
>
> Le funzioni lunghe stanno in `funzioni.py`. I dati sono in `Dataset/`, fuori da git: sul PC sono link ai tentativi
> 12 e 13, sul server sono hardlink.

**Storia.** La prima versione del tentativo 14 (commit `a0c2506`, 26 ottobre) usava il protocollo del tentativo 13,
troppo complicato da spiegare in un paper e che sembrava costruito a posteriori:
- il filtro solo politici arrivava a metà;
- c'erano testate "valutate" e testate "solo training";
- si valutavano al più 5 testate per gruppo, e solo quelle con almeno 20 articoli;
- le testate avevano pesi nel training.

La sera del 26 ottobre il dataset è stato ripensato con quattro regole, decise prima di vedere i risultati:

1. **Solo testate politiche** fin dall'inizio: per NewsGuard 2026 hanno un solo argomento, "Political news or
   commentary", e un punteggio.
2. **Pulizia** come nei tentativi 06–13:
   - articoli: più di 100 parole, niente doppioni, solo inglese, dal 2025;
   - testo: tolte le frasi ripetute dal sito, il titolo e la data in testa, link e tweet;
   - tolti i quasi doppioni;
   - almeno 100 parole dopo la pulizia.
3. **Gruppi** di testate parenti, ciascuno tutto in un fold. Due testate sono parenti se:
   - si copiano gli articoli;
   - hanno le stesse tracce nelle pagine (codici, titolare legale, dominio, autori);
   - per NewsGuard 2022 o 2026 hanno lo stesso Owner o lo stesso Parent Domain.

   Un testo copiato tra gruppi diversi resta solo nella copia più vecchia.
4. **Almeno 10 articoli** per testata. Poi 5 fold stratificati per livello e per gruppo. **Ogni testata è valutata**
   (una volta nel test, quattro nel training), e tutte pesano uguale.

Nel training di ModernBERT, a ogni epoca entrano 20 articoli a caso per testata (40 pezzi nella versione a pezzi),
senza tetto per gruppo. Il classificatore a feature usa tutti gli articoli, con le classi bilanciate.

## Risultati

Il dataset: 323 testate politiche, 42.219 articoli, 223 gruppi.

| Modello | F1 (5 livelli) | Tolerant ±1 | Binario |
|---|---|---|---|
| Feature (voto) | **0,403** | 0,768 | 0,703 |
| ModernBERT troncato | 0,272 | 0,666 | 0,619 |
| ModernBERT a pezzi (7 aggregazioni) | 0,278–0,303 | 0,709–0,746 | 0,610–0,625 |

- Il feature batte tutti i ModernBERT in modo significativo: da −0,10 a −0,13 di F1, con intervalli tutti sotto lo zero.
- I pezzi aiutano poco rispetto al troncato (da +0,01 a +0,03), in modo non significativo.

ModernBERT soffre i **gruppi grandi**, che ora contano per intero. 67 testate stanno in 6 gruppi con più di 5 testate,
quasi sempre di un solo livello, e quando un gruppo grande è nel test ModernBERT lo sbaglia tutto insieme: nel fold 1
il troncato fa 0,12.

## Esecuzione

Sul server, in `~/John/Tentativo14`, gira `esegui.sh` (screen `t14`):
- prima una prova breve dei due notebook (100 testate);
- poi la versione completa: il notebook 1 in circa 10 minuti, il notebook 2 in circa 1 ora e mezza sulla GPU.

Dal PC, `./scarica.sh` mostra lo stato e, a lavoro finito, scarica i notebook eseguiti e i risultati.

Prova veloce del notebook 1 sul PC, in circa un minuto:
`papermill -k irgood-venv -p N_JOBS 8 -p TESTATE 150 01_preparazione_dataset.ipynb prova.ipynb`.
