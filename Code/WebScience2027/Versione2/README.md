# Versione 2 — più pulizia, training allargato

Stessa pipeline della prima versione (`../1-data-cleaning`, `../2-classificazione`), stesso scraping
(`../Dataset/NuovoScraping`, non si rifà), con tre cambiamenti decisi il 19 ottobre 2026, **prima** di vedere i
risultati:

1. **Una regola di pulizia in più** (`1-data-cleaning/01_pulizia_articoli`, passo 4c): via i resti dei contenuti
   incorporati — link (`http…`, `www…`, `pic.twitter.com/…`), firme dei tweet incorporati ("— Nome (@utente) data"),
   la scritta "View this post on Instagram", righe che erano solo un link. Il testo dei tweet citati resta.
   Espressioni regolari fisse, uguali per tutte le testate: nessun LLM, nessuna modifica a mano. Nella prima
   versione il classificatore a feature usava proprio queste parole (*pic twitter*, *https*, *www*) per $l_1$.
   Sul dataset della prima versione toglie lo 0,3% delle parole e tocca il 24% degli articoli $l_1$ (3% di $l_5$).
2. **Training allargato con il tetto a rotazione** (`1-data-cleaning/03_dataset_finale`):
   - il **test non cambia**: testate con almeno 20 articoli, al più 5 per gruppo (le "valutate", 304 nella prima
     versione), con gli stessi fold;
   - nel **training** entrano anche le altre testate dei gruppi di training (almeno 5 articoli): circa 860 testate
     invece di 304; non vanno mai nel test;
   - un gruppo non pesa mai più di 5 testate: TF-IDF ed embedding usano un **peso** per testata
     ($\min(1, 5/n)$), i transformer pescano a ogni epoca al più 5 testate per gruppo, diverse a ogni epoca.
3. **Una combinazione fissata a priori** (`2-classificazione/03_confronto`): media 0,5/0,5 delle probabilità di
   TF-IDF (`feature_tuned`) e dell'ensemble di ModernBERT-base.

Controlli: `feature_tuned_tetto5` e `modernbert_base_1024_ep2_tetto5_s42` si addestrano solo sulle testate valutate,
come la prima versione, per misurare quanto conta allargare il training.

## Cartelle

```
1-data-cleaning/    01 pulizia (+ passo 4c), 02 reti, 03 dataset finale (testate valutate e di solo training)
2-classificazione/  01 feature, 02 transformer, 04 embedding, 03 confronto, coda.py (la coda degli esperimenti)
Dataset/            derived/ e versioni/ di questa versione (fuori da git); lo scraping sta in ../Dataset
Results/            notebook eseguiti, log, predizioni, metriche
esegui_tutto.py     pulizia 01 -> 02 -> 03 e poi coda.py, con tempi massimi, tentativi e ripresa
stato.sh            sul server: a che punto è
controlla_server.sh dal PC: ./controlla_server.sh [log|risultati]
```

## Sul server

Con `~/John/WebScience2027/Dataset/` (lo scraping con l'HTML grezzo, e `newsguard.csv`) e questa cartella in
`~/John/WebScience2027/Versione2`:

```bash
cd ~/John/WebScience2027/Versione2
screen -dmS v2 bash -c 'until ~/John/johnVenv/bin/python esegui_tutto.py; do sleep 120; done'
./stato.sh
```

Il server è condiviso: i batch dei transformer sono piccoli (4 per il base, 2 per il large, con lo stesso batch
effettivo 32 grazie all'accumulo del gradiente); se la memoria non basta, il notebook dimezza il batch da solo e
riparte dal fold. La coda aspetta fino a 2 ore che si liberi la memoria indicata, poi prova comunque.
