# 1 — Data cleaning

> **Versione 1** (18 ottobre). Le versioni successive sono i tentativi 06–08; il percorso completo è nel README di `Code/WebScience2027`.

Dal dataset dello scraping al dataset finale. Pulizia **rigorosa ma non eccessiva**: si toglie ciò che non è testo
dell'articolo o che metterebbe lo stesso testo nel training e nel test; ciò che c'è in uno scenario reale
(citazioni, pezzi ripresi dentro la stessa rete) resta. Cieco alle etichette NewsGuard: tornano solo in `03`; il
campo Owner serve a verificare le reti in `02` e a formare i gruppi in `03`. Gli eseguiti stanno in
`../Results/1-data-cleaning/esecuzioni/`.

| Notebook | |
|---|---|
| `01_pulizia_articoli.ipynb` | via i doppioni esatti, gli articoli prima del 2025, le frasi ripetute in molti articoli della stessa testata (firme, promozioni), titolo e timbro all'inizio del testo, i doppioni sul testo pulito, i testi sotto le 100 parole; si trovano (non si tolgono) le copie tra testate (MinHash) |
| `02_reti.ipynb` | testate parenti da cinque segnali: copie e autori in comune (se sistematici, ≥10% degli articoli), codici di account nell'HTML (AdSense, Analytics, Tag Manager, Facebook, Chartbeat…), titolare legale (stessa privacy policy su un altro dominio), dominio e redirect; verifica con l'Owner (B-cubed) |
| `03_dataset_finale.ipynb` | gruppi = reti ∪ stesso owner; copie tra gruppi diversi (si tiene la più vecchia); testate con almeno 20 articoli; al massimo 5 testate per gruppo; etichette; fold per gruppo e per testata |
| `comune.py` | percorsi e funzioni comuni |

Output: `../Dataset/derived/01_articoli_puliti.csv`, `01_copie_tra_testate.csv`; `../Results/1-data-cleaning/02_reti.csv`,
`02_collegamenti.csv`, `02_verifica_owner.csv`, `03_passi.csv`, `03_composizione.csv`;
`../Dataset/versioni/dataset_finale.csv`, `fold_finale.csv` (304 testate, 44.029 articoli, 220 gruppi).
