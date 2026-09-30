# HANDOFF — Revisione paper "From Articles to Publishers" per ECIR 2027 IR for Good

> Documento di contesto per continuare il lavoro dentro VS Code.
> Leggilo per intero prima di toccare qualsiasi file.

---

## 1. Obiettivo e stato attuale

Stiamo **rivedendo e riscrivendo** il paper *"From Articles to Publishers: Aggregating
Language Model Predictions for News Source Reliability Inference"* (autori: J. Bianchi,
M. Pratelli, F. Pinelli, M. Petrocchi).

- Il paper è stato **respinto da IEEE TCSS** (reject, no resubmit invited).
- Lo stiamo riproponendo, rivisto, a un'altra sede (vedi §2).
- Il vecchio sorgente LaTeX (IEEEtran) è in `Paper/OldPaperRejected/main.tex`.
- Il nuovo paper va costruito in `Paper/NewIRForGood/` **in template LNCS**.

**Filosofia di revisione:** il punto di forza del paper è la *narrazione / qualità di
scrittura*. Va preservata il più possibile. Le correzioni richieste dai revisori sono
in gran parte **additive** (analisi/esperimenti in più), non riscritture. NON banalizzare
o "sovravvendere" la novità; NON gonfiare i claim.

---

## 2. Target di submission

**Primario — ECIR 2027, track "IR for Good"** (core track, atti Springer LNCS):
- Deadline paper: **5 ottobre 2026** (AoE). Nessun abstract di pre-registrazione richiesto.
- Notifica: 7 dicembre 2026. Camera-ready: 10 gennaio 2027.
- Formato: **LNCS**, da 6 a **12 pagine + riferimenti illimitati** (appendici e la sezione
  "Theory of Change" contano nel limite). Doppio cieco.
- **Requisito obbligatorio: sezione "Theory of Change"** (vedi §9).

**Fallback — WebSci 2027** (ACM, deadline 15 dic 2026, ~10 pp incl. refs). Solo se ECIR
respinge. NB: WebSci è ACM sigconf, template diverso → NON è quello che usiamo ora.

**Dopo (se serve):** rivista Q2 veloce (IEEE Access / PeerJ CS / J. Computational Social
Science). Vincolo personale: PhD da chiudere entro dicembre 2027, quindi conta avere la
pubblicazione, non necessariamente in Q1.

---

## 3. Perché è stato respinto (sintesi delle due review TCSS)

**Reviewer 1 (Poor):**
1. Novità metodologica limitata (fine-tuning standard + majority voting).
2. Il guadagno dell'aggregazione non è misurato in modo pulito (0.60 article vs 0.69
   publisher cambia unità *e* pesi insieme).
3. L'analisi del bias politico va **controllata per il livello di reliability** (confound).
4. Il chi-quadro tratta osservazioni raggruppate come indipendenti → **pseudo-replicazione**
   (leaning e reliability sono attributi del publisher, condivisi da tutti i suoi articoli).
5. Non escluse **firme del publisher / artefatti di scraping** (manca text cleaning /
   masking degli identificatori / ablation su testo sanificato).
6. Possibile **selection bias** nello scraping (439/669 publisher; tassi di successo diversi
   per livello).
7. Confronto sulla dimensione del modello non controllato (architettura, fine-tuning,
   schedule, context length variano insieme).
8. Mancano stime di incertezza (CI, bootstrap, test appaiati) sui confronti tra modelli.

**Reviewer 2 (Fair):**
1. La riformulazione publisher-level non convince (resta ancorata all'article-level).
2. **Mancano baseline comparative** (LLM generalista zero-shot es. Qwen; e/o un modello
   dedicato all'article-level reliability). ← ora coperto in parte, vedi §5.
3. "Large-scale" non giustificato: solo 439 publisher.

---

## 4. Piano di revisione (cosa fare nel nuovo paper)

**Fix statistici (economici, additivi — i revisori hanno ragione):**
- Chi-quadro corretto per il clustering: aggregare a livello publisher **oppure** modello a
  effetti misti `errore ~ leaning + (1|publisher)` / standard error clusterizzati.
- Bias politico **controllato per reliability level** (`errore ~ leaning * reliability`).
- **Intervalli di confidenza (bootstrap)** e test appaiati su tutti i confronti tra modelli.
- **Curva di apprendimento dell'aggregazione**: accuratezza publisher in funzione di k
  articoli votati (k = 1, 3, 5, 10, 20...), con k=1 come baseline corretta → isola l'effetto
  dell'aggregazione (risponde a R1.2). Quasi gratis: si ri-aggregano le predizioni esistenti.
- **Controllo selection-bias scraping**: confrontare i 439 scaricati vs i ~230 persi su
  score/tier/leaning (test KS). Poche righe.

**Esperimenti aggiuntivi:**
- **Baseline zero-shot LLM** → FATTO (vedi §5).
- (Per la revisione, se c'è tempo) **ablation su testo sanificato** (masking degli
  identificatori di fonte) applicata a tutti i sistemi → risponde a R1.5.

**Ridimensionamento claim:**
- Ammorbidire "model scale is not the bottleneck" (R1.7).
- Togliere "large-scale"; riposizionare il contributo come *benchmark publisher-disjoint
  realistico* (R2.3).
- Riformulare la novità: protocollo publisher-disjoint deployment-realistic + evidenza che
  l'aggregazione stabilizza il segnale + analisi del bias fatta correttamente.

---

## 5. Baseline zero-shot GPT-5-mini — COMPLETATO

**Cosa:** classificazione zero-shot dei 19.476 articoli con **GPT-5-mini** (Azure OpenAI),
poi stesse metriche e stessa aggregazione publisher-level (majority vote) dei modelli
fine-tuned. Serve a rispondere a R2.2 in modo *equo e non contestabile*.

**Come (per la difendibilità nel paper):**
- Stesso input dei modelli fine-tuned: **solo la colonna `text`**, nessun metadato
  (no url/domain/authors/title). Run principale con **testo intero** (CAP=None) = "best shot"
  dell'LLM (companion opzionale a 256 token = apples-to-apples con gli encoder).
- Prompt con la **rubrica NewsGuard completa** (9 criteri, 5 di credibilità + 4 di
  trasparenza, somma 100) e i 5 bin con i range di punteggio (l1 0-39 … l5 100). Dare la
  rubrica è l'equivalente zero-shot della supervisione: NON è "aiutarlo troppo", ometterla
  sarebbe lo strawman.
- Output: singola cifra 1-5, mappata a 0-4 per allinearsi alla colonna `label`.
- Precedente peer-reviewed a supporto: il paper Reddit/Sci Reports (accettato) usa lo stesso
  schema di baseline zero-shot (GPT-4.1 mini, label-only, temp 0, seed 42).

**Risultati (19.469/19.476 articoli; 7 mancanti = articoli lunghissimi, trascurabili):**

| Livello | Metrica | GPT-5-mini zero-shot | Fine-tuned (migliore) |
|---|---|---|---|
| Article | Accuracy | **0.245** | ~0.60 |
| Article | macro-F1 | 0.200 | ~0.55 (RoBERTa) |
| Article | Tol@0.5 / Tol@1.0 | 0.429 / 0.612 | ~0.72 / ~0.84 |
| Publisher | Accuracy | **0.239** | **0.69 (RoBERTa)** |
| Publisher | macro-F1 | 0.179 | — |
| Publisher | Tol@1.0 | 0.547 | 0.88 (Llama) |

**Interpretazioni chiave (per la sezione risultati):**
- GPT zero-shot è al livello del caso; i modelli fine-tuned lo triplicano.
- GPT **collassa sul centro**: ~47% delle predizioni in l3, quasi zero l5 → senza i dati
  NewsGuard non sa calibrare la scala a 5 bin.
- **L'aggregazione NON salva GPT** (publisher ≈ article), mentre migliora i fine-tuned
  (0.60→0.69): il voto di maggioranza aiuta uno scorer calibrato, non uno zero-shot.

**Caveat da dichiarare:** è stato usato *mini*, non il flagship. Il gap è enorme, ma
conviene aggiungere una nota di scelta (costo/riproducibilità) o far girare il flagship su
un campione (~500 articoli) per blindare l'obiezione "avete provato solo mini".

**File dei risultati:**
- `Results/predictions/zeroshot/preds_gpt5mini_capNone.csv` — colonne: `idx, domain, label, pred_1_5`
- `Results/predictions/zeroshot/preds_gpt5mini_enriched.csv` — url + tutti i metadati (no `text`) + `pred_1_5` + `pred_0_4`
  (`pred_0_4` = `pred_1_5 - 1`, allineato a `label`)
- Chiave di join: `idx` = posizione di riga del dataset caricato con `reset_index(drop=True)`.
  L'identificatore leggibile dell'articolo è la **`url`**.
- Nota: la API key è in `NewsGuard/Other/key.txt` (NON committare; aggiungere a .gitignore).

---

## 6. Struttura del progetto (dati e codice)

- `Dataset/CleannedDataset/newDataframeWithOldCleanedPoliticsVersion.csv` — dataset finale
  (19.476 articoli). Colonne: url, title, text, authors, domain, score, country, language,
  topics, paywall, opinion_advocacy, label (0-4 = l1-l5).
- **Struttura riorganizzata il 30/09/2026** in stadi sequenziali (vedi `README.md`):
  - `0-scraping/` — `0 - NgScraperPol.ipynb` (scraping), `1 - Data preparation.ipynb`
  - `1-classification/` — `2 - …Separate.ipynb` (BERT/RoBERTa/Llama, 5-fold
    publisher-disjoint, `text` only, max_length 256), `3 - …mistral22B.ipynb`
    (Mistral, context 1024), `7 - Get prediction scores.ipynb` (probabilità per
    classe, GPU), `8 - ChatGPT Run.ipynb` (baseline zero-shot, §5)
  - `2-analysis/` — `paper_results.ipynb`: **un unico notebook** che rigenera tutte
    le figure e tabelle del paper, sopra il pacchetto `irgood/`. Generato da
    `build_notebook.py`. Vedi `2-analysis/FINDINGS.md`.
  - I notebook 4, 5 e 6 (Plotter, Political Error Analysis, Domain impact) sono
    stati **eliminati**: sostituiti da `2-analysis/`. Recuperabili dalla storia
    git (commit `cfa9682`).
- Risultati: `Results/predictions/finetuned/{bert,roberta,llama}.csv` +
  `mistral/24B_results.csv` (e i corrispondenti `*_enriched.csv` con
  `Article_IDs`/`URLs`/`Domains`); `Results/predictions/zeroshot/` per GPT;
  `Results/{figures,tables,derived}` output rigenerato; `Results/_archive/` il
  vecchio materiale.
- Paper vecchio: `Paper/OldPaperRejected/` (main.tex IEEEtran, bibliography.bib, img/,
  IEEEtran.cls/.bst, supplementary.tex).
- Paper nuovo (da creare): `Paper/NewIRForGood/`.

---

## 7. Numeri chiave dei modelli fine-tuned (dal paper vecchio)

- Dataset: 19.476 articoli, 439 publisher scaricati (669 selezionati), 5 livelli.
  Bin: l1 0-39, l2 40-59, l3 60-74, l4 75-99, l5 100.
- Article-level: acc ~0.60 (Mistral 0.61 max). macro-F1: RoBERTa 0.55, Mistral 0.54,
  Llama 0.54, BERT 0.52. Tol@0.5 ~0.72; Tol@1.0: RoBERTa 0.84, altri 0.82-0.83.
- Publisher-level: **RoBERTa 0.69**, Mistral 0.68. Tol@1.0: Llama 0.88, RoBERTa 0.87,
  Mistral 0.86.
- Bias politico (Cramér's V): BERT 0.294, RoBERTa 0.270, Mistral 0.263, Llama 0.249
  (tutti p < 10^-250). NB: da ricalcolare correggendo il clustering (§4).
- Errori concentrati su livelli adiacenti; l5 difficilissimo (error rate >70%, confuso con
  l4); l1 il più facile (<0.20).

---

## 8. Conversione template IEEEtran → LNCS

Il vecchio `main.tex` è IEEEtran a due colonne; il nuovo è **LNCS a colonna singola**.
- `\documentclass[runningheads]{llncs}`; servono `llncs.cls` e `splncs04.bst`.
- Rimuovere pacchetti IEEE-specifici (`cuted`, `cite`, il blocco `caption` in conflitto).
- Front matter: affiliazioni in `\institute` con `\inst{n}`; `\titlerunning`/`\authorrunning`;
  keyword via `\keywords{... \and ...}` DENTRO l'abstract; abstract DOPO `\maketitle`.
  ORCID via `\orcidID{...}`.
- Ringraziamenti (SIMULARE) in `\begin{credits}\subsubsection{\ackname}...\end{credits}`.
- `\bibliographystyle{splncs04}`.
- Il corpo (sezioni, equazioni, tabelle, figure) resta quasi identico; `figure*`/`table*`
  spanning diventano non-stellate (singola colonna).
- **Vincolo: ≤ 12 pagine + refs illimitate.** Passando a colonna singola il testo si allunga:
  probabile necessità di tagliare / spostare in supplementary.

---

## 9. Theory of Change (sezione obbligatoria IR for Good)

Per i paper che propongono metodi/strumenti serve una sezione dedicata che argomenti:
1. il **bisogno sociale** identificato e come il lavoro lo affronta;
2. le **precondizioni/assunzioni** perché abbia l'effetto voluto, e quanto sono plausibili;
3. le possibili **esternalità negative / nuovi danni**.

Materiale già presente (≈80%): l'analisi dei bias politici (sovrastima degli estremi,
penalizzazione dello slightly-left) È una discussione di esternalità negative; la cornice
"strumento di supporto al giudizio esperto, non sostituto del giornalista" è già mezza
theory of change. Va scritta come sezione centrale e onesta, non incollata alla fine.

---

## 10. Principi guida / guardrail

- Preservare la narrazione; le modifiche sono additive.
- Non sovravvendere la novità; niente claim gonfiati ("large-scale", "model scale is the
  bottleneck").
- Ogni esperimento va chiuso PRIMA della deadline (5 ott); il margine ECIR→WebSci serve solo
  a riformattare, non a fare esperimenti nuovi.
- Non committare `key.txt` né dati proprietari NewsGuard.

---

## 11. Prossimi passi suggeriti (da concordare prima di scrivere)

1. Creare `Paper/NewIRForGood/main.tex` in LNCS (struttura + scaletta), riportando il testo
   del vecchio paper.
2. Aggiungere: sottosezione **baseline zero-shot** (§5), **fix statistici** (§4), sezione
   **Theory of Change** (§9).
3. Rigenerare tabella comparativa fine-tuned vs GPT-5-mini e (se si fa) la curva di
   aggregazione, dai CSV dei risultati.
4. Tagliare per stare in 12 pagine.
