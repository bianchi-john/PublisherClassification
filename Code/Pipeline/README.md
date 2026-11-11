# Pipeline of the paper (journal version)

**Question.** Can the text of its articles alone tell how reliable a news outlet is (NewsGuard 2026 score, 0-100), for
outlets never seen in training, once the explicit traces of the outlet's identity are removed from the text? Do large
language models used as frozen encoders do better than TF-IDF?

Born as attempt 20 ("from scratch", `Code/Attempts` has the 19 attempts before it). One notebook per step, built by
`build_notebooks.py` (a notebook whose code does not change is left untouched, outputs included). Everything is in
English, seeded and reproducible. The paper draft is in `Paper/Journal/`.

**Status (11/11/2026).** The whole pipeline is being re-run with the anonymization of step 3 (rules + GLiNER); the
outputs of the first run (anonymization made of six rules) are kept in `rules_only/` (see Results).

## Steps used in the paper

| step | notebook | code | what it does |
|---|---|---|---|
| 1 | `01_scraping` | `scraper.py`, `run_scraper.sh` | the October 2026 scraping of the NewsGuard outlets, raw HTML saved (not re-run: the notebook reads its output) |
| 2 | `02_cleaning` | `cleaning.py` | political outlets only, text re-extracted from the HTML (trafilatura, precision mode), English, 100-1,500 words, from 2025, exact and near duplicates removed across outlets, >= 10 articles, one outlet per ownership group, NewsGuard score and level: 226 outlets, 25,524 articles |
| 3 | `03_anonymization` | `anonymization_ner.py`, `anonymization.py` | links and e-mails → `[URL]`, `[EMAIL]`; the outlet's own name from its domain → `[OUTLET]`; GLiNER (zero-shot NER, `urchade/gliner_large-v2.1`) with "news outlet" → `[OUTLET]` and "journalist" → `[AUTHOR]` (every outlet and journalist); check: articles still containing their own outlet's name |
| 4 | `04_split` | `models.py` | 5 folds of outlets stratified by the exact score |
| 5 | `05_embeddings` | `models.py` | token counts (no article truncated) and frozen vectors of Qwen3-Embedding-8B and Mistral-Small-24B |
| 7 | `07_regression` | `models.py` | the score of every article: TF-IDF, Qwen3-8B, Mistral-24B (+ ridge), fine-tuned ModernBERT |
| 8-9 | `08_evaluation`, `09_results` | `metrics.py` | per outlet (mean of the article predictions): Spearman, Kendall, MAE, within 20 points; 95% bootstrap intervals; paired comparisons. For the paper: regressor rows, predicted score |
| 12 | `12_anonymization_effect` | `boost.py`, `boost_eval.py` | the regressors on the final text against the same regressors of the first run (rules-only anonymization), same draws, paired bootstrap |

## Side analyses (not in the paper)

| step | notebook | code | what it does |
|---|---|---|---|
| 6 | `06_classification` | `models.py` | the 5 NewsGuard levels with the same models (classifier vs regressor in 08-09) |
| 10 | `10_boost` | `boost*.py`, `run_boost.sh` | 14 more regressors on the frozen embeddings (weights, kernels, small network, attention over articles, ...) |
| 11 | `11_meta` | `meta*.py`, `run_meta.sh` | meta-articles: the leads of ~20 articles of one outlet in one text |
| - | | `mbfc.py` | comparison with Media Bias/Fact Check (used by the slides) |

The slides also use the "stretched" score (predictions re-mapped to the training score quantiles, `metrics.py`); the
paper does not.

## Fixed before the results
- The dataset rules are those of attempt 16 and the split is the first one of attempt 18: steps 2 and 4 give the same
  226 outlets, 25,524 articles and folds (checked at every run).
- Every model reads every article in full (`MAX_TOKENS` = 4,096; the longest article has about 3,000 tokens).
- Linear models: penalty chosen with 3 inner folds of the training outlets (Spearman per outlet for regression).
  ModernBERT: 5 epochs, lr 3e-5, effective batch 32, last epoch.
- For every test article of every fold the predicted score is saved (`predictions/`), and the model of every fold
  (`Modelli/20_FromScratch/`), so every metric can be recomputed.
- One seed (`SEED` = 42; `SPLIT_SEED` = 1), `random_state` in every model, deterministic GPU algorithms, library versions
  printed in every notebook.

## Data and outputs

Everything goes to `Dataset/tentativi/20_FromScratch/` (outside git: NewsGuard labels are not redistributed; the
folder keeps the attempt's name so that the server and the local copy match); `common.py` finds `Dataset/` by walking
up from the working directory (on the server `~/John/Dataset`).

```
articles.csv.gz, extracted.csv.gz   step 2: the clean articles (original text) and the trafilatura cache
outlets.csv, split.csv              step 2 and 4: the 226 outlets (score, level, orientation) and their fold
articles_anonymous.csv.gz           step 3: the text every model reads; anonymization/ the GLiNER spans
embeddings/                         step 5: frozen vectors (one row per article, order in article_order.csv)
predictions/<task>/<model>/         steps 6-7: one file per fold, one row per test article
all_metrics.csv, intervals.csv      step 8
anonymization_effect.csv            step 12
boost/, meta/                       side analyses 10 and 11
rules_only/                         the first run (anonymization made of six rules), kept for step 12
```
Trained models: `Modelli/20_FromScratch/` (first run in `Modelli/20_FromScratch/rules_only/`).

## How to run

On the server (artes4, GPU V100 32 GB shared with others; the code lives in `~/John/Tentativo20`):
- from the PC: `./check_server.sh` (models, libraries, data, GPU), `./upload.sh` (code; `./upload.sh notebooks` also the
  notebooks);
- on the server: `screen -dmS t20_gpu ./run_gpu.sh` and `screen -dmS t20_cpu ./run_cpu.sh` (the whole pipeline, in
  order; every step resumes from what is saved);
- from the PC: `./status.sh` (progress), `./download.sh` (notebooks and tables; `./download.sh all` also predictions).
Steps 10 and 11 have their own queues (`run_boost.sh`, `run_meta.sh`; `status_boost.sh`, `status_meta.sh`).

## Code

| file | content |
|---|---|
| `common.py` | data folders, seeds, NewsGuard levels, logging |
| `scraper.py` | the scraper (step 1) |
| `cleaning.py` | text extraction, language, duplicates, ownership groups (step 2) |
| `anonymization_ner.py` | step 3: the two rules and GLiNER |
| `anonymization.py` | the six rules of the first run; its name rule and link/e-mail patterns are reused by step 3 |
| `models.py` | split, frozen vectors, linear models, ModernBERT |
| `metrics.py` | from articles to outlet, metrics, bootstrap |
| `boost.py`, `boost_eval.py`, `boost_run.py` | step 10; `boost.py` also gives the test draws of k articles and `boost_eval.py` the paired bootstrap used by step 12 and by the paper figures |
| `meta.py`, `meta_run.py` | step 11 |
| `mbfc.py` | Media Bias/Fact Check (slides) |
| `build_notebooks.py` | all the notebooks |
| `run_*.sh`, `status*.sh`, `upload.sh`, `download.sh`, `check_server.sh` | server queues and helpers |

## Results

### Current run (rules + GLiNER anonymization)
To be filled when the re-run of 11/11 ends (`08_evaluation`, `09_results`, `12_anonymization_effect`).

### First run (anonymization made of six rules; kept in `rules_only/`)
- Steps 2 and 4 reproduce attempt 16 (same 25,524 articles, texts and labels) and the first split of attempt 18;
  re-running steps 1-5 gives identical files. No article is truncated (longest: 3,000 tokens).
- Anonymization touches 54.6% of the articles; no distinctive outlet name is left as a word. Known leftovers: acronyms
  of the outlet name (e.g. "TPM") and author names written differently from the author field (e.g. in capitals).
- Levels, F1 macro (without text 0.10): Mistral-24B regressor + training quotas 0.42 [0.35; 0.49], Qwen3-8B 0.40
  (classifier and regressor), TF-IDF 0.39 (regressor) / 0.32 (classifier), ModernBERT 0.34 / 0.26. With the fixed
  NewsGuard bounds the regressor drops to 0.24-0.28 (predictions squeezed to the centre).
- Score, MAE (without text 22.6): Mistral-24B 14.8 (classifier, expected score) / 15.7 (regressor), Qwen3-8B 15.9 /
  16.1, TF-IDF 16.6 / 17.7, ModernBERT 18.3 / 17.7. Spearman 0.73 (Mistral) down to 0.58-0.61 (ModernBERT).
- Anonymization changes almost nothing compared with attempts 16 and 18 (Mistral MAE 14.8 vs 14.7): the models read
  the text, not the outlet name. Large frozen models > TF-IDF > fine-tuned ModernBERT.
- Classifier vs regressor (paired): on the score they are equivalent; on the levels the regressor with training quotas
  is equal or better (Spearman and quadratic kappa significantly higher for Mistral, TF-IDF and ModernBERT); 31 of 100
  paired differences are not chance.

### Side analyses
**Step 10 results** (`10_boost.ipynb`, run 6-7/11; 14 methods × 5 representations, 70 of 70 runs, no errors; per outlet,
stretched score, all articles; paired bootstrap, 1,000 resamples):
- No big jump. Mistral ridge (step 7) Spearman 0.735, Kendall 0.554, MAE 15.2; «auto» (method chosen on the inner folds,
  the honest number) 0.743 / 0.572 / 14.6, not significantly different from the ridge; TF-IDF 0.696 / 0.518 / 16.3.
- Best row: Mistral + small neural network 0.749 / 0.575 / 14.2 (also among the best on the inner folds). Against the
  ridge: not significant on the stretched score; on the predicted score MAE 14.8 vs 15.7 and within 20 points 73.9% vs
  68.6%, significant. With few articles it lowers the error a little (5 articles: MAE 15.2 vs 16.0), the order stays.
- No help from outlet or rare-score weights, mixup, Mistral layer 40, concatenations (Qwen ⊕ Mistral, 27 ⊕ 40) or
  outlet-level models; nearest outlets (`outlet_knn`) are clearly worse (Spearman 0.49-0.62).
- On the inner folds every good method is between 0.72 and 0.73: with these frozen vectors the ceiling is about 0.75.
  The advantage over TF-IDF does not depend on the choice of the regressor.

**Step 11 results** (`11_meta.ipynb`, 8/11; all 10 meta-articles per outlet vs the same model on whole articles, stretched
score, paired bootstrap): meta-articles never help. Spearman meta vs articles: TF-IDF 0.644 vs 0.696, Qwen ridge 0.596 vs
0.726, Qwen network 0.620 vs 0.718, Mistral ridge 0.712 vs 0.735, Mistral network 0.669 vs 0.749, ModernBERT 0.570 vs
0.602 (significantly worse for Qwen and the Mistral network; the others not significant). ModernBERT on meta-articles
looked better on the outlets kept apart in fold 1 (0.66) but not overall.
