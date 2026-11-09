# Attempt 20: from scratch

**Question.** Can the text of its articles alone tell how reliable a news outlet is (NewsGuard 2026 score 0–100 and its
five levels), for outlets never seen in training, once every trace of the outlet's identity is removed from the text?
And is it better to predict the level (classification) or the exact score (regression)?

**The whole pipeline, in order**, one notebook per step (built by `build_notebooks.py`):

| step | notebook | code | what it does |
|---|---|---|---|
| 1 | `01_scraping` | `scraper.py`, `run_scraper.sh` | the October 2026 scraping (not re-run: the notebook reads its output) |
| 2 | `02_cleaning` | `cleaning.py` | political outlets, text from the HTML (trafilatura precision), English, 100–1,500 words, from 2025, no duplicates across outlets, ≥ 10 articles, one outlet per ownership group, NewsGuard score and level |
| 3 | `03_anonymization` | `anonymization.py` | six rules remove the outlet signature, e-mails, links, accounts, own authors and outlet name; automatic check that nothing is left |
| 4 | `04_split` | `models.py` | one split for both tasks: 5 folds of outlets stratified by exact score (balanced by level too) |
| 5 | `05_embeddings` | `models.py` | token counts (no article truncated) and frozen vectors of Qwen3-Embedding-8B and Mistral-Small-24B |
| 6 | `06_classification` | `models.py` | the 5 levels: TF-IDF, Qwen3-8B, Mistral-24B (+ logistic regression), ModernBERT |
| 7 | `07_regression` | `models.py` | the exact score: the same four models (+ ridge) |
| 8 | `08_evaluation` | `metrics.py` | every metric per outlet, classifier and regressor in the same two spaces (levels and score), 95% bootstrap intervals, paired comparison |
| 9 | `09_results` | | all results on one page |
| 10 | `10_boost` | `boost.py`, `boost_eval.py`, `boost_run.py`, `run_boost.sh` | added 6/11: more strategies on the frozen embeddings only (TF-IDF and ModernBERT untouched): outlet and rare-score weights, quantile target, outlet models (mean vector, random bags, mixup), RBF kernel ridge, small neural network, attention over the articles, Qwen ⊕ Mistral and Mistral layer 40, averages and «auto» (method chosen on the inner folds); curves over the articles per outlet, paired bootstrap vs TF-IDF and vs the ridge of step 7 |
| 11 | `11_meta` | `meta.py`, `meta_run.py`, `run_meta.sh` | added 8/11: meta-articles (leads of ~20 articles of one outlet in one text, 10 per outlet, the same number for every outlet) for TF-IDF, Qwen3-8B, Mistral-24B and ModernBERT (fine-tuned with fresh meta-articles every epoch and early stopping on training outlets kept apart), compared with the same models on single articles |
| 12 | `12_more_outlets` | `more_outlets.py`, `run_more.sh` | added 8/11, all in one notebook: the non-political NewsGuard outlets we scraped are cleaned and anonymized with the rules of steps 2-3 (copies of political articles removed, ownership groups isolated, ≤ 30 articles per outlet) and added to the TRAINING only; the test stays the 226 political outlets; TF-IDF, Qwen3-8B and Mistral-24B (ridge, outlet-weighted ridge, small neural network) with and without them |
| – | | `mbfc.py` | comparison with Media Bias/Fact Check (4 Factual Reporting classes): the score, true or predicted, becomes an MBFC class with quotas, medians or thresholds learned on the other folds (used by the final-state deck) |

**Fixed before the results.**
- The dataset rules are those of attempt 16; the split is the first one of attempt 18; so steps 2 and 4 must give the
  same 226 outlets, 25,524 articles and folds.
- Every model reads every article in full (`MAX_TOKENS` = 4,096, above the longest article for every tokenizer).
- Linear models: penalty chosen with 3 inner folds of the training outlets (macro F1 for classification, Spearman for
  regression). ModernBERT: 5 epochs, lr 3e-5, effective batch 32, last epoch.
- For every test article of every fold the logits or the predicted score are saved (`predictions/`), and the model of
  every fold (`Modelli/20_FromScratch/`), so every metric can be recomputed.
- Fully reproducible: one seed (`SEED` = 42; `SPLIT_SEED` = 1), `random_state` in every model, deterministic GPU
  algorithms, library versions printed in every notebook.

**Where the data are.** Everything produced goes to `Dataset/tentativi/20_FromScratch/` (outside git: NewsGuard labels
are not redistributed); trained models to `Modelli/20_FromScratch/`. `common.py` finds `Dataset/` by walking up from
the working directory (on the server `~/John/Dataset`).

**Step 10** runs on its own: `screen -dmS t20_boost ./run_boost.sh` on the server (one process per representation and
method, 3 h limit each, failures retried once at the end; everything resumes from the saved folds), `./status_boost.sh`
from the PC (`./status_boost.sh download` brings back the notebook and the tables). Outputs in `boost/`.

**Step 12**: `screen -dmS t20_more ./run_more.sh` (its GPU part waits for the step 11 queue), `./status_more.sh [download]`;
ModernBERT on meta-articles with the extra outlets only with `-p RUN_BERT True`. Outputs in `more_outlets/`.
**Step 11** the same way: `screen -dmS t20_meta ./run_meta.sh` on the server, `./status_meta.sh [download]` from the PC.
Outputs in `meta/`.

**How to run.** From the PC: `./check_server.sh`, `./upload.sh`. On the server, in `~/John/Tentativo20`:
`screen -dmS t20_cpu ./run_cpu.sh` and `screen -dmS t20_gpu ./run_gpu.sh`. Then from the PC `./status.sh` and
`./download.sh` (`./download.sh all` also brings the article outputs).

**Results** (`09_results.ipynb`; per outlet, 95% bootstrap intervals; 226 outlets, 25,524 articles).
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

**Step 12 results** (`12_more_outlets.ipynb`, 8/11; extra dataset: 1,700 candidate outlets → 232 after cleaning, copies
and ownership groups, mostly large networks such as Natural News (537 sites) or Newsquest (147) → 6,534 articles; training
≈ 181 political + 232 extra outlets per fold; test unchanged, checked): small and mostly not significant changes. All
articles, stretched: TF-IDF Spearman 0.696 → 0.710, Qwen ridge 0.726 → 0.739 (MAE 15.3 → 14.6), Mistral ridge 0.735 →
0.736 (MAE 15.2 → 14.6), Mistral network 0.749 → 0.730; the only significant changes are «within 20 points» for the
Mistral ridge on the predicted score (68.6% → 73.4%). With few articles the same picture. More outlets of other kinds do
not move the ceiling.
