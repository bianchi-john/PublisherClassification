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
