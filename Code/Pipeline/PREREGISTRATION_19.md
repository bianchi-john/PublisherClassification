# Step 19: replication on held-out outlets (pre-registration)

Written on 23 November 2026, **before** the held-out data were built and before any prediction was computed on them.
This file and the code of step 19 (`heldout.py`, `19_heldout.ipynb`, `run_heldout.sh`) are committed to git before the
run; the paper cites that commit. Nothing below is changed after the results are seen.

## Why

The rules of the dataset, the split and the models of the paper were fixed after earlier rounds of experiments on the
same 226 outlets, whose test results we saw (Appendix N of the paper). The advantage of the large frozen models over
TF-IDF is about 2 points of MAE, so part of it could be an adaptation of the design to these outlets. Step 19 tests the
frozen models on outlets that played no part in the development: no experiment ever reported a test result on them.

## Held-out outlets

Both sets are built from the October 2026 scraping used by step 2, with the NewsGuard ratings of 23 October 2026.

**A. Non-political outlets (main test).** Scraped outlets with a NewsGuard score whose topics are not only *political
news or commentary*.
1. The cleaning of step 2: text re-extracted from the HTML with trafilatura (precision), English, 100–1,500 words,
   published from 2025.
2. Duplicates: an article that is an exact or near copy (Jaccard of word 5-grams ≥ 0.5) of any article of the 226
   outlets is dropped; among the held-out articles, the oldest copy is kept.
3. At least 10 articles per outlet.
4. Ownership groups (NewsGuard Owner and Parent Domain, as in step 2) computed together with the 226 outlets: an outlet
   in the group of one of the 226 is dropped; then one outlet per group (the one with the most articles).
5. At most 30 articles per outlet, drawn at random (seed 42).

These rules are those of step 12 (8 November 2026), which built about 230 such outlets and used them only to train
models that were then discarded; their test results were never computed.

**B. Political outlets with fewer than 10 articles (in-domain check).** The political outlets of step 2 that passed
every cleaning step except "at least 10 articles" (60 outlets), without those in the ownership group of one of the 226
outlets (45 left), one outlet per group (the one with the most articles): about 40 outlets with 1–9 articles each.

**Anonymization.** Step 3 unchanged: links, e-mail addresses and the outlet's own name, then GLiNER
(`urchade/gliner_large-v2.1`, threshold 0.5) for news outlets and journalists.

## Models (frozen)

The saved fold models of step 7, applied without any retraining or new setting: TF-IDF + ridge, Qwen3-Embedding-8B +
ridge, Mistral-Small-24B (layer 27) + ridge, ModernBERT fine-tuned. The frozen vectors are computed as in step 5. A
held-out outlet belongs to no fold, so the prediction of an article is the mean of the five fold models (as in the
released tool), and the score of an outlet is the mean over its articles.

Baselines, computed on the 226 outlets: their median score, their mean score, and their mean score per political
orientation (NewsGuard *Orientation*, missing = none).

Check of the pipeline: 30 articles of the 226 outlets (test fold 1) go through the same embedding and prediction code;
their fold-1 predictions must match the saved predictions of step 7 (within 1 point; ModernBERT in half precision
differs by up to about 1 point with a different batching). If they do not, the run stops and the bug is fixed before any
held-out result is looked at.

## Analysis

- Metrics per outlet: MAE (primary), Spearman ρ and share within 20 points.
- 95% percentile bootstrap intervals over outlets (10,000 resamples, seed 42); each model is compared with TF-IDF on the
  same resamples (paired), with a two-sided bootstrap p-value.
- **Primary hypothesis (set A):** MAE(Mistral-Small-24B) − MAE(TF-IDF) < 0, with the 95% interval of the difference
  excluding zero.
- Secondary comparisons with TF-IDF: Qwen3-Embedding-8B and ModernBERT on the MAE of set A; Mistral-Small-24B,
  Qwen3-Embedding-8B and ModernBERT on the ρ of set A; the same six on set B. Holm correction over these 11 tests.

## How the result is reported

Whatever the outcome, in the results of the paper and, if the primary test fails, in the abstract.
- **Primary test passes:** the advantage of the large frozen model over TF-IDF replicates on outlets outside the
  development set.
- **Primary test fails:** the advantage did not replicate. Set A differs from the 226 outlets in topic, so a failure
  cannot separate an adaptation of the design to the development outlets from the change of topic; set B, in the same
  domain but small, is reported as the only in-domain indication.
- Absolute errors are not compared with those of the paper, since the score distributions differ.
- No other model, setting or subset is added after the results are seen; any further analysis is labelled exploratory.
