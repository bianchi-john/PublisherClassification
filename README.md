# From Articles to Publishers

Code for *"From Articles to Publishers: Aggregating Language Model Predictions
for News Source Reliability Inference"* (ECIR 2027, IR for Good track).

Language models rate individual news articles; those ratings are aggregated to
infer the reliability of **publishers the model has never seen**.

## Pipeline

Three stages, in order.

| | | |
|---|---|---|
| `0-scraping/` | collect and clean the corpus | notebooks 0–1 |
| `1-classification/` | produce per-article predictions | notebooks 2–5 |
| `2-analysis/` | every figure and table in the paper | `paper_results.ipynb` |

Stages 0 and 1 need scraping access and a GPU, so they are kept **with their cell
outputs** and are not meant to be re-run. Stage 2 is cheap and reproduces
everything from the saved predictions:

```bash
cd 2-analysis && jupyter lab paper_results.ipynb   # run all cells
```

Needs `pandas numpy scipy scikit-learn statsmodels matplotlib seaborn`.

## Where things end up

```
Results/
  predictions/   per-article predictions: 4 fine-tuned models + GPT-5-mini
  figures/       paper figures
  tables/        paper tables
  derived/       tidy per-article and per-publisher tables
  _archive/      superseded output, kept for provenance
Paper/           the LNCS submission and the earlier rejected version
```

## Data

Reliability and political-orientation labels come from **NewsGuard** and are
licensed to the authors. They are proprietary and **not redistributable**, so
`Dataset/` is not part of this repository.

The analysis runs from `Results/predictions/`, which holds article ids, URLs,
domains and labels but **no article text**.

## Correction to the published numbers

The original plotting code loaded `llama.csv` into the RoBERTa variable and vice
versa, so the article-level F1 and tolerant-accuracy figures in the rejected
paper **attribute each of those two models' scores to the other**. Publisher-level
results are unaffected. The first cells of `paper_results.ipynb` explain this and
the two other defects found, all verified by recomputation.
