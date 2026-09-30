# From Articles to Publishers

Code and results for *"From Articles to Publishers: Aggregating Language Model
Predictions for News Source Reliability Inference"* (ECIR 2027, IR for Good track).

Transformer models estimate the reliability of individual news articles; those
predictions are then aggregated to infer the reliability of **previously unseen
publishers**, under a strict publisher-disjoint evaluation protocol.

---

## Repository layout

The three numbered directories are the pipeline, in order. Each stage consumes
what the previous one produced.

```
0-scraping/         Collect and clean the corpus
  0 - NgScraperPol.ipynb        scrape articles from NewsGuard-rated domains
  1 - Data preparation.ipynb    language filtering, cleaning, label assignment

1-classification/   Produce per-article predictions
  2 - Reliability - 5Fold - Separate.ipynb            BERT, RoBERTa, Llama (5-fold, publisher-disjoint)
  3 - Reliability - 5Fold - Separate - mistral22B.ipynb   Mistral-24B (1024-token context)
  7 - Get prediction scores.ipynb                     re-inference for per-class probabilities (GPU)
  8 - ChatGPT Run.ipynb                               GPT-5-mini zero-shot baseline

2-analysis/         Turn predictions into every figure and table in the paper
  paper_results.ipynb    THE notebook: run it top to bottom
  build_notebook.py      generates it (edit this, not the .ipynb)
  FINDINGS.md            what the old analysis pipeline got wrong, and why
  irgood/                the supporting package

Results/            Everything produced
  predictions/finetuned/   per-fold metrics and per-article predictions (4 models)
  predictions/zeroshot/    GPT-5-mini predictions and run logs
  figures/                 paper figures (regenerated)
  tables/                  paper tables (regenerated)
  derived/                 tidy per-article and per-publisher tables
  _archive/                superseded outputs, kept for provenance

Paper/
  NewIRForGood/            the LNCS submission (+ SUBMISSION-RULES.md)
  OldPaperRejected/        the IEEEtran version rejected by IEEE TCSS

Dataset/            NOT in the repository -- see "Data" below
HANDOFF.md          full project context: reviews, revision plan, status
```

Stages 0 and 1 are expensive (scraping, GPU fine-tuning) and are kept **with
their cell outputs** so the runs remain inspectable without re-executing them.
Stage 2 is cheap and fully reproducible from `Results/predictions/`.

## Reproducing the analysis

Stage 2 needs no GPU and no re-training:

```bash
cd 2-analysis
jupyter lab paper_results.ipynb    # run all cells
```

Requires `pandas numpy scipy scikit-learn statsmodels matplotlib seaborn`.
Everything lands in `Results/figures`, `Results/tables` and `Results/derived`.

To change the notebook, edit `build_notebook.py` and re-run it — the `.ipynb` is
a generated artefact, which keeps its prose and code reviewable as plain text.

## Data

Article-level reliability and political-orientation labels come from
**NewsGuard**, licensed to the authors. Those ratings are proprietary and are
**not redistributable**, so `Dataset/` is excluded from this repository.

The analysis in `2-analysis/` runs from the prediction files in
`Results/predictions/`, which contain article ids, URLs, domains, gold labels and
predicted labels — but no article text. The one proprietary input it still needs
is `Dataset/newsguard.csv`, used solely to map a domain to its political
orientation.

## A note on the published numbers

`2-analysis/FINDINGS.md` documents three defects found in the original analysis
code, all verified by recomputation. The most consequential: the old plotting
notebook loaded `llama.csv` into the RoBERTa variable and vice versa, so the
**article-level F1 and tolerant-accuracy figures attribute each model's scores to
the other**. Publisher-level results are unaffected. The current pipeline binds
each model name to its file in exactly one place, and asserts that every model
saw the same ground truth for the same article.
