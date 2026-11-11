# Publisher Classification

Predicting the **reliability of a news outlet the model has never seen, from the text of its articles alone**: the
NewsGuard score (0-100) of an outlet is estimated from its articles, with all traces of the outlet's identity removed
from the text. The question of the current work: do large language models used as frozen text encoders predict it
better than a TF-IDF model?

**Start here:** `Code/Pipeline/README.md` (the pipeline of the paper in preparation) and `Paper/Journal/` (the draft).
`CLAUDE.md` is a short guide to the repository in Italian, also meant for another Claude instance.

## Layout

```
Code/
  Pipeline/        the pipeline of the paper in preparation (journal version): one notebook per step, in English
  Attempts/        the 19 attempts that led to it (Oct-Nov 2026, in Italian), with the history in its README
  IrForGood/       code of the ECIR 2027 IR for Good submission
  RejectedTCSS/    code of the version rejected by IEEE TCSS
Paper/
  Journal/         LaTeX draft of the journal paper (generic template; results to be filled in)
  NewIRForGood/    the IR for Good submission
  OldPaperRejected/  the rejected TCSS paper
Presentazioni/     slides of the group meetings (Italian)
Dataset/, Modelli/ data and trained models: NOT in git (see Data)
```

## The pipeline (`Code/Pipeline`)

| step | notebook | what it does |
|---|---|---|
| 1 | `01_scraping` | October 2026 scraping of the NewsGuard political outlets (raw HTML; not re-run) |
| 2 | `02_cleaning` | text from the HTML (trafilatura, precision mode), English, 100-1,500 words, from 2025, duplicates removed across outlets, >= 10 articles per outlet, one outlet per ownership group: 226 outlets, 25,524 articles |
| 3 | `03_anonymization` | links and e-mails, the outlet's own name, and every news outlet and journalist found by GLiNER (zero-shot NER) become placeholders |
| 4 | `04_split` | 5 folds of outlets, stratified by the exact score |
| 5 | `05_embeddings` | frozen vectors of Qwen3-Embedding-8B and Mistral-Small-24B |
| 7 | `07_regression` | the score of every article with TF-IDF, Qwen and Mistral (+ ridge) and fine-tuned ModernBERT; the outlet score is the mean |
| 8-9 | `08_evaluation`, `09_results` | per outlet: Spearman, Kendall, MAE, within 20 points; bootstrap intervals |
| 12 | `12_anonymization_effect` | results with this anonymization against a simpler one (rules only) |

Steps 6, 10 and 11 (classification, more regressors on the embeddings, meta-articles) are side analyses not used in
the paper. The heavy steps run on the server (`run_cpu.sh`, `run_gpu.sh`; progress with `status.sh`).

## Data

Reliability and political-orientation labels come from **NewsGuard** and are licensed to the authors. They are
proprietary and **not redistributable**: they are not in git, not even inside results.

All data (scraping, labels, the intermediate and final datasets) are in a single `Dataset/` folder at the repository
root, trained models in `Modelli/`; both are outside git and shared separately (`Dataset/LEGGIMI.md`, tracked,
describes the layout). The pipeline writes to `Dataset/tentativi/20_FromScratch/` (the name of the attempt it comes
from, kept so that the server and the local copy match). `Code/Attempts/crea_archivio_dataset.py` builds the
`Dataset.zip` of the earlier protocol.

## Earlier versions

`Code/Attempts/` holds the 19 attempts (Oct 2026) on the way to the pipeline; its README says what each one found.
The older paper versions keep their own layout:

| | |
|---|---|
| `0-scraping/` | collect and clean the corpus |
| `1-…/` | the version's experiments (classification, data cleaning, …) |
| `2-analysis/` | figures and tables of the paper |
| `Dataset/` | NewsGuard labels and scraped articles (**not in git**) |
| `Results/` | predictions, figures, tables |

### IrForGood

Stages 0 and 1 need scraping access and a GPU, so they are kept **with their cell
outputs** and are not meant to be re-run. Stage 2 is cheap and reproduces
everything from the saved predictions:

```bash
cd Code/IrForGood/2-analysis && jupyter lab paper_results.ipynb   # run all cells
```

Needs `pandas numpy scipy scikit-learn statsmodels matplotlib seaborn`.

```
Code/IrForGood/Results/
  predictions/   per-article predictions: 4 fine-tuned models + GPT-5-mini
  figures/       paper figures
  tables/        paper tables
  derived/       tidy per-article and per-publisher tables
  _archive/      superseded output, kept for provenance
```

`Code/IrForGood/HANDOFF.md` is the context document of the IR for Good revision.

#### Correction to the published numbers

The original plotting code loaded `llama.csv` into the RoBERTa variable and vice
versa, so the article-level F1 and tolerant-accuracy figures in the rejected
paper **attribute each of those two models' scores to the other**. Publisher-level
results are unaffected. The first cells of `paper_results.ipynb` explain this and
the two other defects found, all verified by recomputation.

#### Rendering delle figure

Le figure sono composte da LaTeX (backend PGF di matplotlib, `text.usetex=True`),
come nel paper originale. Serve quindi un `pdflatex` nel PATH. Senza sudo:

```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tlmgr install pgf type1cm cm-super underscore ucs preview
```

Senza `pdflatex`, `plots.set_style()` avvisa e ripiega su mathtext: il layout
resta corretto ma i font non coincidono con quelli del paper.


The IrForGood analysis runs from `Code/IrForGood/Results/predictions/`, which holds article ids, URLs, domains and
labels but **no article text**.
