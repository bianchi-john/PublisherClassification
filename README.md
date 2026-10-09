# Publisher Classification

Inferring the reliability of **news publishers the model has never seen** from
collections of their articles. Language models rate individual articles; those
ratings are aggregated per publisher.

The work has gone through several paper versions. Each has its own code, data and
results under `Code/`; the papers themselves are under `Paper/`.

```
Code/
  RejectedTCSS/    code of the version rejected by IEEE TCSS
  IrForGood/       code of the ECIR 2027 IR for Good submission
  WebScience2027/  work in progress for Web Science 2027
Paper/             the rejected TCSS paper and the IR for Good submission
Riunioni/          slides and notes of the group meetings
```

Every version follows the same layout:

| | |
|---|---|
| `0-scraping/` | collect and clean the corpus |
| `1-…/` | the version's experiments (classification, data cleaning, …) |
| `2-analysis/` | figures and tables of the paper |
| `Dataset/` | NewsGuard labels and scraped articles (**not in git**) |
| `Results/` | predictions, figures, tables |

`WebScience2027/` starts from the same scraped dataset as the other versions:
its `0-scraping/` notebooks are a copy of the ones in `IrForGood/`.

## IrForGood version

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

### Correction to the published numbers

The original plotting code loaded `llama.csv` into the RoBERTa variable and vice
versa, so the article-level F1 and tolerant-accuracy figures in the rejected
paper **attribute each of those two models' scores to the other**. Publisher-level
results are unaffected. The first cells of `paper_results.ipynb` explain this and
the two other defects found, all verified by recomputation.

### Rendering delle figure

Le figure sono composte da LaTeX (backend PGF di matplotlib, `text.usetex=True`),
come nel paper originale. Serve quindi un `pdflatex` nel PATH. Senza sudo:

```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
tlmgr install pgf type1cm cm-super underscore ucs preview
```

Senza `pdflatex`, `plots.set_style()` avvisa e ripiega su mathtext: il layout
resta corretto ma i font non coincidono con quelli del paper.

## Data

Reliability and political-orientation labels come from **NewsGuard** and are
licensed to the authors. They are proprietary and **not redistributable**, so no
`Dataset/` folder is part of this repository.

The IrForGood analysis runs from `Code/IrForGood/Results/predictions/`, which
holds article ids, URLs, domains and labels but **no article text**.
