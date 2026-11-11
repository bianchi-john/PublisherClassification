# Journal draft

Draft of the journal version (venue not chosen yet). Generic template: the plain `article` class, so that the body can
be moved to any journal class (elsarticle, svjour3, acmart, IEEEtran, ...) by replacing preamble and front matter.

- `main.tex`: the whole paper, described at a high level; author notes are `\todo{...}` (red), results to fill are `--`.
- `references.bib`: only references we are sure of; every entry has `note = {verify}`.
- `figures/make_figures.py`: the figures, from `Dataset/` (outside git) and the code in `Code/Pipeline`:
  `dataset.pdf` (final: the dataset and the split do not change) and `curves.pdf` (drawn as soon as the step-7
  predictions of the current run are in `Dataset/tentativi/20_FromScratch/predictions/regression/`; until then the
  paper shows a placeholder).

Build: `cd figures && python make_figures.py && cd .. && latexmk -pdf main.tex`.

Where the numbers come from: dataset and cleaning (`Code/Pipeline/02_cleaning.ipynb`), anonymization
(`03_anonymization.ipynb`), main results (`08_evaluation.ipynb`, `09_results.ipynb`, regressor rows, predicted score),
anonymization robustness (`12_anonymization_effect.ipynb`).
