# Journal draft

Draft of the journal version (venue not chosen yet). Generic template: the plain `article` class, so that the body can
be moved to any journal class (elsarticle, svjour3, acmart, IEEEtran, ...) by replacing preamble and front matter.

- `main.tex`: the whole paper; author notes are `\todo{...}` (red).
- `references.bib`: the entries of the first version (checked then) and the new ones; entries with `note = {verify}`
  still need checking.
- `figures/`: `paper_figures.ipynb` draws every figure of the paper (`dataset`, `groups`, `network`, `anonymization`,
  `results`, `curves`, `scatter`, `errors`.pdf) with the style of `paper_style.py`; `analyses.py` computes the numbers
  of the further analyses (`analyses.json`, outside git with the other aggregates); `dataset_construction.py` runs on the server and writes the aggregates of
  the dataset figures into `Dataset/tentativi/20_FromScratch/paper_figures/` (outside git).
- `_da_cancellare/`: previous versions (the draft before the full rewrite, the old figure scripts), for John to decide.

Build: run `figures/paper_figures.ipynb` (kernel `irgood-venv`; e.g. `papermill paper_figures.ipynb paper_figures.ipynb -k irgood-venv --cwd figures`), then `latexmk -pdf main.tex`. Figures follow the style of the first (rejected) version and stay readable without colour (hatches, markers, dash patterns; the notebook's last cell renders them in greyscale).

Where the numbers come from: dataset and cleaning (`Code/Pipeline/02_cleaning.ipynb`), anonymization
(`03_anonymization.ipynb`), main results (`08_evaluation.ipynb`, `09_results.ipynb`, regressor rows, predicted score),
anonymization robustness (`12_anonymization_effect.ipynb`), ownership groups, shared text and anonymization figures
(`figures/dataset_construction.py`, run on the server; aggregates in `Dataset/tentativi/20_FromScratch/paper_figures/`),
released resources and tool (Section "Released resources
and demonstration tool": the tool's repository is `demo/publisher-evaluation`, outside this repository's git; its
fidelity numbers are in `demo/publisher-evaluation/docs/checks/`).
