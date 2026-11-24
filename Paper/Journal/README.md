# Journal draft

Draft of the journal version (venue not chosen yet). Generic template: the plain `article` class, so that the body can
be moved to any journal class (elsarticle, svjour3, acmart, IEEEtran, ...) by replacing preamble and front matter.

- `PIANO_REVISIONE.md`: the plan to make the paper reviewer-proof (status of every experiment and writing task; update it
  as work is done); `MODIFICHE_TESTO.md`: the changes made to the text for it, section by section.
- `main.tex`: the paper, with the appendices at the end (Appendices A-N: details, extra tables and the tool); author
  notes are `\todo{...}` (red). Until 23/11/2026 the appendices were a separate `supplementary.tex` (now in
  `_da_cancellare/`).
- `references.bib`: every entry checked against its source (ACL Anthology, Crossref, arXiv, PMLR, publishers) on
  17/11/2026.
- `figures/`: `paper_figures.ipynb` draws every figure of the paper (`dataset`, `groups`, `network`, `anonymization`,
  `results`, `curves`, `scatter`, `errors`, `group_maps`.pdf) with the style of `paper_style.py`, from the tables of step 15 of the
  pipeline (`paper_*.csv`); `dataset_construction.py` and `group_maps.py` (ownership-group maps through the cleaning, Figure 2; `overlap_steps.py` is the earlier bar-chart version) run on the server and writes the aggregates of the dataset
  figures into `Dataset/tentativi/20_FromScratch/paper_figures/` (outside git).
- `_da_cancellare/`: previous versions (the draft before the full rewrite, the old figure scripts), for John to decide.

Build: run `figures/paper_figures.ipynb` (kernel `irgood-venv`; e.g. `papermill paper_figures.ipynb paper_figures.ipynb -k irgood-venv --cwd figures`), then `latexmk -pdf main.tex`. Figures follow the style of the first (rejected) version and stay readable without colour (hatches, markers, dash patterns; the notebook's last cell renders them in greyscale).

Where the numbers come from: dataset and cleaning (`Code/Pipeline/02_cleaning.ipynb`), anonymization
(`03_anonymization.ipynb`), every result table, comparison and analysis of Sections 5.2-5.7 and Appendices A-D
(`15_paper_results.ipynb`, files `paper_*.csv`; the ladder from `14_leakage_ladder.ipynb`, the other splits from
`16_other_splits.ipynb`), the earlier rule-based anonymization (`12_anonymization_effect.ipynb`), ownership groups, shared text and anonymization figures
(`figures/dataset_construction.py`, run on the server; aggregates in `Dataset/tentativi/20_FromScratch/paper_figures/`),
released resources and tool (Section "Released resources
and demonstration tool": the tool's repository is `demo/publisher-evaluation`, outside this repository's git; its
fidelity numbers are in `demo/publisher-evaluation/docs/checks/`).
