"""Generates `paper_results.ipynb`.

The notebook is kept under version control as a generated artefact so that its
prose and its code stay reviewable as ordinary text. Edit this file, re-run it,
then execute the notebook.

    python build_notebook.py
"""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Paper results — one notebook, end to end

Regenerates **every figure and table** in *"From Articles to Publishers"* from the
model outputs in `Results/predictions/finetuned/`.

The GPT-5-mini zero-shot baseline is deliberately **not** part of this notebook.
Its predictions remain in `Results/predictions/zeroshot/` and `irgood` can still
load them (`data.load_all_predictions(include_gpt=True)`), but nothing here reads
or plots them.

Run it top to bottom. There is no hidden state: every cell depends only on cells
above it, and nothing is written outside `Results/figures`, `Results/tables` and
`Results/derived`.

**What this replaces.** Notebooks `4 - Plotter`, `5 - Political Error Analysis`,
`6 - Domain inpact analisys` and the enrichment half of `7 - Get prediction scores`.
Those are kept for provenance but are no longer the source of any paper artefact.

**Three defects in the old pipeline that this one fixes** (details in
`2-analysis/FINDINGS.md`):

1. `4 - Plotter.ipynb` read `llama.csv` into `robertaDf` and `roberta.csv` into
   `llamaDf`, so the published **article-level F1 and tolerant accuracy have
   RoBERTa and Llama swapped**. The swap was also baked into
   `confusion_matrices_all_models.csv`, which notebook 5 then read.
2. "Mean error direction" was computed two different ways under one name —
   divided by the number of errors in one place, by the number of articles in
   another. The paper states the first formula but reports the second.
3. Publisher aggregation breaks ties toward the **lowest** reliability class,
   silently deciding 2.5–4.6% of publishers. Never documented.
""")

md("## 1. Setup")
code('''
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Make the `irgood` package importable no matter where Jupyter was started.
HERE = Path.cwd()
if not (HERE / "irgood").exists() and (HERE / "2-analysis" / "irgood").exists():
    HERE = HERE / "2-analysis"
sys.path.insert(0, str(HERE))

import pandas as pd

from irgood import config as C
from irgood import data, metrics, aggregate, bias, plots

# Reproduces the paper's PGF/LaTeX rendering. Needs pdflatex on PATH; without
# it the call falls back to mathtext and warns (layout right, fonts not).
plots.set_style(use_latex=True)
C.ensure_output_dirs()

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

print("version root :", C.VERSION_ROOT)
print("figures   ->", C.FIGURES_DIR)
print("tables    ->", C.TABLES_DIR)
print("derived   ->", C.DERIVED_DIR)
''')

md("""
## 2. Load and validate

One tidy table for everything: `model | article_id | url | domain | fold | y_true | y_pred`,
plus the publisher's political leaning.

The validation step is not decoration. It asserts that every model saw the same
ground truth for the same article — the check that would have caught the
RoBERTa/Llama swap immediately.
""")
code('''
# Fine-tuned models only. The zero-shot baseline is excluded on purpose.
preds = data.load_all_predictions(include_gpt=False, with_leaning=True)

report = data.validate(preds)
display(report.round(4))
print("Same ground truth across all models:", report.attrs["ground_truth_consistent"])

preds.to_csv(C.DERIVED_DIR / "predictions_long.csv", index=False)
print("\\nSaved:", C.DERIVED_DIR / "predictions_long.csv", preds.shape)
''')

md("""
## 3. Stage 1 — article-level reliability estimation

Two views of the same thing: metrics pooled over all five test folds (quoted in
the text) and the per-fold spread (the box plot).
""")
code('''
article_tbl = aggregate.article_metrics(preds)
display(article_tbl.round(4))

per_fold = data.load_per_fold_metrics()
y_true_articles = preds[preds["model"] == C.MODEL_ORDER[0]]["y_true"]

plots.fig_article_metrics_boxplot(per_fold, y_true_articles)
article_tbl.to_csv(C.TABLES_DIR / "article_level_metrics.csv", index=False)
''')

md("""
### Per-fold metrics, recomputed

The training runs each wrote their own per-fold metrics. Recomputing them from
the stored predictions is a cheap integrity check: if the two disagree, either
the predictions or the saved metrics have drifted.
""")
code('''
per_fold_all = data.per_fold_metrics(preds)

# Consistency check: recomputation must agree with what the training runs saved.
check = per_fold.merge(per_fold_all, on=["model", "fold"], suffixes=("_stored", "_recomputed"))
max_drift = max(
    (check[f"{m}_stored"] - check[f"{m}_recomputed"]).abs().max()
    for m in ["Accuracy", "F1_Macro", "Tolerant_1.0"]
)
print(f"max |stored - recomputed| across folds: {max_drift:.2e}")

spread = (
    per_fold_all.groupby("model", observed=True)["Accuracy"]
    .agg(fold_min="min", fold_max="max")
    .assign(range=lambda d: d["fold_max"] - d["fold_min"])
)
print("\\nAccuracy spread across the five folds:")
display(spread.round(4))

per_fold_all.to_csv(C.TABLES_DIR / "per_fold_metrics.csv", index=False)
''')

md("""
### Uncertainty

Bootstrap confidence intervals that resample **publishers**, not articles.
Articles from one outlet share a label and are not independent observations, so
an article-level bootstrap would understate the interval.
""")
code('''
rows = []
for model, g in preds.groupby("model", observed=True):
    point, lo, hi = metrics.bootstrap_ci(
        g["y_true"].to_numpy(), g["y_pred"].to_numpy(),
        metric="Accuracy", n_boot=2000, cluster=g["domain"].to_numpy(),
    )
    rows.append({"model": model, "accuracy": point, "ci_low": lo, "ci_high": hi})

ci_tbl = pd.DataFrame(rows)
display(ci_tbl.round(4))
print("Note how wide these are, and how much they overlap: once publishers are")
print("the resampling unit, the four models are not cleanly separable.")
ci_tbl.to_csv(C.TABLES_DIR / "article_accuracy_bootstrap_ci.csv", index=False)
''')

md("""
## 4. Stage 2 — publisher-level inference

Majority vote over each publisher's articles. The tie-break is reported rather
than assumed: it decides a small but non-trivial share of publishers, always in
the same direction (toward lower reliability), and by a different amount per
model.
""")
code('''
pub = aggregate.aggregate_to_publisher(preds)

publisher_tbl = aggregate.publisher_metrics(pub)
display(publisher_tbl.round(4))

print("\\nPublishers decided by the tie-break rule:")
display(aggregate.tie_report(preds).round(2))

y_true_publishers = pub[pub["model"] == C.MODEL_ORDER[0]]["y_true"]
plots.fig_publisher_metrics(publisher_tbl, y_true_publishers)

publisher_tbl.to_csv(C.TABLES_DIR / "publisher_level_metrics.csv", index=False)
pub.to_csv(C.DERIVED_DIR / "publisher_predictions.csv", index=False)
''')

md("""
### How much does aggregation actually buy?

The paper compares 0.60 at article level with 0.69 at publisher level, but those
two numbers differ in **two** ways at once: the unit of evaluation *and* the
weighting, since article-level accuracy is dominated by prolific publishers.

This curve separates them. Every point uses the same unit — the publisher — and
the same fixed set of publishers; only `k`, the number of articles voted,
changes. `k=1` is therefore the honest baseline, and the rise above it is what
aggregation contributes and nothing else.
""")
code('''
curve = aggregate.aggregation_curve(
    preds, k_values=(1, 3, 5, 10, 20, 30), n_repeats=20, fixed_population=True
)
display(curve.round(4))

plots.fig_aggregation_curve(curve)
curve.to_csv(C.TABLES_DIR / "aggregation_curve.csv", index=False)

gain = (
    curve.pivot_table(index="model", columns="k", values="accuracy_mean", observed=True)
    .assign(gain_1_to_30=lambda d: d[30] - d[1])
)
print("\\nAccuracy gained by going from 1 to 30 articles per publisher:")
display(gain[["gain_1_to_30"]].round(4))
''')

md("""
## 5. Error structure by reliability level

`mean_error_direction` takes an explicit `over=` argument because the two
choices give materially different numbers and the old code used both.
`over="all"` reproduces the published figures; `over="errors"` is what the
paper's equation actually defines. Pick one and say which in the text.
""")
code('''
err_by_class = metrics.error_rate_by_class(preds)
display(err_by_class.pivot_table(index="class_label", columns="model",
                                 values="error_rate", observed=True).round(3))
plots.fig_error_rate_by_class(err_by_class)

med_all    = metrics.mean_error_direction(preds, ["model", "y_true"], over="all")
med_errors = metrics.mean_error_direction(preds, ["model", "y_true"], over="errors")

print("Denominator = all articles (reproduces the published figure):")
display(med_all.pivot_table(index="y_true", columns="model", values="med", observed=True).round(2))
print("Denominator = misclassified only (the paper's stated formula):")
display(med_errors.pivot_table(index="y_true", columns="model", values="med", observed=True).round(2))

plots.fig_med_by_class(med_all)
err_by_class.to_csv(C.TABLES_DIR / "error_rate_by_class.csv", index=False)
pd.concat([med_all, med_errors]).to_csv(C.TABLES_DIR / "med_by_class.csv", index=False)
''')

md("""
## 6. Political leaning

### 6.1 Look at the corpus first

Reliability and leaning are strongly associated in the media ecosystem itself.
That association is the confound: an apparent political bias in the errors may
be a reliability effect in disguise. Read this table before the tests.
""")
code('''
crosstab = bias.leaning_by_reliability_crosstab(preds)
display(crosstab)
plots.fig_leaning_distribution(crosstab)
crosstab.to_csv(C.TABLES_DIR / "leaning_by_reliability.csv")
''')

md("### 6.2 Error rates and error direction by leaning")
code('''
er_leaning = bias.error_rate_by_leaning(preds)
display(er_leaning.pivot_table(index="leaning", columns="model",
                               values="error_rate", observed=True).round(3))
plots.fig_error_rate_by_leaning(er_leaning)

med_leaning = bias.med_by_leaning(preds, over="errors")
display(med_leaning.pivot_table(index="leaning", columns="model",
                                values="med", observed=True).round(3))
plots.fig_med_by_leaning(med_leaning)

er_leaning.to_csv(C.TABLES_DIR / "error_rate_by_leaning.csv", index=False)
med_leaning.to_csv(C.TABLES_DIR / "med_by_leaning.csv", index=False)
''')

md("""
### 6.3 Is the association real? Three tests, increasing in rigour

The published chi-square treated 19,476 articles as independent observations.
They are not: leaning and reliability are properties of the **publisher**, so
there are really 439 clusters. That inflates chi-square roughly by the average
cluster size, which is why the original statistic was in the thousands.

Three versions are computed so the effect of the correction is visible:

1. **article level, uncorrected** — reproduces the published table;
2. **publisher level** — one observation per publisher, so independence holds;
3. **logistic regression with publisher-clustered standard errors**, optionally
   controlling for the true reliability level, which addresses the confound and
   the clustering at the same time.
""")
code('''
chi_article   = bias.chi_square_article_level(preds)
chi_publisher = bias.chi_square_publisher_level(pub)

comparison = pd.concat([chi_article, chi_publisher], ignore_index=True)
display(comparison.round(4))
comparison.to_csv(C.TABLES_DIR / "chi_square_comparison.csv", index=False)
''')
code('''
# Leaning effects with clustered SEs, with and without controlling for reliability.
mixed_raw        = bias.mixed_effects_by_leaning(preds, control_for_reliability=False)
mixed_controlled = bias.mixed_effects_by_leaning(preds, control_for_reliability=True)

leaning_terms = mixed_controlled[mixed_controlled["term"].str.contains("leaning", na=False)]
print("Odds of an error vs the 'Non labelled' reference, controlling for reliability:")
display(leaning_terms[["model", "term", "odds_ratio", "p_value", "ci_low", "ci_high"]].round(4))

pd.concat([mixed_raw, mixed_controlled]).to_csv(
    C.TABLES_DIR / "leaning_logit_clustered.csv", index=False
)
''')

md("""
## 7. Supplementary figures

These do not go in the paper. ECIR 2027 makes **no provision for supplementary
material**: the call mentions only appendices, and those "count toward the page
limit" and must be placed "before the references". So this set is meant for the
external companion document (`Paper/NewIRForGood/supplementary.tex`), hosted on
OSF and cited from the paper -- which is what the rejected TCSS version did.

Five families, 14 figures:

| file | what |
|---|---|
| `cm_{model}.pdf` | article-level confusion matrices, five folds + average |
| `domain_confusion_matrix_{model}.pdf` | publisher-level confusion matrices |
| `cmOnPoliticalLeaning{Model}.pdf` | article-level matrices split by leaning |
| `loss_plot.pdf` | Llama 3 convergence over 20 epochs |
| `training_loss_mistral.pdf` | Mistral training loss across the five folds |

Three of these families were drawn *without* LaTeX in the old code, so they are
rendered outside the PGF backend here too, to match. The last one is heavy --
about 3,900 loss points per fold -- and rendering it through PGF exhausts TeX's
main memory, which is the other reason it must stay on the plain PDF backend.
""")
code('''
from irgood import supplementary as supp

supp_files = supp.generate_all(preds, pub)
print(f"{len(supp_files)} supplementary figures in {C.FIGURES_DIR / 'supplementary'}:")
for f in supp_files:
    print("  ", f)
''')

md("""
## 8. Publish the figures to the paper

Copies `Results/figures/*.pdf` into the paper's `img/` directory, so the LaTeX
source always points at regenerated output instead of a stale file someone saved
by hand. Set `PUBLISH = True` when you want the paper to pick them up.
""")
code('''
import shutil

PUBLISH = False   # flip to True to overwrite the paper's figures

if PUBLISH:
    C.PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
    # Paper figures and supplementary figures both land in the paper's img/,
    # which is what main.tex and supplementary.tex expect.
    sources = sorted(C.FIGURES_DIR.glob("*.pdf")) + \
              sorted((C.FIGURES_DIR / "supplementary").glob("*.pdf"))
    for pdf in sources:
        shutil.copy2(pdf, C.PAPER_IMG_DIR / pdf.name)
        print("->", C.PAPER_IMG_DIR / pdf.name)
else:
    print("PUBLISH is False. Figures produced in this run:")
    for pdf in sorted(C.FIGURES_DIR.glob("*.pdf")):
        print("  ", pdf.name)
''')


def main() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Every code cell must at least parse. Cell sources travel through a
    # triple-quoted literal here, so an unescaped "\\n" inside a string in a
    # cell silently becomes a real newline and breaks it. That failure only
    # surfaces when the notebook is executed, which is far too late.
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        try:
            compile(cell["source"], f"cell{i}", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} does not parse: {exc}") from exc

    out = Path(__file__).with_name("paper_results.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(cells)} cells, all parse)")


if __name__ == "__main__":
    main()
