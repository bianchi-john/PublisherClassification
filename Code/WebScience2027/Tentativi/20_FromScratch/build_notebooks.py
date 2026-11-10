"""Builds the notebooks of attempt 20. A notebook whose code did not change is left untouched (outputs included)."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def notebook(cells, name):
    for c in cells:                           # the parameter cell (UPPERCASE = value) is tagged for papermill
        if c.cell_type == "code" and c.source.split("\n")[0].split("=")[0].strip().isupper() and "import" not in c.source:
            c.metadata["tags"] = ["parameters"]; break
    for c in cells:
        if c.cell_type == "code":
            compile(c.source, name, "exec")
    old = HERE / name
    if old.exists() and [c.source for c in nbf.read(old, as_version=4).cells
                         if "injected-parameters" not in c.metadata.get("tags", [])] == [c.source for c in cells]:
        print(name, "unchanged"); return
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "johnvenv", "display_name": "johnVenv", "language": "python"}
    nbf.write(nb, old)
    print(name, len(cells), "cells")


SETUP = '''import warnings

import numpy as np
import pandas as pd

import common as C

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 90)
C.OUT.mkdir(parents=True, exist_ok=True)
LOG = C.OUT / "log.txt"
print(C.environment())'''

# ------------------------------------------------------------------------------------------------ 1. scraping
notebook([
    md("""# 1. Scraping

`scraper.py` (run with `run_scraper.sh`) downloaded up to 200 articles per political NewsGuard outlet in October 2026,
in two rounds, saving the raw HTML. **Not re-run here**: this notebook reads what the two rounds produced."""),
    code(SETUP + '''
import json

from cleaning import read_status'''),
    code('''for folder in C.SCRAPING:
    cfg = json.load(open(folder / "config.json")); p = cfg["parametri"]
    print(f"{folder.name}: {cfg['testate']} outlets | max {p['MAX_ARTICOLI']} articles and {p['MAX_TENTATIVI']} URLs per outlet, "
          f"min {p['MIN_PAROLE']} words, {p['PAUSA']} s between requests, {p['TEMPO_TESTATA'] // 60} min per outlet")'''),
    code('''status = read_status(C.SCRAPING)
print(f"outlets tried: {len(status):,}; articles over 100 words: {status.articoli.sum():,}")
status.esito.str.replace(r" \\(.*\\)", "", regex=True).value_counts().rename_axis("outcome").to_frame("outlets")'''),
    code('''status.articoli.describe().round(1).to_frame("articles per outlet").T'''),
], "01_scraping.ipynb")

# ------------------------------------------------------------------------------------------------ 2. cleaning
notebook([
    md("""# 2. Cleaning

Political outlets → text re-extracted from the HTML (trafilatura, precision) → English, 100–1,500 words, from 2025 →
no exact or near duplicates across outlets → at least 10 articles per outlet → one outlet per ownership group →
NewsGuard 2026 score and level (l1 < 40, l2 40–59, l3 60–74, l4 75–99, l5 = 100)."""),
    code('''N_JOBS = 32
NEAR_DUPLICATE = 0.5          # Jaccard of word 5-grams
CANDIDATES = 0.3              # MinHash looks for candidate pairs from this similarity (approximate), then >= NEAR_DUPLICATE
MIN_WORDS, MAX_WORDS = 100, 1500
MIN_ARTICLES = 10
FROM_DATE = "2025-01-01"'''),
    code(SETUP + '''
import hashlib

from joblib import Parallel, delayed

import cleaning as CL

steps = []
def step(name, df):
    steps.append({"step": name, "articles": len(df), "outlets": df.domain.nunique()})
    print(f"{name}: {len(df):,} articles, {df.domain.nunique():,} outlets")'''),
    md("## 2.1 Political outlets and text extraction"),
    code('''ng = CL.read_newsguard(C.NEWSGUARD)
outlets = CL.political_outlets(CL.read_status(C.SCRAPING), ng)
cache = C.OUT / "extracted.csv.gz"
if cache.exists():
    raw = pd.read_csv(cache)
else:
    rows = Parallel(n_jobs=N_JOBS)(delayed(CL.extract_outlet)(d, C.SCRAPING) for d in outlets)
    raw = pd.DataFrame([r for rr in rows for r in rr]); raw.to_csv(cache, index=False)
step("articles of political outlets", raw)
art = raw[raw.text.notna()].reset_index(drop=True)
step("with extracted text", art)'''),
    md("## 2.2 Language, length, date"),
    code('''art["language"] = Parallel(n_jobs=N_JOBS, batch_size=200)(delayed(CL.language)(t) for t in art.text)
art = art[art.language == "en"]
step("English", art)
art["words"] = art.text.str.split().str.len()
art = art[art.words.between(MIN_WORDS, MAX_WORDS)]
step(f"{MIN_WORDS}-{MAX_WORDS:,} words", art)
art["date"] = pd.to_datetime(art.date, errors="coerce")
art = art[art.date >= FROM_DATE].reset_index(drop=True).rename_axis("article_id").reset_index()
step(f"published from {FROM_DATE[:4]}", art)'''),
    md("## 2.3 Duplicates and near duplicates (across all outlets, the oldest copy is kept)"),
    code('''art["norm"] = art.text.map(CL.normalize)
art["fingerprint"] = art.norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
art = art.sort_values(["date", "article_id"]).drop_duplicates("fingerprint").sort_values("article_id")
step("no exact duplicates", art)
pairs = CL.similar_pairs(art.reset_index(drop=True), CANDIDATES, N_JOBS)
pairs = pairs[pairs.jaccard >= NEAR_DUPLICATE]
print(f"near-duplicate pairs: {len(pairs):,}, of which across outlets: {(pairs.outlet_a != pairs.outlet_b).sum():,}")
art = art[~art.article_id.isin(CL.copies_to_drop(pairs, art.set_index("article_id").date))]
step(f"no near duplicates (Jaccard >= {NEAR_DUPLICATE})", art)'''),
    md("## 2.4 Enough articles, one outlet per ownership group"),
    code('''n = art.groupby("domain").size()
art = art[art.domain.isin(n.index[n >= MIN_ARTICLES])]
step(f"outlets with >= {MIN_ARTICLES} articles", art)
art["group"] = art.domain.map(CL.ownership_groups(sorted(art.domain.unique()), ng))
kept = (art.groupby("domain").agg(group=("group", "first"), n=("article_id", "size")).reset_index()
        .sort_values(["n", "domain"], ascending=[False, True]).groupby("group").head(1).domain)
art = art[art.domain.isin(kept)]
step("one outlet per ownership group", art)'''),
    md("## 2.5 Labels and dataset"),
    code('''art["score"] = art.domain.map(ng.Score).astype(float)
art["label"] = C.level_from_score(art.score)
art["level"] = art.label.map(dict(enumerate(C.LEVELS)))
outlets = art.groupby("domain").agg(score=("score", "first"), label=("label", "first"), level=("level", "first"),
                                    articles=("article_id", "size"))
outlets["orientation"] = outlets.index.map(ng.Orientation).fillna("none")
art[["article_id", "domain", "url", "title", "date", "author", "text", "score", "label", "level"]] \\
    .to_csv(C.OUT / "articles.csv.gz", index=False)
outlets.to_csv(C.OUT / "outlets.csv")
pd.DataFrame(steps).set_index("step")'''),
    code('''pd.DataFrame({"outlets": outlets.level.value_counts(), "articles": art.level.value_counts()}).reindex(C.LEVELS).T'''),
    md("## Check: same dataset as attempt 16 (same rules, same data)"),
    code('''old = C.DATA / "tentativi" / "16_DataPreparation" / "articoli.csv.gz"
if old.exists():
    o = pd.read_csv(old)
    m = art.merge(o, on="article_id", suffixes=("", "_16"))
    print(f"articles {len(art):,} vs {len(o):,}; outlets {art.domain.nunique()} vs {o.domain.nunique()}; same ids: {set(art.article_id) == set(o.article_id)}")
    print({c: f"{(m[c].astype(str) == m[c16].astype(str)).mean():.2%}" for c, c16 in
           [("domain", "domain_16"), ("text", "text_16"), ("score", "score_16"), ("label", "label_16")]})'''),
], "02_cleaning.ipynb")

# ------------------------------------------------------------------------------------------------ 3. anonymization
notebook([
    md("""# 3. Anonymization, part 1: rules

The first of two parts (the second, with boilerplate and a language model, is `03b_anonymization_llm`; the checks are
in `03c_anonymization_checks`). Six rules, in order: outlet signature in the titles → `[OUTLET]`; e-mails → `[EMAIL]`; links → `[URL]`;
@accounts → `[ACCOUNT]`; the outlet's own authors → `[AUTHOR]`; the outlet name from its domain → `[OUTLET]`
(see `anonymization.py`). The checks are automatic."""),
    code(SETUP + '''
import re

import anonymization as AN'''),
    code('''art = pd.read_csv(C.OUT / "articles.csv.gz")
anon, counts, found = AN.anonymize(art)
RULES = ["signature", "email", "url", "account", "author", "name"]
print(f"articles with at least one replacement: {((counts[RULES].sum(axis=1) + counts[[f'title_{r}' for r in RULES]].sum(axis=1)) > 0).mean():.1%}")
pd.DataFrame({"articles touched (%)": [(counts[r] > 0).mean() * 100 for r in RULES],
              "replacements": [counts[r].sum() for r in RULES],
              "titles touched (%)": [(counts[f"title_{r}"] > 0).mean() * 100 for r in RULES]}, index=RULES).round(1)'''),
    code('''print("common-word names:", ", ".join(sorted(found["common"])))
print("written in lowercase by their own outlet:", ", ".join(sorted(found["lowercase"])))
print(f"signatures: {len(found['signatures'])} outlets; own authors: {sum(map(len, found['authors'].values())):,}")
for token in ["[OUTLET]", "[AUTHOR]"]:
    for e in AN.examples(anon.text, re.compile(re.escape(token)), n=3):
        print(" ·", e)'''),
    md("## Check: nothing left"),
    code('''left = AN.leftovers(anon, found)
distinct = left[~left.common_word]
print(f"distinctive names ({len(distinct)} outlets): articles with the name as a word {distinct.left_as_word.sum()}, "
      f"inside other words {distinct.left_inside_words.sum()}")
assert distinct.left_as_word.sum() == 0
left[left.common_word | (left.left_inside_words > 0)]'''),
    code('''anon[["article_id", "domain", "title", "date", "text", "score", "label", "level"]].to_csv(C.OUT / "articles_rules.csv.gz", index=False)
left.to_csv(C.OUT / "anonymization_check.csv", index=False)
print("saved", C.OUT / "articles_rules.csv.gz", "(part 1; the final text is written by 03b)")'''),
], "03_anonymization.ipynb")

# ------------------------------------------------------------------------------------------------ 4. split
notebook([
    md("""# 4. Split

One split for both tasks, generated once: outlets sorted by exact score, in blocks of 5 consecutive outlets each one
goes to a different fold (seed `SPLIT_SEED`). It is balanced by score and, since levels are score ranges, by level."""),
    code(SETUP + '''
from models import score_split'''),
    code('''outlets = pd.read_csv(C.OUT / "outlets.csv", index_col="domain")
fold = score_split(outlets.score, C.SPLIT_SEED)
fold.rename("fold").rename_axis("domain").to_csv(C.OUT / "split.csv")
pd.crosstab(fold.rename("fold"), outlets.level)'''),
    code('''outlets.assign(fold=fold).groupby("fold").agg(outlets=("score", "size"), mean_score=("score", "mean"),
                                              sd_score=("score", "std"), articles=("articles", "sum")).round(1)'''),
    code('''old = C.DATA / "tentativi" / "18_PunteggioEsatto" / "split.csv"
if old.exists():
    print("same as the first split of attempt 18:", bool((pd.read_csv(old, index_col=0).split1.reindex(fold.index) == fold).all()))'''),
], "04_split.ipynb")

# ------------------------------------------------------------------------------------------------ 5. text representations (GPU)
notebook([
    md("""# 5. Frozen text representations (GPU)

Every article must fit whole: token counts for every tokenizer, no article above the limit. Then the vectors of
Qwen3-Embedding-8B (instruction + article, last token) and Mistral-Small-24B Base (4 bit, mean of the tokens, layers 27
and 40; the linear models use 27)."""),
    code('''QWEN = "Qwen/Qwen3-Embedding-8B"
MISTRAL = "mistralai/Mistral-Small-24B-Base-2501"
BERT = "answerdotai/ModernBERT-base"
QWEN_BUDGET = 32768           # tokens per batch
MISTRAL_BUDGET = 8192         # smaller: all hidden states stay in memory
GPU_GB = 22'''),
    code(SETUP + '''

import models as MO

art, _ = MO.load()
texts = art.text.fillna("").tolist()
C.EMBEDDINGS.mkdir(parents=True, exist_ok=True)'''),
    code('''rows = []
for name, model, prefix in [("ModernBERT", BERT, ""), ("Qwen3-8B", QWEN, f"Instruct: {MO.QWEN_INSTRUCTION}\\nQuery:"), ("Mistral-24B", MISTRAL, "")]:
    n = MO.count_tokens(texts, model, prefix)
    limit = MO.MAX_TOKENS + (64 if prefix else 0)
    rows.append({"model": name, "median": int(np.median(n)), "99%": int(np.percentile(n, 99)), "max": int(n.max()),
                 "limit": limit, "truncated": int((n > limit).sum())})
tokens = pd.DataFrame(rows).set_index("model"); tokens.to_csv(C.OUT / "token_lengths.csv")
assert tokens.truncated.sum() == 0
tokens'''),
    code('''f = C.EMBEDDINGS / MO.EMBEDDING_FILES["Qwen3-8B"]
if not f.exists():
    MO.wait_for_gpu(GPU_GB, LOG)
    np.save(f, MO.qwen_embeddings(texts, QWEN, QWEN_BUDGET, LOG))
f = C.EMBEDDINGS / MO.EMBEDDING_FILES["Mistral-24B"]
if not f.exists():
    MO.wait_for_gpu(GPU_GB, LOG)
    E = MO.mistral_embeddings(texts, MISTRAL, [27, 40], MISTRAL_BUDGET, LOG, C.EMBEDDINGS / "mistral_partial.npy")
    np.save(C.EMBEDDINGS / "mistral_small_24b_layer40.npy", E[40]); np.save(f, E[27])
    (C.EMBEDDINGS / "mistral_partial.npy").unlink(missing_ok=True)
art[["article_id"]].to_csv(C.EMBEDDINGS / "article_order.csv", index=False)
for name, file in MO.EMBEDDING_FILES.items():
    E = np.load(C.EMBEDDINGS / file); print(name, E.shape, "finite:", np.isfinite(E).all())'''),
], "05_embeddings.ipynb")

# ------------------------------------------------------------------------------------------------ 6-7. classification and regression
for number, task, what in [(6, "classification", "the 5 NewsGuard levels (logistic regression with balanced classes; "
                            "ModernBERT with balanced cross-entropy)"),
                           (7, "regression", "the exact NewsGuard score 0–100 (ridge; ModernBERT with squared error on "
                            "score/100)")]:
    notebook([
        md(f"""# {number}. {task.capitalize()}

Predict {what}. Same split, same models for both tasks. Linear models: the penalty is chosen with 3 inner folds of the
training outlets. ModernBERT: 5 epochs, lr 3e-5, effective batch 32, last epoch kept. For every test article the
outputs are saved in `predictions/{task}/<model>/`, and every fold model in `Modelli/20_FromScratch/`."""),
        code(f'''TASK = "{task}"
MODELS = "TF-IDF,Qwen3-8B,Mistral-24B"     # "ModernBERT" runs on the GPU
N_JOBS = 16'''),
        code(SETUP + '''

import models as MO

art, outlets = MO.load()'''),
        code('''for name in MODELS.split(","):
    for k in range(1, 6):
        if name == "ModernBERT":
            MO.bert_fold(TASK, art, k, "answerdotai/ModernBERT-base", name, LOG)
        else:
            MO.linear_fold(TASK, name, art, k, X=None if name == "TF-IDF" else MO.load_embeddings(name, art),
                           n_jobs=N_JOBS, logfile=LOG)'''),
        code('''for name in MODELS.split(","):
    folder = C.PREDICTIONS / TASK / name
    if name == "ModernBERT":
        print(pd.concat([pd.read_csv(f).assign(fold=f.stem[-1]) for f in sorted(folder.glob("loss_fold*.csv"))])
              .pivot(index="epoch", columns="fold", values="loss").round(4))
    else:
        p = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("penalty_fold*.csv"))])
        print(name, "chosen penalty per fold:", p[p.chosen].set_index("fold").penalty.to_dict())'''),
    ], f"0{number}_{task}.ipynb")

# ------------------------------------------------------------------------------------------------ 8. evaluation
notebook([
    md("""# 8. Evaluation

Per outlet, from the saved article outputs. The two approaches in the same two spaces:

| | levels | score |
|---|---|---|
| classifier | majority level | expected score (probabilities × mean training score of each level) |
| regressor | NewsGuard bounds · training quotas · article vote | mean score · score remapped to training quantiles |
| no text | most frequent level | mean training score |

95% intervals: bootstrap over outlets (fixed seed). Classifier vs regressor: paired, on the same outlets."""),
    code('''MODELS = "TF-IDF,Qwen3-8B,Mistral-24B,ModernBERT"
B = 2000'''),
    code(SETUP + '''

import metrics as ME
import models as MO

art, outlets = MO.load()'''),
    code('''VIEWS = {}                                         # (model, approach, view) -> (space, true, predicted)
for name in MODELS.split(","):
    pa = MO.predictions("classification", name)
    if pa is not None:
        T = ME.outlets_classifier(pa, outlets)
        VIEWS.update({(name, "classifier", v): x for v, x in ME.classifier_views(T, outlets).items()})
    pa = MO.predictions("regression", name)
    if pa is not None:
        T = ME.outlets_regressor(pa, outlets)
        VIEWS.update({(name, "regressor", v): x for v, x in ME.regressor_views(T, pa, art).items()})
VIEWS.update({("no text", "baseline", v): x for v, x in ME.baseline_views(outlets).items()})
rows = [{"model": m, "approach": a, "view": v, "space": s, **ME.METRICS[s](y, p)} for (m, a, v), (s, y, p) in VIEWS.items()]
all_metrics = pd.DataFrame(rows); all_metrics.to_csv(C.OUT / "all_metrics.csv", index=False)
print(f"{len(all_metrics)} views; models: {sorted({m for m, _, _ in VIEWS})}")'''),
    code('''all_metrics[all_metrics.space == "levels"].dropna(axis=1, how="all").set_index(["model", "approach", "view"]).round(3)'''),
    code('''all_metrics[all_metrics.space == "score"].dropna(axis=1, how="all").set_index(["model", "approach", "view"]).round(3)'''),
    md("## Intervals"),
    code('''KEYS = {"levels": ["F1 macro", "tolerant (±1 level)", "quadratic kappa", "Spearman", "reliable: accuracy"],
        "score": ["MAE", "Spearman", "within 10 points", "within 20 points", "reliable: accuracy"]}
rows = []
for (m, a, v), (s, y, p) in VIEWS.items():
    ci, value = ME.bootstrap(y, p, s, KEYS[s], B=B), ME.METRICS[s](y, p)
    rows += [{"model": m, "approach": a, "view": v, "space": s, "metric": c, "value": value[c], "low": ci[c][0],
              "high": ci[c][1]} for c in KEYS[s]]
intervals = pd.DataFrame(rows); intervals.to_csv(C.OUT / "intervals.csv", index=False)
print(len(intervals), "intervals")'''),
    md("## Classifier vs regressor (regressor − classifier, same outlets)"),
    code('''PAIRS = {"levels": ("majority level", ["levels: NewsGuard bounds", "levels: training quotas", "levels: article vote"]),
         "score": ("expected score", ["mean score", "remapped score"])}
rows = []
for name in MODELS.split(","):
    for space, (cls_view, reg_views) in PAIRS.items():
        if (name, "classifier", cls_view) not in VIEWS or (name, "regressor", reg_views[0]) not in VIEWS:
            continue
        for reg_view in reg_views:
            for key in KEYS[space]:
                d, (lo, hi) = ME.bootstrap_difference(VIEWS[(name, "regressor", reg_view)], VIEWS[(name, "classifier", cls_view)], key, B=B)
                rows.append({"model": name, "space": space, "regressor view": reg_view, "metric": key,
                             "difference": d, "low": lo, "high": hi, "not chance": not (lo <= 0 <= hi)})
pairs = pd.DataFrame(rows); pairs.to_csv(C.OUT / "classifier_vs_regressor.csv", index=False)
pairs.round(3)'''),
    md("## Probability metrics of the classifiers"),
    code('''pd.DataFrame({name: ME.probability_metrics(ME.outlets_classifier(MO.predictions("classification", name), outlets))
              for name in MODELS.split(",") if MO.predictions("classification", name) is not None}).T.round(3)'''),
], "08_evaluation.ipynb")

# ------------------------------------------------------------------------------------------------ 9. results
notebook([
    md("""# 9. Results at a glance

226 anonymous outlets, one split by score, 5 folds; every number is per outlet with its 95% interval (bootstrap over
outlets). Darker = better (for MAE and errors, lighter = better)."""),
    code('''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

pd.set_option("display.width", 220)
all_metrics = pd.read_csv(C.OUT / "all_metrics.csv")
intervals = pd.read_csv(C.OUT / "intervals.csv")
pairs = pd.read_csv(C.OUT / "classifier_vs_regressor.csv")


def table(space, metrics, lower_is_better=()):
    """Model × view, coloured column by column."""
    d = all_metrics[all_metrics.space == space].set_index(["model", "approach", "view"])[metrics]
    st = d.style.format("{:.3f}")
    for c in metrics:
        st = st.background_gradient(cmap="Blues_r" if c in lower_is_better else "Blues", subset=[c])
    return st


def with_intervals(space):
    d = intervals[intervals.space == space]
    d = d.assign(text=[f"{v:.3f} [{a:.3f}; {b:.3f}]" if abs(v) < 2 else f"{v:.1f} [{a:.1f}; {b:.1f}]"
                       for v, a, b in zip(d.value, d.low, d.high)])
    return d.pivot_table(index=["model", "approach", "view"], columns="metric", values="text", aggfunc="first")'''),
    md("## Levels"),
    code('''table("levels", ["F1 macro", "tolerant (±1 level)", "quadratic kappa", "Spearman", "reliable: accuracy", "accuracy",
                 "mean level error"], lower_is_better=["mean level error"])'''),
    code('''with_intervals("levels")'''),
    md("## Score"),
    code('''table("score", ["MAE", "Spearman", "within 10 points", "within 20 points", "reliable: accuracy", "R²", "Lin's CCC"],
      lower_is_better=["MAE"])'''),
    code('''with_intervals("score")'''),
    md("## Classifier or regressor? (regressor − classifier; ✓ = interval without 0; for MAE negative = regressor better)"),
    code('''p = pairs.assign(text=[f"{d:+.3f} [{a:+.3f}; {b:+.3f}]{' ✓' if n else ''}" for d, a, b, n in
                       zip(pairs.difference, pairs.low, pairs.high, pairs["not chance"])])
p.pivot_table(index=["space", "metric", "regressor view"], columns="model", values="text", aggfunc="first")'''),
    md("## In one figure"),
    code('''ROWS = [("levels", ["F1 macro", "tolerant (±1 level)"]), ("score", ["MAE", "Spearman"])]
COLOUR = {"classifier": "#2a78d6", "regressor": "#eb6834", "baseline": "#8a8983"}
n = [intervals[intervals.space == s].groupby(["model", "view"]).ngroups for s, _ in ROWS]
fig, axes = plt.subplots(2, 2, figsize=(13, 0.32 * sum(n) + 2), gridspec_kw={"height_ratios": n})
for r, (space, metrics) in enumerate(ROWS):
    for c, metric in enumerate(metrics):
        ax, d = axes[r, c], intervals[(intervals.space == space) & (intervals.metric == metric)].reset_index(drop=True)
        y = np.arange(len(d))
        ax.errorbar(d.value, y, xerr=[d.value - d.low, d.high - d.value], fmt="none", ecolor="#b5b3ab", lw=1.5, capsize=2)
        ax.scatter(d.value, y, s=40, c=[COLOUR[a] for a in d.approach], zorder=3)
        ax.set_yticks(y, [f"{m} · {v}" for m, v in zip(d.model, d.view)] if c == 0 else [""] * len(y), fontsize=8)
        ax.invert_yaxis(); ax.set_title(f"{space}: {metric}", fontsize=10, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e1e0d9"); ax.spines[["top", "right"]].set_visible(False)
for a, col in COLOUR.items():
    axes[0, 0].scatter([], [], c=col, label=a)
fig.legend(loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(); plt.savefig(C.OUT / "results.png", dpi=150, bbox_inches="tight"); plt.show()'''),
    md("## In short"),
    code('''def best(space, metric, lower=False):
    d = all_metrics[(all_metrics.space == space) & (all_metrics.approach != "baseline")]
    r = d.loc[d[metric].idxmin() if lower else d[metric].idxmax()]
    base = all_metrics[(all_metrics.space == space) & (all_metrics.approach == "baseline")][metric].iloc[0]
    return f"- {space}, {metric}: {r.model} ({r.approach}, {r.view}) = {r[metric]:.3f}; without text {base:.3f}"


print("\\n".join([best("levels", "F1 macro"), best("levels", "tolerant (±1 level)"), best("levels", "Spearman"),
                 best("score", "MAE", lower=True), best("score", "Spearman"), best("score", "within 10 points")]))
print(f"classifier/regressor differences that are not chance: {pairs['not chance'].sum()} of {len(pairs)}")'''),
], "09_results.ipynb")

# ------------------------------------------------------------------------------------------------ 10. more strategies on the frozen embeddings
notebook([
    md("""# 10. More strategies on the frozen embeddings (added 6/11)

**Question.** Starting from the same frozen vectors (Qwen3-Embedding-8B, Mistral-Small-24B), can another regressor or
another way of training predict the NewsGuard score better than the ridge of step 7? TF-IDF stays as it is (step 7) and
is the reference; ModernBERT is not touched.

**Fixed before the results** (details in `boost.py`):
- same outlets, same 5 outer folds, same 3 inner folds and inner criterion (Spearman per outlet) as step 7;
- representations: Qwen3-8B, Mistral-24B layer 27 (step 7), layer 40, Qwen ⊕ Mistral, Mistral 27 ⊕ 40;
- methods: ridge (step 7), ridge with every outlet counting the same, with rare scores counting more (LDS), on the
  score quantile; ridge, mixup and RBF kernel ridge on the outlet (mean vector and random bags of articles); RBF kernel
  ridge and a small neural network on the articles; attention over the articles of an outlet (multiple-instance);
  then the mean of Qwen and Mistral and «auto» (the method with the best inner score, fold by fold);
- every setting chosen on the inner folds only; the stretched score uses the training scores only; every outlet is
  predicted from all its articles and from k random articles (20 draws, the same for all, TF-IDF included);
- paired bootstrap over outlets (1,000, seed 7) against TF-IDF and against the ridge of step 7.

**Honest reading.** Many rows are compared: the best row of a table is optimistic. The numbers to quote are those of
«auto», which choose the method without looking at the test outlets.

Run on the server: `run_boost.sh` (one process per representation and method, with a time limit). This notebook only
reads the saved predictions."""),
    code(SETUP + '''
import matplotlib.pyplot as plt

import boost as BO
import boost_eval as BE

art, outlets, rows = BO.load()'''),
    md("## What has been computed"),
    code('''done = BO.collect().groupby(["embedding", "method"]).domain.nunique().unstack()
print("outlets predicted (226 = all 5 folds):"); display(done.reindex(columns=BO.METHODS))
err = BO.OUT / "errors.txt"
print(open(err).read()[-3000:] if err.exists() else "no errors")'''),
    md("## Settings chosen on the inner folds (best inner Spearman, mean over the 5 outer folds)"),
    code('''inner = BE.inner_choices()
inner.pivot_table(index="method", columns="embedding", values="inner Spearman").reindex(BO.METHODS).round(3)'''),
    code('''inner.sort_values(["embedding", "method", "fold"])[["embedding", "method", "fold", "inner Spearman", "chosen"]]'''),
    md("## Results"),
    code('''S = BE.sources(outlets, art, rows)
R = BE.evaluate(S, outlets)
R.to_csv(BO.OUT / "results.csv", index=False)
print(S.source.nunique(), "sources")'''),
    code('''def table(k="all", score="stretched"):
    """✓ = significantly better than TF-IDF; ▲ = significantly better than the ridge of step 7 (same representation;
    Mistral-24B for the averages and auto)."""
    d = R[(R.k == k) & (R.score == score)].copy()
    d["text"] = [f"{v:.3f}" if m in ("Spearman", "Kendall") else f"{v:.1f}" for v, m in zip(d.value, d.measure)]
    d["text"] += np.where(d.get("beats TF-IDF", False) == True, " ✓", "") + np.where(d.get("beats ridge", False) == True, " ▲", "")
    t = d.pivot_table(index="source", columns="measure", values="text", aggfunc="first")[BE.MEASURES]
    order = d[d.measure == "Spearman"].set_index("source").value.sort_values(ascending=False).index
    return t.reindex(order)'''),
    md("### All articles, stretched score"),
    code('''table("all", "stretched")'''),
    md("### All articles, predicted score (not stretched)"),
    code('''table("all", "normal")'''),
    md("### 5 articles per outlet, stretched score"),
    code('''table("5", "stretched")'''),
    md("### 1 article per outlet, stretched score"),
    code('''table("1", "stretched")'''),
    md("## Every source with all articles (stretched)"),
    code('''d = R[(R.k == "all") & (R.score == "stretched")]
fig, axes = plt.subplots(1, 2, figsize=(14, 0.22 * d.source.nunique() + 1.5), sharey=True)
order = d[d.measure == "Spearman"].set_index("source").value.sort_values().index
for ax, m in zip(axes, ["Spearman", "MAE"]):
    x = d[d.measure == m].set_index("source").reindex(order)
    colour = np.where(x["beats TF-IDF"] == True, "#1d4f91", np.where(x.index == "TF-IDF", "#eb6834", "#8a8983"))
    ax.scatter(x.value, range(len(x)), c=colour, s=25)
    tf = x.loc["TF-IDF", "value"]; ax.axvline(tf, color="#eb6834", lw=1, ls="--")
    ax.set_title(f"{m} (blue: better than TF-IDF, significant)", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#e1e0d9")
axes[0].set_yticks(range(len(order)), order, fontsize=7)
plt.tight_layout(); plt.savefig(BO.OUT / "all_sources.png", dpi=150, bbox_inches="tight"); plt.show()'''),
    md("## Curves: TF-IDF, the ridge of step 7, auto and the three methods with the best inner score"),
    code('''top = inner.groupby(["embedding", "method"])["inner Spearman"].mean().sort_values(ascending=False).head(3)
show = ["TF-IDF", "Qwen3-8B · ridge", "Mistral-24B · ridge", "auto · all"] + [f"{e} · {m}" for e, m in top.index]
X = [1, 2, 3, 5, 10, 20, 50, 160]
fig, axes = plt.subplots(1, 4, figsize=(19, 5))
for ax, m in zip(axes, BE.MEASURES):
    for s in show:
        d = R[(R.source == s) & (R.score == "stretched") & (R.measure == m)].set_index("k").reindex([str(k) for k in BO.K])
        if d.value.isna().all():
            continue
        ax.plot(X, d.value, "o-", ms=3, lw=1.6, label=s)
        win = d["beats TF-IDF"].fillna(False).to_numpy(bool)
        ax.scatter(np.array(X)[win], d.value.to_numpy()[win], s=70, facecolor="none", edgecolor="k", zorder=4)
    ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in BO.K]); ax.minorticks_off()
    ax.set_title(f"{m} (stretched)", loc="left", fontsize=11, fontweight="bold"); ax.set_xlabel("articles per outlet")
    ax.grid(color="#e1e0d9")
axes[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(BO.OUT / "curves.png", dpi=150, bbox_inches="tight"); plt.show()'''),
    md("## In short"),
    code('''for k in ["all", "5", "1"]:
    for score in ["stretched", "normal"]:
        d = R[(R.k == k) & (R.score == score)]
        line = []
        for s in ["TF-IDF", "Mistral-24B · ridge", "auto · Mistral-24B", "auto · all"]:
            x = d[d.source == s].set_index("measure").value
            if len(x):
                line.append(f"{s}: ρ {x['Spearman']:.3f}, MAE {x['MAE']:.1f}")
        print(f"k={k:>3}, {score:9}  " + " | ".join(line))
print()
for name in ["TF-IDF", "ridge"]:
    col = f"beats {name}"
    n = R[(R.k == "all") & (R[col] == True)].groupby(["score", "measure"]).size()
    print(f"sources significantly better than {name} with all articles:"); print(n.to_string(), "\\n")'''),
], "10_boost.ipynb")

# ------------------------------------------------------------------------------------------------ 11. meta-articles
notebook([
    md("""# 11. Meta-articles (added 8/11)

**Question.** If every training example is a *meta-article* (the leads of many articles of ONE outlet in one text) and
every outlet has the same number of them, do the main models (TF-IDF, Qwen3-8B, Mistral-24B, ModernBERT) predict the
NewsGuard score better than when they read single articles?

**Fixed before the results** (details in `meta.py`): a meta-article = leads (first 120 words) of articles of the outlet in
random order up to 2,400 words; 10 per outlet, built from the outlet's own articles only, the same way for training and
test outlets. Same outer folds, inner folds and inner criterion as before. Frozen models: ridge and the small neural
network of step 10. ModernBERT: fresh meta-articles every epoch (6 per outlet), mean pooling, Huber loss, lr 3e-5,
at most 10 epochs, epoch chosen on 20% of the training outlets kept apart. Outlet score = mean over its 10 meta-articles.

**Compared with** the same model on whole articles (steps 7 and 10), all articles, on the same 226 outlets; paired
bootstrap over outlets (1,000, seed 7). Note: meta-articles read only the leads (about 2,400 words per meta-article,
24,000 per outlet), single articles are read in full."""),
    code(SETUP + '''
import matplotlib.pyplot as plt

import boost as BO
import boost_eval as BE
import meta as ME

art, outlets, rows = BO.load()
M, mrows = ME.load_meta()
print(f"{len(M):,} meta-articles; articles per meta-article: median {M.articles.median():.0f}; words: median {M.words.median():.0f}")
done = {m: len(list((ME.PRED / m).glob("fold*.csv.gz"))) for m in ME.MODELS}
print("folds saved:", done)
err = ME.OUT / "errors.txt"
print(open(err).read()[-3000:] if err.exists() else "no errors")'''),
    md("## Meta-articles against single articles (all articles / all 10 meta-articles)"),
    code('''idx = np.vstack([np.arange(len(outlets))[None, :], np.random.default_rng(7).integers(0, len(outlets), (BE.B, len(outlets)))])
V = {}
for name in [m for m in ME.MODELS if done[m] == 5]:
    V[(name, "meta")] = BE._one(ME.meta_matrix(name, outlets, "all"), name, "all", outlets, idx)[2]
    V[(name, "articles")] = BE._one(ME.article_matrix(name, outlets, art, rows), name, "all", outlets, idx)[2]
rows_out = []
for (name, how), res in V.items():
    for (score, mes), v in res.items():
        r = {"model": name, "input": how, "score": score, "measure": mes, "value": v[0]}
        other = V.get((name, "articles"))
        if how == "meta" and other is not None:
            d = v - other[(score, mes)]; lo, hi = np.percentile(d[1:], [2.5, 97.5])
            r.update({"meta − articles": d[0], "low": lo, "high": hi,
                      "better": bool(hi < 0) if mes in BE.LOWER else bool(lo > 0), "worse": bool(lo > 0) if mes in BE.LOWER else bool(hi < 0)})
        tf = V.get(("TF-IDF", "articles"))
        if tf is not None and (name, how) != ("TF-IDF", "articles"):
            d = v - tf[(score, mes)]; lo, hi = np.percentile(d[1:], [2.5, 97.5])
            r["beats TF-IDF (articles)"] = bool(hi < 0) if mes in BE.LOWER else bool(lo > 0)
        rows_out.append(r)
T = pd.DataFrame(rows_out); T.to_csv(ME.OUT / "results.csv", index=False)'''),
    code('''def table(score):
    """✓ = meta-articles significantly better than the same model on articles, ✗ = significantly worse."""
    d = T[T.score == score].copy()
    d["text"] = [f"{v:.3f}" if m in ("Spearman", "Kendall") else f"{v:.1f}" for v, m in zip(d.value, d.measure)]
    d["text"] += np.where(d.get("better", False) == True, " ✓", "") + np.where(d.get("worse", False) == True, " ✗", "")
    return d.pivot_table(index=["model", "input"], columns="measure", values="text", aggfunc="first")[BE.MEASURES]


table("stretched")'''),
    code('''table("normal")'''),
    code('''T[(T.input == "meta") & T["meta − articles"].notna()].round(3)[["model", "score", "measure", "meta − articles", "low", "high", "better", "worse"]]'''),
    md("## How many meta-articles are needed (stretched score; 1 meta-article ≈ 20 leads)"),
    code('''rows_k = []
for name in [m for m in ME.MODELS if done[m] == 5]:
    for k in ME.K_META:
        res = BE._one(ME.meta_matrix(name, outlets, k), name, str(k), outlets, idx[:1])[2]
        rows_k.append({"model": name, "meta-articles": str(k), **{m: res[("stretched", m)][0] for m in BE.MEASURES}})
pd.DataFrame(rows_k).pivot_table(index="model", columns="meta-articles", values="Spearman")[[str(k) for k in ME.K_META]].round(3)'''),
    md("## ModernBERT: Spearman on the outlets kept apart, epoch by epoch"),
    code('''curves = [pd.read_csv(f) for f in sorted((ME.PRED / "ModernBERT").glob("curve_fold*.csv"))]
if curves:
    C_ = pd.concat(curves)
    fig, ax = plt.subplots(figsize=(8, 4))
    for f, d in C_.groupby("fold"):
        ax.plot(d.epoch, d["validation Spearman"], "o-", label=f"fold {f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("Spearman (outlets kept apart)"); ax.legend(fontsize=8); ax.grid(color="#e1e0d9")
    plt.show()
    display(C_.pivot_table(index="epoch", columns="fold", values="validation Spearman").round(3))'''),
    md("## Settings chosen on the inner folds"),
    code('''for m in ME.MODELS[:-1]:
    for f in sorted((ME.PRED / m).glob("inner_fold*.csv")):
        d = pd.read_csv(f); print(m, f.stem, d.loc[d.chosen].drop(columns=["chosen"]).to_dict("records"))'''),
    md("## In short"),
    code('''for score in ["stretched", "normal"]:
    d = T[(T.score == score) & (T.measure.isin(["Spearman", "MAE"]))].pivot_table(index="model", columns=["measure", "input"], values="value")
    print(score); print(d.round(3).to_string(), "\\n")
s = T[(T.input == "meta") & T["meta − articles"].notna()]
print("meta better than articles (significant):", s.better.sum(), "| worse:", s.worse.sum(), "| of", len(s))'''),
], "11_meta.ipynb")


# ------------------------------------------------------------------------------------------------ 3b. anonymization, part 2
notebook([
    md("""# 3b. Anonymization, part 2: boilerplate and a language model

On the text of part 1 (`articles_rules.csv.gz`), two more passes, so that "the models do not see who published the
article" is hard to attack (details and reasons in `anonymization_llm.py`):
1. **boilerplate**: a sentence repeated in at least 15% of the articles of an outlet (and in at least 3) is removed from
   that outlet (newsletter and donation calls, disclaimers, fixed sign-offs);
2. **language model**: a local instruction model (Qwen2.5-7B-Instruct, greedy) lists, article by article, the phrases that
   reveal the outlet (its name in any form, its own staff). The phrases really in the text that are not common words or
   public names (frequent in OTHER outlets) form a lexicon per outlet, replaced in ALL its articles with `[OUTLET]` /
   `[AUTHOR]`. The model never rewrites the text: every change is a deletion or an exact replacement, listed in files.
The result, `articles_anonymous.csv.gz`, is the text every later step reads. Checks: `03c_anonymization_checks`."""),
    code('''DETECTOR_BUDGET = 24000        # tokens per batch (prompt + answer)'''),
    code(SETUP + '''
import json
import re

import anonymization_llm as AL

AL.WORK.mkdir(parents=True, exist_ok=True)
art = pd.read_csv(C.OUT / "articles_rules.csv.gz")               # part 1
part1 = art.text.fillna("").copy()'''),
    md("## Boilerplate"),
    code('''art["text"], removed = AL.boilerplate(art)
removed.to_csv(AL.WORK / "boilerplate.csv", index=False)
w0, w1 = part1.str.split().str.len(), art.text.str.split().str.len()
print(f"sentences removed: {len(removed):,} in {removed.domain.nunique()} outlets; articles touched {(w1 < w0).mean():.1%}; "
      f"words removed {1 - w1.sum() / w0.sum():.2%}; articles under 50 words after: {(w1 < 50).sum()}")
removed.sort_values("articles", ascending=False).head(15)[["sentence", "articles", "of"]]'''),
    md("## The language model lists the phrases that reveal the outlet"),
    code('''f = AL.WORK / "detector_answers.json"
answers = {int(k): v for k, v in json.load(open(f)).items()} if f.exists() else {}
if len(answers) < len(art):
    import models as MO
    MO.wait_for_gpu(18, AL.LOG)
    tok, m = AL.load_llm(AL.DETECTOR)
    save = lambda d: json.dump({str(k): v for k, v in d.items()}, open(f, "w"))
    answers = AL.generate(tok, m, AL.detection_prompts(art), 160, budget=DETECTOR_BUDGET, done=answers, save=save, logfile=AL.LOG)
    del m; import torch; torch.cuda.empty_cache()
bad = sum(1 for a in answers.values() if not re.search(r"\\{.*\\}", a, flags=re.S))
print(f"answers: {len(answers):,}; without JSON: {bad}")'''),
    code('''L = AL.lexicon(art, answers)
L.to_csv(AL.WORK / "lexicon.csv", index=False)
print(f"phrases proposed: {L.phrase.nunique():,} in {L.domain.nunique()} outlets; kept: {L[L.kept].groupby(['domain', 'phrase']).ngroups:,} "
      f"(outlet {L[L.kept & (L.kind == 'outlet')].phrase.nunique()}, staff {L[L.kept & (L.kind == 'staff')].phrase.nunique()})")
L[~L.kept].sort_values("elsewhere", ascending=False).drop_duplicates("phrase").head(15)[["kind", "phrase", "elsewhere"]]'''),
    code('''art["text"], art["title"], n_llm = AL.apply_lexicon(art, L)
print(f"replacements from the lexicon: {n_llm.sum():,} in {(n_llm > 0).mean():.1%} of the articles")
art[["article_id", "domain", "title", "date", "text", "score", "label", "level"]].to_csv(C.OUT / "articles_anonymous.csv.gz", index=False)
print("saved", C.OUT / "articles_anonymous.csv.gz", "(the text of every later step)")'''),
], "03b_anonymization_llm.ipynb")

# ------------------------------------------------------------------------------------------------ 3c. anonymization checks
notebook([
    md("""# 3c. Anonymization checks

On the original text (step 2), after part 1 (rules) and after part 2 (final text):
- distinctive outlet names left as a word (as in step 3);
- an **LLM attacker** (Mistral-Small-24B-Instruct, 4 bit; larger than the detector) asked which website published the
  article, on 3 articles per outlet (678): share of right domains;
- a **classifier** that tries to recognize the outlet among the 226 (TF-IDF + linear SVM, 80/20 of every outlet's
  articles). The writing style stays (it is what the paper studies), so this does not go to zero;
- the **agreement** of the detector with the larger model on 400 articles: outlet phrases the larger one finds that are
  still in the final text;
- a **sheet for a check by hand** of 100 random articles (`anonymization/audit.html`, to fill `audit.csv`)."""),
    code(SETUP + '''
import json

import anonymization as AN
import anonymization_llm as AL

raw = pd.read_csv(C.OUT / "articles.csv.gz")
part1 = pd.read_csv(C.OUT / "articles_rules.csv.gz")
art = pd.read_csv(C.OUT / "articles_anonymous.csv.gz")
assert (raw.article_id.to_numpy() == art.article_id.to_numpy()).all() and (part1.article_id.to_numpy() == art.article_id.to_numpy()).all()
versions = {"original": raw, "rules (part 1)": part1, "final (part 2)": art}'''),
    code('''_, _, found = AN.anonymize(raw)
for v, d in versions.items():
    left = AN.leftovers(d, found)
    print(f"{v}: distinctive names left as a word: {left[~left.common_word].left_as_word.sum()}")'''),
    code('''S = AL.sample(art)
f = AL.WORK / "attack_answers.json"
att = json.load(open(f)) if f.exists() else {}
if any(v not in att for v in versions):
    import models as MO
    MO.wait_for_gpu(18, AL.LOG)
    try:                               # the attacker is the larger model (stronger); the detector if it is missing
        tok, m = AL.load_llm(AL.CHECKER, four_bit=True); att["attacker"] = AL.CHECKER; budget, mb = 12000, 8
    except Exception as e:
        print("larger model missing, the attacker is the detector:", e)
        tok, m = AL.load_llm(AL.DETECTOR); att["attacker"] = AL.DETECTOR; budget, mb = 24000, 32
    for v, d in versions.items():
        if v not in att:
            prompts = [AL.ATTACK.format(title=str(d.title.fillna("")[i]), text=" ".join(str(d.text.fillna("")[i]).split()[:1500])) for i in S]
            att[v] = [a for _, a in sorted(AL.generate(tok, m, prompts, 12, budget=budget, max_batch=mb).items())]
            json.dump(att, open(f, "w"))
    del m; import torch; torch.cuda.empty_cache()
print("attacker:", att.get("attacker"))
checks = {"LLM attacker names the website": {v: AL.attack_accuracy(att[v], art.domain[S]) for v in versions},
          "classifier recognizes the outlet (226)": {v: AL.outlet_classifier(d.text.fillna("").to_numpy(), art.domain.to_numpy())
                                                     for v, d in versions.items()}}
T = pd.DataFrame(checks).T; T.to_csv(AL.WORK / "checks.csv"); T.round(3)'''),
    code('''f = AL.WORK / "checker_answers.json"
idx = art.sample(n=AL.CHECK_SAMPLE, random_state=C.SEED).index
try:                                   # the larger model is only a check: if it is missing, the step goes on
    if not f.exists():
        import models as MO
        MO.wait_for_gpu(18, AL.LOG)
        tok, m = AL.load_llm(AL.CHECKER, four_bit=True)
        ch = AL.generate(tok, m, AL.detection_prompts(part1.loc[idx].reset_index(drop=True)), 160, budget=12000, max_batch=8)
        json.dump({str(int(idx[k])): v for k, v in ch.items()}, open(f, "w"))
        del m; import torch; torch.cuda.empty_cache()
    checker = {int(k): v for k, v in json.load(open(f)).items()}
    Lc = AL.lexicon(part1, checker)
    phrases = set(map(tuple, Lc[Lc.kept & (Lc.kind == "outlet")][["domain", "phrase"]].to_numpy()))
    left = sorted(p for d, p in phrases if any(AL._word_re(p, True).search(t) for t in art.text[art.domain == d].fillna("")))
    print(f"on {len(idx)} articles the larger model finds {len(phrases)} outlet phrases; still in the final text: {len(left)}")
    pd.DataFrame({"phrase": left}).to_csv(AL.WORK / "checker_left.csv", index=False); print(left[:40])
except Exception as e:
    print("check with the larger model not done:", type(e).__name__, e)'''),
    code('''A = art.sample(n=AL.AUDIT, random_state=C.SEED + 1)
AL.audit_sheet(part1.text.fillna("")[A.index], A.text.fillna(""), A.article_id, A.domain, AL.WORK / "audit.html")
pd.DataFrame({"article_id": A.article_id, "identifiers left": "", "wrong replacements": ""}).to_csv(AL.WORK / "audit.csv", index=False)
print("for the check by hand:", AL.WORK / "audit.html", "and", AL.WORK / "audit.csv")'''),
], "03c_anonymization_checks.ipynb")

# ------------------------------------------------------------------------------------------------ 12. effect of the anonymization
notebook([
    md("""# 12. Does the stronger anonymization change the results?

The pipeline was first run with the rules of part 1 only (outputs kept in `rules_only/`). Here the regressors of step 7
on the final text (parts 1 + 2) are compared with the same regressors on the part-1 text: same outlets, folds,
settings and draws of k articles; paired bootstrap over outlets (1,000, seed 7)."""),
    code(SETUP + '''
import boost as BO
import boost_eval as BE

art, outlets, rows = BO.load()
NEW = C.PREDICTIONS / "regression"
OLD = C.OUT / "rules_only" / "predictions" / "regression"
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B", "ModernBERT"]
done = [m for m in MODELS if len(list((NEW / m).glob("fold*.csv.gz"))) == 5 and len(list((OLD / m).glob("fold*.csv.gz"))) == 5]
print("compared:", done)'''),
    code('''def matrix(folder, name, k):
    p = pd.concat([pd.read_csv(f) for f in sorted((folder / name).glob("fold*.csv.gz"))]).set_index("article_id").prediction
    p = p.reindex(art.article_id).to_numpy()
    draws = [d for f in range(1, 6) for d in BO.test_draws(outlets, rows, f) if d[1] == str(k)]
    t = pd.DataFrame([(d, j, p[b].mean()) for d, _, j, b in draws], columns=["domain", "draw", "prediction"])
    return t.pivot_table(index="draw", columns="domain", values="prediction").reindex(columns=outlets.index).to_numpy()


idx = np.vstack([np.arange(len(outlets))[None, :], np.random.default_rng(7).integers(0, len(outlets), (BE.B, len(outlets)))])
out = []
for m in done:
    for k in ["1", "5", "all"]:
        a = BE._one(matrix(OLD, m, k), m, k, outlets, idx)[2]
        b = BE._one(matrix(NEW, m, k), m, k, outlets, idx)[2]
        for (score, mes), v in b.items():
            d = v - a[(score, mes)]; lo, hi = np.percentile(d[1:], [2.5, 97.5])
            out.append({"model": m, "k": k, "score": score, "measure": mes, "rules only": a[(score, mes)][0], "final": v[0],
                        "difference": d[0], "low": lo, "high": hi,
                        "better": bool(hi < 0) if mes in BE.LOWER else bool(lo > 0), "worse": bool(lo > 0) if mes in BE.LOWER else bool(hi < 0)})
T = pd.DataFrame(out); T.to_csv(C.OUT / "anonymization_effect.csv", index=False)'''),
    code('''def table(k="all", score="stretched"):
    d = T[(T.k == k) & (T.score == score)].copy()
    f = lambda v, m: f"{v:.3f}" if m in ("Spearman", "Kendall") else f"{v:.1f}"
    d["text"] = [f"{f(a, m)} → {f(b, m)}" + (" ✓" if g else " ✗" if w else "") for a, b, m, g, w in
                 zip(d["rules only"], d["final"], d.measure, d.better, d.worse)]
    return d.pivot_table(index="model", columns="measure", values="text", aggfunc="first")[BE.MEASURES].reindex(done)


table("all", "stretched")'''),
    code('''table("all", "normal")'''),
    code('''table("5", "stretched")'''),
    code('''table("1", "stretched")'''),
], "12_anonymization_effect.ipynb")
