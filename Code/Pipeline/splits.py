"""Step 16: how much the results depend on the split of the outlets.

The ridge models of step 7 (TF-IDF, Qwen3-Embedding-8B, Mistral-Small-24B) again with four more splits, made with the
procedure of step 4 (folds stratified on the exact score, `models.score_split`) and the seeds 2-5; the paper's split is
seed 1 (`SPLIT_SEED`). Everything else as in step 7: the same anonymous articles, the frozen vectors of step 5, the
penalty (and for Mistral the layer) chosen on the inner folds. CPU only. Outputs: splits/seed<s>/<model>/ in the data
folder (predictions only, the models are not saved).
"""
import pandas as pd
from scipy.stats import spearmanr

import common as C
import models as MO

SEEDS = [2, 3, 4, 5]
MODELS = ["TF-IDF", "Qwen3-8B", "Mistral-24B"]
DIR = C.OUT / "splits"


def folder(seed, name):
    """Where the predictions of a split are: step 7 for the paper's split, splits/ for the others."""
    return C.PREDICTIONS / "regression" / name if seed == C.SPLIT_SEED else DIR / f"seed{seed}" / name


def run(seed, art, outlets, models=MODELS, n_jobs=16, logfile=None):
    """The models of step 7 on the split of `seed` (the folds already saved are skipped)."""
    fold = MO.score_split(outlets.score, seed)
    a = art.assign(fold=art.domain.map(fold).to_numpy())
    C.log(f"split seed {seed}: outlets per fold {fold.value_counts().sort_index().tolist()}", logfile)
    for name in models:
        X = None if name == "TF-IDF" else MO.load_embeddings(name, art)
        for k in range(1, 6):
            MO.linear_fold("regression", name, a, k, X=X, n_jobs=n_jobs, logfile=logfile, folder=folder(seed, name),
                           models_dest=False)


def results(outlets, seeds=None, models=MODELS):
    """Per split and model: Spearman and MAE per outlet (mean prediction of all its articles), and the difference from
    TF-IDF on the same split."""
    seeds = seeds or [C.SPLIT_SEED] + SEEDS
    rows = []
    for s in seeds:
        for m in models:
            p = MO.predictions("regression", m, folder(s, m))
            if p is not None:
                p = p.groupby("domain").prediction.mean().reindex(outlets.index)
                rows.append({"split seed": s, "model": m, "Spearman": spearmanr(p, outlets.score)[0],
                             "MAE": (p - outlets.score).abs().mean()})
    T = pd.DataFrame(rows)
    ref = T[T.model == "TF-IDF"].set_index("split seed")
    T["Spearman - TF-IDF"] = T.Spearman - T["split seed"].map(ref.Spearman)
    T["MAE - TF-IDF"] = T.MAE - T["split seed"].map(ref.MAE)
    return T
