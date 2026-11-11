"""Shared settings for all steps: data folders, random seeds, NewsGuard levels, small helpers."""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42                 # the seed of everything: sampling, inner folds, models, batches, bootstrap
SPLIT_SEED = 1            # the seed of the outlet split (same as the first split of attempt 18)
LEVELS = ["l1", "l2", "l3", "l4", "l5"]
BOUNDS = [39.5, 59.5, 74.5, 99.5]          # NewsGuard score -> level: 0-39, 40-59, 60-74, 75-99, 100
POLITICS = "Political news or commentary"

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # needed by cuBLAS for deterministic GPU results


def data_root():
    """The single data folder: Dataset/ at the repo root (on the server ~/John/Dataset), found by walking up from the
    working directory to a Dataset/ folder that contains LEGGIMI.md."""
    for p in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (p / "Dataset" / "LEGGIMI.md").exists():
            return p / "Dataset"
    raise FileNotFoundError("Dataset/ (with LEGGIMI.md) not found above the working directory")


DATA = data_root()
SCRAPING = [DATA / "scraping" / "NuovoScraping_ottobre", DATA / "scraping" / "NuovoScraping_giro3"]   # October 2026
NEWSGUARD = DATA / "etichette" / "newsguard_2026.csv"
OUT = DATA / "tentativi" / "20_FromScratch"                     # everything this attempt produces (outside git)
MODELS_DIR = DATA.parent / "Modelli" / "20_FromScratch"         # the trained model of every fold (outside git)
PREDICTIONS = OUT / "predictions"                               # logits / predicted scores of every article
EMBEDDINGS = OUT / "embeddings"


def level_from_score(score):
    """NewsGuard score (0-100) -> level 0-4 (l1-l5)."""
    return np.digitize(np.asarray(score, dtype=float), BOUNDS)


def majority(values):
    """Majority vote; on ties the lowest level wins."""
    c = pd.Series(values).value_counts()
    return int(c[c == c.max()].index.min())


def log(msg, file=None):
    """Print a timestamped line and append it to a log file."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    if file is not None:
        with open(file, "a") as f:
            f.write(line + "\n")


def environment():
    """Python and library versions and the seeds, to repeat the computation."""
    import importlib
    import platform
    v = {"python": platform.python_version(), "seed": SEED, "split seed": SPLIT_SEED}
    for lib in ["numpy", "pandas", "sklearn", "scipy", "torch", "transformers", "bitsandbytes", "trafilatura"]:
        try:
            v[lib] = importlib.import_module(lib).__version__
        except ImportError:
            pass
    return v
