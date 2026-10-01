"""Single source of truth for paths, model registry and plotting style.

Every other module imports from here, so a path or a colour is defined once.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths. REPO_ROOT is resolved from this file's location, so the notebook works
# regardless of the directory it is launched from.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Inputs (read-only -- never written to by this pipeline).
# Model predictions live under Results/predictions: they are outputs of stage 1
# (1-classification) and inputs to stage 2 (2-analysis).
RESULTS_DIR = REPO_ROOT / "Results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CLASSIFICATION_DIR = PREDICTIONS_DIR / "finetuned"
ZEROSHOT_DIR = PREDICTIONS_DIR / "zeroshot"
FINETUNING_LOG = CLASSIFICATION_DIR / "finetuning_run.log"
MISTRAL_LOSS_LOG = CLASSIFICATION_DIR / "mistral" / "mistralTrainingLoss.log"

GPT_PRED_CSV = ZEROSHOT_DIR / "preds_gpt5mini_capNone.csv"

# The only proprietary input: NewsGuard ratings and political orientation.
# Not redistributable, and excluded from the repository.
NEWSGUARD_CSV = REPO_ROOT / "Dataset" / "newsguard.csv"

# Outputs
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DERIVED_DIR = RESULTS_DIR / "derived"

# Where the paper picks its figures up from.
PAPER_IMG_DIR = REPO_ROOT / "Paper" / "NewIRForGood" / "img"


def ensure_output_dirs() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, DERIVED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model registry.
#
# NOTE: the display name is bound to the file HERE and nowhere else. The old
# `4 - Plotter.ipynb` bound them separately in two places and got them crossed
# (`robertaDf = pd.read_csv('.../llama.csv')`), which is how RoBERTa and Llama
# ended up swapped in the published article-level F1 and tolerant-accuracy
# numbers. Keeping the binding in one table makes that class of bug impossible.
# --------------------------------------------------------------------------
FINETUNED_MODELS: dict[str, dict] = {
    "BERT": {
        "per_fold_csv": CLASSIFICATION_DIR / "bert.csv",
        "enriched_csv": CLASSIFICATION_DIR / "bert_enriched.csv",
    },
    "RoBERTa": {
        "per_fold_csv": CLASSIFICATION_DIR / "roberta.csv",
        "enriched_csv": CLASSIFICATION_DIR / "roberta_enriched.csv",
    },
    "Llama": {
        "per_fold_csv": CLASSIFICATION_DIR / "llama.csv",
        "enriched_csv": CLASSIFICATION_DIR / "llama_enriched.csv",
    },
    "Mistral": {
        "per_fold_csv": CLASSIFICATION_DIR / "mistral" / "24B_results.csv",
        "enriched_csv": CLASSIFICATION_DIR / "mistral" / "24B_results_enriched.csv",
    },
}

GPT_MODEL_NAME = "GPT-5-mini (zero-shot)"

# Order used on every axis and in every table.
MODEL_ORDER = ["BERT", "RoBERTa", "Llama", "Mistral"]
MODEL_ORDER_WITH_GPT = MODEL_ORDER + [GPT_MODEL_NAME]

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
# Reliability labels are stored 0..4 and displayed as l1..l5.
N_CLASSES = 5
CLASS_CODES = list(range(N_CLASSES))
CLASS_LABELS = [f"l{i + 1}" for i in CLASS_CODES]

SCORE_RANGES = {
    "l1": "0-39 (Max Caution)",
    "l2": "40-59 (Caution)",
    "l3": "60-74 (Credible w/ Exc.)",
    "l4": "75-99 (Generally Credible)",
    "l5": "100 (High Credibility)",
}

# Political leaning. Ordered left -> right, with the unlabelled bucket in the
# middle, which is the order used in the paper's figures.
LEANING_ORDER = [
    "Far Left",
    "Slightly Left",
    "Non labelled",
    "Slightly Right",
    "Far Right",
]
LEANING_UNLABELLED = "Non labelled"

# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
# Publisher-level prediction = mode of the article-level predictions.
# Ties are broken by the LOWEST class code. This is not a neutral detail: it is
# what reproduces the published publisher-level tolerant accuracies
# (RoBERTa 0.87, Llama 0.88, Mistral 0.86). Changing it shifts them.
TIE_BREAK = "lowest"

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
MODEL_COLORS = {
    "BERT": "#4C72B0",
    "RoBERTa": "#DD8452",
    "Llama": "#55A868",
    "Mistral": "#C44E52",
    # Neutral grey on purpose: the zero-shot baseline is the reference system,
    # not a fifth competitor. It also has to stay clear of the purple random
    # baseline line, which its boxes sit directly on top of.
    GPT_MODEL_NAME: "#8C8C8C",
}
MODEL_HATCHES = {
    "BERT": "//",
    "RoBERTa": "\\\\",
    "Llama": "xx",
    "Mistral": "||",
    GPT_MODEL_NAME: "..",
}

BASELINE_MAJORITY_COLOR = "#B22222"
BASELINE_RANDOM_COLOR = "#6A3D9A"

FIG_DPI = 300
RANDOM_SEED = 42
