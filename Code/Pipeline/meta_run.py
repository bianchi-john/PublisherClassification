"""Step 11, one piece at a time (used by run_meta.sh). A fold that fails is logged and the next one starts; whatever is
already saved is skipped, so every command can simply be re-run.
Usage:  python meta_run.py build | embed <Qwen3-8B|Mistral-24B> | fit <TF-IDF|Qwen3-8B · ridge|...> | bert"""
import sys
import traceback

import numpy as np

import boost as BO
import meta as ME
import models as MO20
from common import log

ME.OUT.mkdir(parents=True, exist_ok=True)
what = sys.argv[1]
art, outlets, rows_art = BO.load()


def guarded(label, f, *a):
    try:
        f(*a); return 0
    except Exception as e:                           # noqa: BLE001  (log and go on)
        log(f"ERROR {label}: {type(e).__name__}: {e}", ME.LOG)
        with open(ME.OUT / "errors.txt", "a") as fh:
            fh.write(f"==== {label}\n{traceback.format_exc()}\n")
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            pass
        return 1


if what == "build":
    if not (ME.OUT / "meta_texts.csv.gz").exists():
        M = ME.build(art, outlets)
        M.to_csv(ME.OUT / "meta_texts.csv.gz", index=False)
        for name, model in [("ModernBERT", ME.BERT), ("Mistral-24B", ME.MISTRAL)]:
            n = MO20.count_tokens(M.text.tolist(), model)
            log(f"meta-articles: {len(M):,}, {name} tokens median {int(np.median(n))}, max {n.max()}, "
                f"over {MO20.MAX_TOKENS}: {(n > MO20.MAX_TOKENS).sum()}", ME.LOG)
        log(f"meta-articles: articles per meta median {int(M.articles.median())}, words median {int(M.words.median())}", ME.LOG)
    sys.exit(0)
M, rows = ME.load_meta()
failed = 0
if what == "embed":
    failed += guarded(f"embed {sys.argv[2]}", ME.embed, sys.argv[2], M)
elif what == "fit":
    name = sys.argv[2]
    for f in range(1, 6):
        if name == "TF-IDF":
            failed += guarded(f"TF-IDF fold {f}", ME.fit_tfidf, f, M, outlets)
        else:
            emb, method = name.split(" · ")
            failed += guarded(f"{name} fold {f}", ME.fit_embedding, emb, method, f, M, rows, outlets)
elif what == "bert":
    for f in range(1, 6):
        failed += guarded(f"ModernBERT fold {f}", ME.fit_bert, f, M, art, outlets)
sys.exit(1 if failed else 0)
