"""Step 10, one process: one representation and one method on the 5 outer folds (boost.run). A fold that fails is
logged with its error and the next fold starts; folds already saved are skipped, so the command can simply be re-run.
Usage:  python boost_run.py <embedding> <method> [folds]   e.g.  python boost_run.py Qwen3-8B mlp   (folds: "1,2")"""
import sys
import traceback

import boost as BO
from common import log

emb, method = sys.argv[1], sys.argv[2]
assert emb in BO.EMBEDDINGS and method in BO.METHODS, (emb, method)
BO.OUT.mkdir(parents=True, exist_ok=True)
folds = [int(f) for f in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(1, 6)
if all((BO.PRED / emb / method / f"fold{f}.csv.gz").exists() for f in folds):
    log(f"{emb}, {method}: already done", BO.LOG); sys.exit(0)
art, outlets, rows = BO.load()
X = BO.embedding(emb, art)
log(f"{emb}, {method}: start ({X.shape[1]} features)", BO.LOG)
failed = 0
for f in folds:
    try:
        BO.run(emb, method, f, X, art, outlets, rows)
    except Exception as e:                           # noqa: BLE001  (log and go on with the next fold)
        failed += 1
        log(f"ERROR {emb}, {method}, fold {f}: {type(e).__name__}: {e}", BO.LOG)
        with open(BO.OUT / "errors.txt", "a") as fh:
            fh.write(f"==== {emb}, {method}, fold {f}\n{traceback.format_exc()}\n")
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            pass
sys.exit(1 if failed else 0)
