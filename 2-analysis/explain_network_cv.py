"""Network-disjoint cross-validation for the TF-IDF + LR baseline.

Publisher-disjoint folds still let sites of the same content network (shared
templates, syndicated articles) sit on both sides of a split. Here every
network detected in explain/publisher_networks.csv is kept inside one fold
(StratifiedGroupKFold over publishers, group = network, stratified by label),
and the baseline is retrained under those folds.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import config as C
import run_tfidf_lr_baseline as B
import explain_tfidf as X

OUT = C.DERIVED_DIR / "explain"
net = pd.read_csv(OUT / "publisher_networks.csv", index_col=0)
df = B.load().rename(columns={"label": "y_true"})

pubs = net.reset_index().rename(columns={"index": "domain"})
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
pubs["net_fold"] = -1
for k, (_, te) in enumerate(sgkf.split(pubs, pubs.label, groups=pubs.comp), start=1):
    pubs.loc[te, "net_fold"] = k
df = df.drop(columns="fold").merge(pubs[["domain", "net_fold"]], on="domain").rename(columns={"net_fold": "fold"})
assert df.groupby("domain").fold.nunique().max() == 1
comp_folds = pubs.groupby("comp").net_fold.nunique()
assert comp_folds.max() == 1, "a network spans folds"

print("publishers per fold and class:")
print(pd.crosstab(pubs.net_fold, pubs.label))

art = X.cv(df, df.text, X.VEC_KW)
art.to_csv(OUT / "pred_network_disjoint.csv", index=False)
res = {"network_disjoint_all": X.scores(art)}
innet = art.domain.map(net.comp_size >= 3)
res["network_disjoint_in_networks"] = X.scores(art[innet])
res["network_disjoint_outside_networks"] = X.scores(art[~innet])
pub = B.aggregate_publisher(art).set_index("domain")
pub["innet"] = net.comp_size.reindex(pub.index) >= 3
pub["ok"] = pub.y_true == pub.y_pred
res["network_disjoint_pub_acc_by_class_network"] = (
    pub.groupby(["y_true", "innet"]).ok.agg(["mean", "size"]).round(3)
    .reset_index().to_dict(orient="records"))
(OUT / "network_cv_results.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=2, default=float))
