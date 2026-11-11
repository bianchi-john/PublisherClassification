"""Step 10 (added 6/11): can the frozen embeddings (Qwen3-Embedding-8B, Mistral-Small-24B) predict the NewsGuard score
better than the ridge of step 7? The TF-IDF model stays as it is (step 7) and is the reference; ModernBERT is not touched.

Same data, same 5 outer folds and same inner protocol as step 7 (3 inner folds of the training outlets, split by
score; a setting is judged by the Spearman between true score and predicted outlet score on the inner outlets).
Every method is fitted on the training outlets only and predicts, for every test outlet, its score from ALL its articles
and from k random articles (k = 1, 2, 3, 5, 10, 20, 50; 20 draws, the same draws for every method and for TF-IDF), so
the results are curves like in the final-state deck.

Representations (`EMBEDDINGS`): Qwen3-8B; Mistral-24B layer 27 (step 7); Mistral-24B layer 40; Qwen ⊕ Mistral-27;
Mistral layers 27 ⊕ 40.
Every feature is standardised on the training articles.

Methods (`METHODS`), three ideas:
1. better regressors on the articles (outlet score = mean of its article predictions):
   - ridge          the baseline of attempt 20 (each article counts the same);
   - ridge_outlet   every outlet counts the same (outlets with 200 articles no longer dominate);
   - ridge_lds      as ridge_outlet, and outlets with rare scores count more (label distribution smoothing, LDS:
                    weight = 1 / sqrt(density of the training scores, Gaussian kernel of 5 points));
   - ridge_median   as ridge_outlet, but the outlet score is the MEDIAN of its article predictions (robust to odd
                    articles);
   - ridge_quantile as ridge_outlet, but it learns the position of the score among the training scores (0-1, uniform)
                    and turns it back into a score with the training quantiles: the rare extremes are no longer
                    squeezed by the loss;
   - krr            kernel ridge with an RBF kernel (non-linear), weights none / outlet / LDS chosen inside;
   - mlp            a small neural network (1 hidden layer), training batches drawn with outlet or LDS weights
                    (oversampling of the rare scores); 3 seeds averaged.
2. models of the outlet (one row per outlet, not per article):
   - outlet_mean    ridge on the mean article vector of every training outlet (~180 rows);
   - outlet_bags    the same plus 20 random "bags" of k articles per outlet (k = 1..50): more rows, robust to k;
   - outlet_mixup   outlet_bags plus 20 synthetic outlets each, mixing two outlets with close scores (C-Mixup):
                    more examples where scores are rare;
   - outlet_krr     RBF kernel ridge on the bags;
   - outlet_knn     the mean score of the k training outlets most similar to the outlet (cosine of mean vectors);
   - mil            attention over the articles of a bag (multiple-instance learning): the network learns which
                    articles matter; bags drawn at random every epoch (k = 1..64), outlets drawn uniformly or with LDS
                    weights; 3 seeds averaged;
   - mil_rank       the same network trained mainly to put the outlets IN ORDER (pairwise ranking loss, plus a small
                    score loss to keep the scale): what Spearman, Kendall and the stretched score measure.
3. combinations, computed afterwards from the saved predictions (no new training): the mean of the Qwen and Mistral
   predictions of the same method; the mean of all the methods of a representation; "auto" = in every outer fold the method with the best inner Spearman.

Nothing is chosen on the test outlets: every setting is chosen on the inner folds, the stretched score uses only the
training scores, and "auto" picks the method with the inner folds too.
Everything is seeded (SEED). Outputs: Dataset/tentativi/20_FromScratch/boost/predictions/<embedding>/<method>/
fold<k>.csv.gz (domain, k, draw, prediction) and inner_fold<k>.csv (every setting tried and its inner score).
Run: run_boost.sh (one process per embedding and method, with a time limit: a crash or a hang skips to the next)."""
import time

import numpy as np
import pandas as pd

import models as MO20
from common import OUT as SOURCE
from common import SEED, log

OUT = SOURCE / "boost"
PRED = OUT / "predictions"
LOG = OUT / "log.txt"
K = [1, 2, 3, 5, 10, 20, 50, "all"]
DRAWS = 20
EMBEDDINGS = {"Qwen3-8B": ["qwen3_embedding_8b.npy"], "Mistral-24B": ["mistral_small_24b_layer27.npy"],
              "Mistral-24B-L40": ["mistral_small_24b_layer40.npy"],
              "Qwen+Mistral": ["qwen3_embedding_8b.npy", "mistral_small_24b_layer27.npy"],
              "Mistral-24B-L27+L40": ["mistral_small_24b_layer27.npy", "mistral_small_24b_layer40.npy"]}
METHODS = ["ridge", "ridge_outlet", "ridge_lds", "ridge_median", "ridge_quantile", "outlet_mean", "outlet_bags", "outlet_mixup",
           "outlet_krr", "outlet_knn", "krr", "mlp", "mil", "mil_rank"]
RIDGES = ("ridge", "ridge_outlet", "ridge_lds", "ridge_median", "ridge_quantile")
ALPHAS = [0.01, 0.1, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
KRR_G, KRR_A = (0.125, 0.25, 0.5, 1), (0.01, 0.1, 1, 10)                  # kernel ridge on the articles
KRR_OUTLET_G, KRR_OUTLET_A = (0.125, 0.25, 0.5, 1), (0.0001, 0.001, 0.01, 0.1)  # kernel ridge on the outlets
BAG_SIZES = [1, 2, 3, 5, 10, 20, 50]


# ------------------------------------------------------------------------- data
def load():
    """Articles (with fold), outlets, and the row indices of the articles of every outlet."""
    art, outlets = MO20.load()
    rows = pd.Series(np.arange(len(art))).groupby(art.domain.to_numpy()).apply(np.array)
    return art, outlets, rows


def embedding(name, art):
    order = pd.read_csv(SOURCE / "embeddings" / "article_order.csv").article_id.to_numpy()
    assert (order == art.article_id.to_numpy()).all()
    return np.hstack([np.load(SOURCE / "embeddings" / f) for f in EMBEDDINGS[name]]).astype(np.float32)


def test_draws(outlets, rows, fold):
    """For every test outlet of the fold: the rows of ALL its articles and of 20 draws of k random articles. The same
    for every method and for TF-IDF (it depends only on the fold)."""
    rng = np.random.default_rng(SEED * 100 + fold)
    out = []
    for d in outlets.index[outlets.fold == fold]:
        r = rows[d]
        out.append((d, "all", 0, r))
        for k in K[:-1]:
            for j in range(DRAWS):
                out.append((d, str(k), j, rng.choice(r, size=min(k, len(r)), replace=False)))
    return out


def lds_weight(train_scores, scores, sigma=5.0):
    """Label distribution smoothing: 1 / sqrt(density of the training outlet scores) at `scores`, mean 1 over the
    training outlets."""
    t = np.asarray(train_scores, float)
    dens = lambda s: np.exp(-(np.asarray(s, float)[:, None] - t[None, :]) ** 2 / (2 * sigma ** 2)).sum(1)
    return (1 / np.sqrt(dens(scores))) / (1 / np.sqrt(dens(t))).mean()


def article_weights(scheme, dom, score, train_outlet_scores):
    """none: every article 1; outlet: every outlet the same total weight; lds: outlet × LDS. Mean 1."""
    if scheme == "none":
        return np.ones(len(dom))
    n = pd.Series(dom).map(pd.Series(dom).value_counts()).to_numpy()
    w = 1.0 / n
    if scheme == "lds":
        w = w * lds_weight(train_outlet_scores, score)
    return w / w.mean()


def to_quantile(train_scores, s):
    """Position of a score among the training outlet scores, 0-1 (ties: the middle)."""
    t = np.sort(train_scores)
    return (np.searchsorted(t, s, "left") + np.searchsorted(t, s, "right")) / (2 * len(t))


def from_quantile(train_scores, q):
    return np.quantile(train_scores, np.clip(q, 0, 1))


def ridge_target(F, setting):
    y = F.y_art[F.tr]
    return to_quantile(F.train_scores, y) if setting.get("target") == "quantile" else y


# ------------------------------------------------------------------------- linear algebra on the GPU
def torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_path(A, y, w, alphas):
    """Weighted ridge for every alpha at once (one eigendecomposition): returns [(beta, intercept)]."""
    import torch
    w = torch.as_tensor(w, dtype=torch.float32, device=A.device); y = torch.as_tensor(y, dtype=torch.float32, device=A.device)
    mx = (w[:, None] * A).sum(0) / w.sum(); my = (w * y).sum() / w.sum()
    Ac = A - mx
    G = (Ac * w[:, None]).T @ Ac
    b = (Ac * w[:, None]).T @ (y - my)
    ev, V = torch.linalg.eigh(G.double())
    p = V.T @ b.double()
    out = []
    for a in alphas:
        beta = (V @ (p / (ev.clamp(min=0) + a))).float()
        out.append((beta, my - mx @ beta))
    del G, Ac
    return out


def sq_dist(A, B):
    import torch
    return (A.pow(2).sum(1)[:, None] + B.pow(2).sum(1)[None, :] - 2 * A @ B.T).clamp(min=0)


def krr_fit_predict(A, y, w, B_list, gammas, alphas):
    """RBF kernel ridge, weighted: (K + alpha W^-1) c = y - mean. gamma is relative to the median squared distance.
    Returns {(gamma, alpha): [predictions for every matrix in B_list]}."""
    import torch
    dev = A.device
    y = torch.as_tensor(y, dtype=torch.float64, device=dev); w = torch.as_tensor(w, dtype=torch.float64, device=dev)
    my = (w * y).sum() / w.sum()
    s = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(SEED))[:2000].to(dev)]
    med = sq_dist(s, s).median().item()
    D = sq_dist(A, A).double()
    DB = [sq_dist(B, A) for B in B_list]
    out = {}
    for g in gammas:
        Kx = torch.exp(-g * D / med)
        KB = [torch.exp(-g * d.double() / med) for d in DB]
        for a in alphas:
            M = Kx + torch.diag(a / w)
            L = torch.linalg.cholesky(M)
            c = torch.cholesky_solve((y - my)[:, None], L)[:, 0]
            out[(g, a)] = [(kb @ c + my).float().cpu().numpy() for kb in KB]
            del M, L
        del Kx, KB
    del D, DB; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------- the methods
class Fold:
    """The data of one fit: standardised features on the GPU, the rows of every outlet, scores."""

    def __init__(self, X, art, outlets, rows, train_outlets):
        import torch
        self.dev = torch_device()
        tr = np.concatenate([rows[d] for d in train_outlets])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        self.X = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=self.dev)
        self.rows, self.train_outlets = rows, list(train_outlets)
        self.tr = tr
        self.score = outlets.score
        self.y_art = art.score.to_numpy().astype(float)
        self.dom_art = art.domain.to_numpy()
        self.train_scores = outlets.score.reindex(self.train_outlets).to_numpy()

    def bag_means(self, bags):
        import torch
        return torch.stack([self.X[torch.as_tensor(b, device=self.dev)].mean(0) for b in bags])


def predictor_articles(F, predict_rows, agg=np.mean):
    """Outlet score = mean (or `agg`) of the predictions of its articles; predictions of each row computed once."""
    cache = np.full(F.X.shape[0], np.nan)

    def predict(bags):
        need = np.unique(np.concatenate(bags))
        new = need[np.isnan(cache[need])]
        if len(new):
            cache[new] = predict_rows(new)
        return np.array([agg(cache[b]) for b in bags])
    return predict


def make_bags(F, rng, n_bags):
    """Training rows of outlet models: for every training outlet its mean (all articles) and n_bags random bags."""
    bags, ys, who = [], [], []
    for d in F.train_outlets:
        r = F.rows[d]
        bags.append(r); ys.append(F.score[d]); who.append(d)
        for _ in range(n_bags):
            k = min(int(rng.choice(BAG_SIZES)), len(r))
            bags.append(rng.choice(r, size=k, replace=False)); ys.append(F.score[d]); who.append(d)
    return bags, np.array(ys, float), np.array(who)


def mixup_rows(F, rng, n_per_outlet, sigma=10.0):
    """C-Mixup: synthetic outlets mixing a bag of an outlet with a bag of an outlet of close score."""
    import torch
    s = F.train_scores
    xs, ys = [], []
    for i, d in enumerate(F.train_outlets):
        p = np.exp(-(s - s[i]) ** 2 / (2 * sigma ** 2)); p[i] = 0; p /= p.sum()
        for _ in range(n_per_outlet):
            j = rng.choice(len(s), p=p); lam = rng.beta(2, 2)
            bi, bj = [rng.choice(F.rows[o], size=min(int(rng.choice(BAG_SIZES)), len(F.rows[o])), replace=False)
                      for o in (d, F.train_outlets[j])]
            xs.append(lam * F.X[torch.as_tensor(bi, device=F.dev)].mean(0) + (1 - lam) * F.X[torch.as_tensor(bj, device=F.dev)].mean(0))
            ys.append(lam * s[i] + (1 - lam) * s[j])
    return torch.stack(xs), np.array(ys)


def fit_ridge_articles(F, setting, seed):
    w = article_weights(setting["weights"], F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
    beta, b0 = ridge_path(F.X[F.tr], ridge_target(F, setting), w, [setting["alpha"]])[0]
    base = predictor_articles(F, lambda r: (F.X[r] @ beta + b0).cpu().numpy(),
                              np.median if setting.get("aggregate") == "median" else np.mean)
    if setting.get("target") == "quantile":
        return lambda bags: from_quantile(F.train_scores, base(bags))
    return base


def fit_outlet_ridge(F, setting, seed):
    import torch
    rng = np.random.default_rng(seed)
    bags, y, who = make_bags(F, rng, setting["bags"])
    A = F.bag_means(bags)
    if setting.get("mixup"):
        Am, ym = mixup_rows(F, rng, setting["mixup"])
        A, y = torch.cat([A, Am]), np.concatenate([y, ym])
    w = lds_weight(F.train_scores, y) if setting["weights"] == "lds" else np.ones(len(y))
    beta, b0 = ridge_path(A, y, w / w.mean(), [setting["alpha"]])[0]
    return lambda bags: (F.bag_means(bags) @ beta + b0).cpu().numpy()


def fit_outlet_krr(F, setting, seed):
    rng = np.random.default_rng(seed)
    bags, y, _ = make_bags(F, rng, setting["bags"])
    A = F.bag_means(bags)
    w = lds_weight(F.train_scores, y) if setting["weights"] == "lds" else np.ones(len(y))

    def predict(test_bags):
        return krr_fit_predict(A, y, w / w.mean(), [F.bag_means(test_bags)], [setting["gamma"]], [setting["alpha"]])[
            (setting["gamma"], setting["alpha"])][0]
    return predict


def fit_krr_articles(F, setting, seed):
    w = article_weights(setting["weights"], F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
    A = F.X[F.tr]
    return predictor_articles(F, lambda r: krr_fit_predict(A, F.y_art[F.tr], w, [F.X[r]], [setting["gamma"]],
                                                           [setting["alpha"]])[(setting["gamma"], setting["alpha"])][0])


def train_mlp(F, setting, seed, epochs):
    """Article MLP; returns the model after each epoch in `epochs` (as state dicts)."""
    import torch
    from torch import nn
    MO20.reproducible(seed)
    d = F.X.shape[1]
    net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, setting["hidden"]), nn.GELU(), nn.Dropout(setting["dropout"]),
                        nn.Linear(setting["hidden"], 1)).to(F.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=setting["wd"])
    w = article_weights(setting["weights"], F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr = torch.as_tensor(F.tr); y = torch.as_tensor((F.y_art - 50) / 25, dtype=torch.float32, device=F.dev)
    states = {}
    for e in range(1, max(epochs) + 1):
        net.train()
        idx = tr[torch.multinomial(torch.as_tensor(w, dtype=torch.float32), len(tr), replacement=True, generator=g)]
        for b in idx.split(256):
            b = b.to(F.dev)
            loss = nn.functional.smooth_l1_loss(net(F.X[b])[:, 0], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if e in epochs:
            states[e] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return net, states


def mlp_predictor(F, net, states_list):
    import torch

    def rows(r):
        out = []
        with torch.no_grad():
            for st in states_list:
                net.load_state_dict(st); net.eval()
                out.append(torch.cat([net(F.X[torch.as_tensor(c, device=F.dev)])[:, 0] for c in np.array_split(r, max(1, len(r) // 4096))]).cpu().numpy())
        return np.mean(out, 0) * 25 + 50
    return predictor_articles(F, rows)


def knn_predict(F, n, V):
    """Mean score of the n training outlets whose mean vector is most similar (cosine) to each row of V."""
    import torch
    T = F.bag_means([F.rows[d] for d in F.train_outlets])
    sim = torch.nn.functional.normalize(V, dim=1) @ torch.nn.functional.normalize(T, dim=1).T
    top = sim.topk(min(n, T.shape[0]), dim=1).indices.cpu().numpy()
    return F.train_scores[top].mean(1)


class MIL:
    """Attention pooling over the articles of a bag."""

    @staticmethod
    def build(d, setting):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, 256), nn.GELU(), nn.Dropout(setting["dropout"]))
                self.att = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
                self.head = nn.Linear(256, 1)

            def forward(self, x, mask):
                h = self.enc(x)
                a = self.att(h)[..., 0].masked_fill(~mask, float("-inf")).softmax(-1)
                return self.head((a[..., None] * h).sum(1))[:, 0]
        return Net()

    @staticmethod
    def batch(F, bags):
        import torch
        L = max(len(b) for b in bags)
        idx = torch.zeros(len(bags), L, dtype=torch.long); mask = torch.zeros(len(bags), L, dtype=torch.bool)
        for i, b in enumerate(bags):
            idx[i, :len(b)] = torch.as_tensor(b); mask[i, :len(b)] = True
        return F.X[idx.to(F.dev)], mask.to(F.dev)


def train_mil(F, setting, seed, epochs, bags_per_outlet=8):
    import torch
    from torch import nn
    MO20.reproducible(seed)
    rng = np.random.default_rng(seed)
    net = MIL.build(F.X.shape[1], setting).to(F.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=setting["wd"])
    n = len(F.train_outlets)
    p = lds_weight(F.train_scores, F.train_scores) if setting["weights"] == "lds" else np.ones(n)
    p = p / p.sum()
    states = {}
    for e in range(1, max(epochs) + 1):
        net.train()
        who = rng.choice(n, size=n * bags_per_outlet, p=p)
        bags, ys = [], []
        for i in who:
            r = F.rows[F.train_outlets[i]]
            k = min(int(rng.choice(BAG_SIZES + [64])), len(r))
            bags.append(rng.choice(r, size=k, replace=False)); ys.append(F.train_scores[i])
        ys = (np.array(ys) - 50) / 25
        for s in range(0, len(bags), 32):
            x, m = MIL.batch(F, bags[s:s + 32])
            out, t = net(x, m), torch.as_tensor(ys[s:s + 32], dtype=torch.float32, device=F.dev)
            loss = nn.functional.smooth_l1_loss(out, t)
            if setting.get("loss") == "rank":                  # pairwise: the higher score must come out higher
                sign = torch.sign(t[:, None] - t[None, :])
                pair = nn.functional.softplus(-sign * (out[:, None] - out[None, :]) * 4)
                loss = (pair * (sign != 0)).sum() / (sign != 0).sum().clamp(min=1) + 0.1 * loss
            opt.zero_grad(); loss.backward(); opt.step()
        if e in epochs:
            states[e] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return net, states


def mil_predictor(F, net, states_list):
    import torch

    def predict(bags):
        out = []
        with torch.no_grad():
            for st in states_list:
                net.load_state_dict(st); net.eval()
                pr = []
                order = np.argsort([len(b) for b in bags])
                res = np.zeros(len(bags))
                for s in range(0, len(bags), 64):
                    sel = order[s:s + 64]
                    x, m = MIL.batch(F, [bags[i] for i in sel])
                    res[sel] = net(x, m).cpu().numpy()
                out.append(res)
        return np.mean(out, 0) * 25 + 50
    return predict


# ------------------------------------------------------------------------- settings tried with the inner folds
def grid(method):
    """Every setting of a method. Ridge-like methods: all alphas share one eigendecomposition (see inner_scores)."""
    if method in ("ridge", "ridge_outlet", "ridge_lds"):
        w = {"ridge": "none", "ridge_outlet": "outlet", "ridge_lds": "lds"}[method]
        return [{"weights": w, "alpha": a} for a in ALPHAS]
    if method == "ridge_median":
        return [{"weights": "outlet", "aggregate": "median", "alpha": a} for a in ALPHAS]
    if method == "outlet_knn":
        return [{"neighbours": n} for n in (3, 5, 10, 20, 40)]
    if method == "ridge_quantile":
        return [{"weights": "outlet", "target": "quantile", "alpha": a} for a in ALPHAS]
    if method == "outlet_mean":
        return [{"weights": "none", "bags": 0, "alpha": a} for a in ALPHAS]
    if method == "outlet_bags":
        return [{"weights": w, "bags": 20, "alpha": a} for w in ("none", "lds") for a in ALPHAS]
    if method == "outlet_mixup":
        return [{"weights": w, "bags": 20, "mixup": 20, "alpha": a} for w in ("none", "lds") for a in ALPHAS]
    if method == "outlet_krr":
        return [{"weights": w, "bags": 20, "gamma": g, "alpha": a} for w in ("none", "lds") for g in KRR_OUTLET_G
                for a in KRR_OUTLET_A]
    if method == "krr":
        return [{"weights": w, "gamma": g, "alpha": a} for w in ("none", "outlet", "lds") for g in KRR_G for a in KRR_A]
    if method == "mlp":
        return [{"weights": w, "dropout": p, "wd": wd, "hidden": 512, "epoch": e} for w in ("outlet", "lds")
                for p in (0.25, 0.5, 0.7) for wd in (0.01, 0.1, 1.0) for e in (3, 6, 10, 20)]
    if method in ("mil", "mil_rank"):
        loss = "rank" if method == "mil_rank" else "huber"
        return [{"weights": w, "dropout": p, "wd": 0.01, "loss": loss, "epoch": e} for w in ("none", "lds") for p in (0.25, 0.5)
                for e in (10, 20, 40, 80)]
    raise ValueError(method)


def _key(s, drop=("alpha", "epoch")):
    return tuple(sorted((k, v) for k, v in s.items() if k not in drop))


def inner_scores(method, X, art, outlets, rows, train_outlets, fold):
    """Every setting on the 3 inner folds: mean Spearman of the outlet predictions (all articles)."""
    from scipy.stats import spearmanr
    inner = MO20.score_split(outlets.score.reindex(train_outlets), SEED + fold, k=3)
    settings = grid(method)
    res = {i: [] for i in range(len(settings))}
    for j in (1, 2, 3):
        fit_out, val_out = inner.index[inner != j], inner.index[inner == j]
        F = Fold(X, art, outlets, rows, fit_out)
        bags = [rows[d] for d in val_out]
        y = outlets.score.reindex(val_out).to_numpy()
        rho = lambda p: spearmanr(y, p).correlation
        seed = SEED + 10 * fold + j
        if method in RIDGES:
            w = article_weights(settings[0]["weights"], F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
            path = ridge_path(F.X[F.tr], ridge_target(F, settings[0]), w, ALPHAS)
            agg = np.median if settings[0].get("aggregate") == "median" else np.mean
            for i, (beta, b0) in enumerate(path):
                res[i].append(rho(predictor_articles(F, lambda r: (F.X[r] @ beta + b0).cpu().numpy(), agg)(bags)))
        elif method in ("outlet_mean", "outlet_bags", "outlet_mixup"):
            import torch
            groups = {}
            for i, s in enumerate(settings):
                groups.setdefault(_key(s), []).append(i)
            for g, ids in groups.items():
                s0 = settings[ids[0]]
                rng = np.random.default_rng(seed)
                bb, yy, _ = make_bags(F, rng, s0["bags"])
                A = F.bag_means(bb)
                if s0.get("mixup"):
                    Am, ym = mixup_rows(F, rng, s0["mixup"]); A, yy = torch.cat([A, Am]), np.concatenate([yy, ym])
                w = lds_weight(F.train_scores, yy) if s0["weights"] == "lds" else np.ones(len(yy))
                path = ridge_path(A, yy, w / w.mean(), [settings[i]["alpha"] for i in ids])
                V = F.bag_means(bags)
                for i, (beta, b0) in zip(ids, path):
                    res[i].append(rho((V @ beta + b0).cpu().numpy()))
        elif method == "outlet_krr":
            for wsch in ("none", "lds"):
                rng = np.random.default_rng(seed)
                bb, yy, _ = make_bags(F, rng, 20)
                w = lds_weight(F.train_scores, yy) if wsch == "lds" else np.ones(len(yy))
                out = krr_fit_predict(F.bag_means(bb), yy, w / w.mean(), [F.bag_means(bags)], KRR_OUTLET_G, KRR_OUTLET_A)
                for i, s in enumerate(settings):
                    if s["weights"] == wsch:
                        res[i].append(rho(out[(s["gamma"], s["alpha"])][0]))
        elif method == "krr":
            allrows = np.concatenate(bags)
            for wsch in ("none", "outlet", "lds"):
                w = article_weights(wsch, F.dom_art[F.tr], F.y_art[F.tr], F.train_scores)
                out = krr_fit_predict(F.X[F.tr], F.y_art[F.tr], w, [F.X[allrows]], KRR_G, KRR_A)
                ends = np.cumsum([len(b) for b in bags])[:-1]
                for i, s in enumerate(settings):
                    if s["weights"] == wsch:
                        res[i].append(rho([p.mean() for p in np.split(out[(s["gamma"], s["alpha"])][0], ends)]))
        elif method == "outlet_knn":
            V = F.bag_means(bags)
            for i, s in enumerate(settings):
                res[i].append(rho(knn_predict(F, s["neighbours"], V)))
        elif method in ("mlp", "mil", "mil_rank"):
            groups = {}
            for i, s in enumerate(settings):
                groups.setdefault(_key(s, ("epoch",)), []).append(i)
            for g, ids in groups.items():
                s0 = settings[ids[0]]; ep = sorted({settings[i]["epoch"] for i in ids})
                if method == "mlp":
                    net, st = train_mlp(F, s0, seed, ep)
                    for i in ids:
                        res[i].append(rho(mlp_predictor(F, net, [st[settings[i]["epoch"]]])(bags)))
                else:
                    net, st = train_mil(F, s0, seed, ep)
                    for i in ids:
                        res[i].append(rho(mil_predictor(F, net, [st[settings[i]["epoch"]]])(bags)))
        del F
        import torch; torch.cuda.empty_cache()
    return pd.DataFrame([{**s, "inner Spearman": np.mean(res[i])} for i, s in enumerate(settings)])


def final_predictor(method, setting, F, fold):
    seed = SEED + 10 * fold
    if method in RIDGES:
        return fit_ridge_articles(F, setting, seed)
    if method in ("outlet_mean", "outlet_bags", "outlet_mixup"):
        return fit_outlet_ridge(F, setting, seed)
    if method == "outlet_krr":
        return fit_outlet_krr(F, setting, seed)
    if method == "krr":
        return fit_krr_articles(F, setting, seed)
    if method == "outlet_knn":
        return lambda bags: knn_predict(F, setting["neighbours"], F.bag_means(bags))
    seeds = [seed, seed + 1, seed + 2]                       # neural networks: 3 seeds, predictions averaged
    if method == "mlp":
        nets = [train_mlp(F, setting, s, [setting["epoch"]]) for s in seeds]
        return mlp_predictor(F, nets[0][0], [st[setting["epoch"]] for _, st in nets])
    nets = [train_mil(F, setting, s, [setting["epoch"]]) for s in seeds]
    return mil_predictor(F, nets[0][0], [st[setting["epoch"]] for _, st in nets])


def run(emb_name, method, fold, X, art, outlets, rows):
    """One method, one embedding, one outer fold: inner choice of the setting, final fit, test predictions."""
    folder = PRED / emb_name / method; folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"fold{fold}.csv.gz").exists():
        return
    MO20.wait_for_gpu(12, LOG)
    start = time.time()
    train_outlets = outlets.index[outlets.fold != fold]
    scores = inner_scores(method, X, art, outlets, rows, train_outlets, fold)
    best = scores.loc[scores["inner Spearman"].idxmax()].drop("inner Spearman").to_dict()
    best = {k: v for k, v in best.items() if not (isinstance(v, float) and np.isnan(v))}
    for k in ("bags", "mixup", "epoch", "hidden", "neighbours"):
        if k in best:
            best[k] = int(best[k])
    F = Fold(X, art, outlets, rows, train_outlets)
    predict = final_predictor(method, best, F, fold)
    draws = test_draws(outlets, rows, fold)
    p = predict([b for *_, b in draws])
    pd.DataFrame({"domain": [d for d, *_ in draws], "k": [k for _, k, _, _ in draws], "draw": [j for _, _, j, _ in draws],
                  "prediction": p}).to_csv(folder / f"fold{fold}.csv.gz", index=False)
    scores.assign(fold=fold, chosen=scores.index == scores["inner Spearman"].idxmax()).to_csv(folder / f"inner_fold{fold}.csv", index=False)
    del F
    import torch; torch.cuda.empty_cache()
    log(f"{emb_name}, {method}, fold {fold}: {best} inner {scores['inner Spearman'].max():.3f}, {time.time() - start:.0f} s", LOG)


def tfidf_draws(art, outlets, rows):
    """TF-IDF of attempt 20 (unchanged) on the same draws: mean of its article predictions."""
    pa = MO20.predictions("regression", "TF-IDF").set_index("article_id").prediction.reindex(art.article_id).to_numpy()
    out = []
    for f in range(1, 6):
        for d, k, j, b in test_draws(outlets, rows, f):
            out.append((d, k, j, pa[b].mean()))
    return pd.DataFrame(out, columns=["domain", "k", "draw", "prediction"])


def collect():
    """All the saved outlet predictions: one row per (embedding, method, domain, k, draw)."""
    parts = []
    for f in sorted(PRED.glob("*/*/fold*.csv.gz")):
        parts.append(pd.read_csv(f, dtype={"k": str}).assign(embedding=f.parent.parent.name, method=f.parent.name))
    return pd.concat(parts, ignore_index=True)
