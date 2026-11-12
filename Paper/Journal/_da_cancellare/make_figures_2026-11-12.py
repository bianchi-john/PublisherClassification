"""Figures of the journal draft (Paper/Journal). Run from this folder:  python make_figures.py

- dataset.pdf: the 226 outlets: NewsGuard score (with the five NewsGuard levels), articles per outlet, score per fold.
  Final: the dataset and the split do not depend on the models.
- groups.pdf, network.pdf, anonymization.pdf: how the dataset is built (ownership groups and duplicates, outlets linked
  by shared text and by similar writing, what the anonymization replaces). From the aggregates written on the server by
  dataset_construction.py into Dataset/tentativi/20_FromScratch/paper_figures/ (outlets are numbered, never named).
- curves.pdf: Spearman and MAE per outlet as more articles per outlet are read (k = 1, 2, 3, 5, 10, 20, 50, all;
  20 random draws per k, the same for every model), for the regressors of step 7 and the no-text baseline. Drawn only
  when the step-7 predictions of the current run are in Dataset/tentativi/20_FromScratch/predictions/regression/.
Only aggregate statistics are plotted (NewsGuard labels are not redistributable)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "Dataset" / "tentativi" / "20_FromScratch"
sys.path.insert(0, str(REPO / "Code" / "Pipeline"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
BOUNDS = [39.5, 59.5, 74.5, 99.5]
LEVEL_COLOURS = ["#b02a2a", "#e58a8a", "#bdbdbd", "#7fb0e0", "#1d4f91"]
MODELS = {"TF-IDF": "#8a8983", "Qwen3-8B": "#2a78d6", "Mistral-24B": "#1d4f91", "ModernBERT": "#eb6834"}
FIGDATA = DATA / "paper_figures"
# categorical slots in fixed order (validated: adjacent pairs for 5 slots, all pairs for the first 3)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, FAINT, SURFACE = "#2b2b28", "#6b6a63", "#cfcdc5", "#ffffff"


def dataset_figure():
    o = pd.read_csv(DATA / "outlets.csv", index_col="domain")
    o["fold"] = pd.read_csv(DATA / "split.csv", index_col="domain").fold
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    edges = [0, *BOUNDS, 100.5]
    for k in range(5):
        a.axvspan(edges[k], edges[k + 1], color=LEVEL_COLOURS[k], alpha=0.15, lw=0)
    a.hist(o.score, bins=np.arange(0, 102.5, 5), color="#444444", rwidth=0.9)
    a.set_xlabel("NewsGuard score"); a.set_ylabel("outlets"); a.set_title("(a) score of the 226 outlets", fontsize=9)
    b.hist(o.articles, bins=np.arange(0, 211, 15), color="#444444", rwidth=0.9)
    b.set_xlabel("articles per outlet"); b.set_title("(b) articles per outlet", fontsize=9)
    c.boxplot([o.score[o.fold == f] for f in range(1, 6)], widths=0.5, medianprops={"color": "#b02a2a"})
    c.set_xlabel("fold"); c.set_title("(c) score per fold", fontsize=9)
    plt.tight_layout(); plt.savefig(HERE / "dataset.pdf", bbox_inches="tight"); plt.close()
    print("dataset.pdf")


def groups_figure():
    """Outlets after every cleaning step: one per ownership group + the other outlets of the same group (sisters)."""
    s = pd.read_csv(FIGDATA / "steps.csv")
    s = s[s.step != "sent to the scraper"].reset_index(drop=True)
    names = {"at least one article": "at least\none article", "text extracted": "text\nextracted", "English": "English",
             "100-1,500 words": "100–1,500\nwords", "from 2025": "from\n2025", "no exact duplicates": "no exact\nduplicates",
             "no near duplicates": "no near\nduplicates", ">= 10 articles": "≥ 10\narticles",
             "one outlet per group": "one outlet\nper group"}
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(x, s.groups, 0.62, color=SLOTS[0], edgecolor=SURFACE, lw=1, label="one outlet per ownership group")
    ax.bar(x, s.sister_outlets, 0.62, bottom=s.groups, color=SLOTS[1], edgecolor=SURFACE, lw=1,
           label="sister outlets (other outlets of the same group)")
    for i, r in s.iterrows():
        ax.text(i, r.outlets + 12, f"{r.outlets}", ha="center", fontsize=8, color=INK, fontweight="bold")
        if r.sister_outlets:
            ax.text(i, r.groups + r.sister_outlets / 2, f"{r.sister_outlets}", ha="center", va="center", fontsize=7, color=INK)
        ax.text(i, r.groups / 2, f"{r.groups}", ha="center", va="center", fontsize=7, color=SURFACE)
    ax.set_xticks(x, [f"{names.get(n, n)}\n({r:,.0f})" for n, r in zip(s.step, s.articles)], fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("cleaning step (articles left)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("outlets"); ax.set_ylim(0, s.outlets.max() * 1.12); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    plt.tight_layout(); plt.savefig(HERE / "groups.pdf", bbox_inches="tight"); plt.close()
    print("groups.pdf")


def packed_layout(edges, seed=20261112):
    """Positions of the outlets that share articles: every connected component laid out on its own (spring layout,
    radius ~ sqrt of its size), the largest on the left, the others packed in rows to its right."""
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(zip(edges.a, edges.b))
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    radius = lambda c: 0.55 * np.sqrt(len(c)) + 0.35
    pos, r0 = {}, radius(comps[0])
    x0, y0, row_h, width = 2 * r0 + 0.8, r0, 0.0, 2.6 * r0               # cursor of the packing on the right
    for i, c in enumerate(comps):
        r = radius(c)
        p = nx.spring_layout(G.subgraph(c), seed=seed, iterations=200) if len(c) > 2 else \
            {v: np.array([(-1) ** j * 0.6, 0.0]) for j, v in enumerate(sorted(c))}
        xy = np.array(list(p.values())); xy = xy - xy.mean(0); xy = xy / max(np.abs(xy).max(), 1e-9)
        if i == 0:
            centre = np.array([r0, 0.0])
        else:
            if x0 + 2 * r > 2 * r0 + 0.8 + width:
                x0, y0 = 2 * r0 + 0.8, y0 - row_h - 0.5; row_h = 0.0
            centre = np.array([x0 + r, y0 - r]); x0 += 2 * r + 0.4; row_h = max(row_h, 2 * r)
        for v, q in zip(p, xy):
            pos[v] = centre + q * (r - 0.3)
    return pos


def network_figure():
    """(a) outlets linked by shared articles, coloured by ownership group; (b) text similarity to the nearest sister
    and to the nearest outlet of another group."""
    nodes, edges = pd.read_csv(FIGDATA / "nodes.csv"), pd.read_csv(FIGDATA / "edges.csv")
    sim = pd.read_csv(FIGDATA / "similarity.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.55, 1]})
    pos = packed_layout(edges)
    shown = nodes[nodes.node.isin(pos)].set_index("node")
    shown["x"] = [pos[v][0] for v in shown.index]; shown["y"] = [pos[v][1] for v in shown.index]
    big = shown[shown.group_size > 1].group.value_counts().index[:3]          # the three largest groups on the map
    colour = {g: SLOTS[i] for i, g in enumerate(big)}
    for e in edges.itertuples():
        if e.a in shown.index and e.b in shown.index:
            pa, pb = shown.loc[e.a], shown.loc[e.b]
            a.plot([pa.x, pb.x], [pa.y, pb.y], color=FAINT, lw=0.25 + 0.25 * np.log10(e.shared), alpha=0.6, zorder=1)
    def style(r):
        if r.group in colour:
            return colour[r.group]
        return "#55544f" if r.group_size > 1 else "#b4b2a9"
    for kept, marker_kw in [(False, {"facecolors": "none", "linewidths": 0.9}), (True, {"edgecolors": SURFACE, "linewidths": 0.6})]:
        d = shown[shown.kept == kept]
        c = [style(r) for r in d.itertuples()]
        kw = dict(marker_kw, **({"edgecolors": c} if not kept else {}))
        a.scatter(d.x, d.y, s=13 if kept else 11, c=c if kept else None, zorder=3 if kept else 2, **kw)
    a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_aspect("equal")
    for sp in a.spines.values():
        sp.set_visible(False)
    sizes = nodes.group_size[nodes.group.isin(big)].groupby(nodes.group).first()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=colour[g], label=f"group of {sizes[g]} outlets") for g in big]
    handles += [Line2D([], [], marker="o", ls="", color="#55544f", label="other groups with sisters"),
                Line2D([], [], marker="o", ls="", color="#b4b2a9", label="outlet without sisters"),
                Line2D([], [], marker="o", ls="", markerfacecolor="none", color=MUTED, label="not in the final dataset")]
    a.legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2,
             handletextpad=0.2, columnspacing=0.8)
    isolated = len(nodes) - len(shown)
    a.set_title(f"(a) outlets sharing articles ({len(shown)} of {len(nodes)}; {isolated} share none)", fontsize=8.5)
    s = sim[sim.has_sister]
    lim = (min(s.nearest_other.min(), s.nearest_sister.min()) - 0.02, 1.0)
    b.plot(lim, lim, color=MUTED, lw=0.8, ls="--", zorder=1)
    b.scatter(s.nearest_other, s.nearest_sister, s=16, color="#55544f", edgecolors=SURFACE, linewidths=0.5, zorder=2)
    above = (s.nearest_sister > s.nearest_other).mean()
    b.text(0.04, 0.96, f"{above:.0%} closer to a sister", transform=b.transAxes, va="top", fontsize=7.5, color=INK)
    b.set_xlim(lim); b.set_ylim(lim); b.set_aspect("equal")
    b.set_xlabel("similarity to the nearest outlet\nof another group", fontsize=8)
    b.set_ylabel("similarity to the nearest sister", fontsize=8)
    b.set_title(f"(b) {len(s)} outlets with a sister, after deduplication", fontsize=8.5)
    plt.tight_layout(); plt.savefig(HERE / "network.pdf", bbox_inches="tight"); plt.close()
    print("network.pdf")


def anonymization_figure():
    """(a) articles with at least one replacement of each kind; (b) articles with the outlet's own name, after each
    pass; (c) replacements per 1,000 words by NewsGuard level."""
    t = pd.read_csv(FIGDATA / "anonymization.csv")
    kinds = {"url": "links [URL]", "email": "e-mails [EMAIL]", "own_name": "own name [OUTLET]",
             "gliner_outlet": "other outlets [OUTLET]", "gliner_journalist": "journalists [AUTHOR]"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 0.95, 1.25]})
    share = [(t[k] > 0).mean() for k in kinds]
    y = np.arange(len(kinds))[::-1]
    a.barh(y, share, 0.6, color=SLOTS, edgecolor=SURFACE, lw=1)
    for yy, v in zip(y, share):
        a.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=7, color=INK)
    a.set_yticks(y, list(kinds.values()), fontsize=7); a.set_xlim(0, 1); a.grid(axis="y", visible=False)
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_title("(a) articles with a replacement", fontsize=8.5)
    stages = [("original", "own_name_left_raw"), ("after\npasses 1–2", "own_name_left_rules"), ("after\nGLiNER", "own_name_left_final")]
    v = [t[col].mean() for _, col in stages]
    b.bar(range(3), v, 0.6, color=[MUTED, SLOTS[2], SLOTS[0]], edgecolor=SURFACE, lw=1)
    for i, vv in enumerate(v):
        b.text(i, vv + 0.01, f"{vv:.1%}", ha="center", fontsize=7, color=INK)
    b.set_xticks(range(3), [s for s, _ in stages], fontsize=6.5); b.grid(axis="x", visible=False)
    b.set_ylim(0, max(v) * 1.15)
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_title("(b) own name left", fontsize=8.5)
    per = t[list(kinds)].div(t.words, axis=0) * 1000
    m = per.groupby(t.level).mean().reindex(["l1", "l2", "l3", "l4", "l5"])
    bottom = np.zeros(len(m))
    for k, col in zip(kinds, SLOTS):
        c.bar(range(len(m)), m[k], 0.62, bottom=bottom, color=col, edgecolor=SURFACE, lw=1)
        bottom += m[k].to_numpy()
    for i, v in enumerate(bottom):
        c.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    c.set_xticks(range(len(m)), ["0–39", "40–59", "60–74", "75–99", "100"], fontsize=6.5); c.grid(axis="x", visible=False)
    c.set_xlabel("NewsGuard score of the outlet", fontsize=7.5)
    c.set_title("(c) replacements per 1,000 words", fontsize=8.5)
    c.set_ylim(0, bottom.max() * 1.15)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col, label=lab) for col, lab in zip(SLOTS, kinds.values())], loc="lower center",
               ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06), handlelength=1)
    plt.tight_layout(); plt.savefig(HERE / "anonymization.pdf", bbox_inches="tight"); plt.close()
    print("anonymization.pdf")


def curves_figure():
    folder = DATA / "predictions" / "regression"
    done = [m for m in MODELS if len(list((folder / m).glob("fold*.csv.gz"))) == 5]
    if not done or not (DATA / "articles_anonymous.csv.gz").exists():
        print("curves.pdf: predictions of the current run not here yet, skipped"); return
    from scipy.stats import spearmanr

    import boost as BO
    art, outlets, rows = BO.load()
    y = outlets.score.to_numpy()
    K = [1, 2, 3, 5, 10, 20, 50, "all"]; X = [1, 2, 3, 5, 10, 20, 50, 160]
    draws = [d for f in range(1, 6) for d in BO.test_draws(outlets, rows, f)]
    base = np.array([outlets.score[outlets.fold != f].mean() for f in outlets.fold])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for m in done:
        p = pd.concat([pd.read_csv(f) for f in sorted((folder / m).glob("fold*.csv.gz"))]).set_index("article_id").prediction
        p = p.reindex(art.article_id).to_numpy()
        rho, mae = [], []
        for k in K:
            t = pd.DataFrame([(d, j, p[r].mean()) for d, kk, j, r in draws if kk == str(k)], columns=["domain", "draw", "p"])
            M = t.pivot_table(index="draw", columns="domain", values="p").reindex(columns=outlets.index).to_numpy()
            rho.append(np.mean([spearmanr(y, row).correlation for row in M])); mae.append(np.mean(np.abs(M - y).mean(1)))
        a.plot(X, rho, "o-", ms=3, lw=1.4, color=MODELS[m], label=m); b.plot(X, mae, "o-", ms=3, lw=1.4, color=MODELS[m])
    b.axhline(np.abs(base - y).mean(), color="#444444", ls=":", lw=1); b.text(1, np.abs(base - y).mean(), " no text", va="bottom", fontsize=8)
    for ax, t in [(a, "Spearman (outlets)"), (b, "MAE (points)")]:
        ax.set_xscale("log"); ax.set_xticks(X, [str(k) for k in K]); ax.minorticks_off()
        ax.set_xlabel("articles read per outlet"); ax.set_title(t, fontsize=9)
    a.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(HERE / "curves.pdf", bbox_inches="tight"); plt.close()
    print("curves.pdf")


if __name__ == "__main__":
    dataset_figure()
    if (FIGDATA / "summary.json").exists():
        groups_figure(); network_figure(); anonymization_figure()
    else:
        print("dataset construction figures: run dataset_construction.py on the server first, skipped")
    curves_figure()
