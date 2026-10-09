"""Compact two-panel version of the bias and publisher-level figures for the paper.

Same data and styling as fig_med_by_leaning and fig_publisher_metrics, resized to
a short, wide format so the two can be stacked as Fig. 3a/3b at text width.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from irgood import aggregate, bias, config as C, data, plots  # noqa: E402

W, H = 8.6, 2.7

plots.set_style(use_latex=True)
preds = data.load_all_predictions(include_gpt=False, with_leaning=True)


def compact(fig, name):
    fig.set_size_inches(W, H)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    if leg is not None:
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        title = leg.get_title().get_text()
        leg.remove()
        ax.legend(handles, labels, title=title or None, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)
    fig.tight_layout()
    out = C.PAPER_IMG_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("->", out)


med = bias.med_by_leaning(preds, over="errors")
f1 = plots.fig_med_by_leaning(med, name="_tmp_med")
f1.axes[0].set_ylabel(r"MED $(\hat{y} - y)$")
compact(f1, "bias_med_by_leaning")

pub = aggregate.aggregate_to_publisher(preds)
tbl = aggregate.publisher_metrics(pub)
y_pub = pub[pub["model"] == C.MODEL_ORDER[0]]["y_true"]
f2 = plots.fig_publisher_metrics(tbl, y_pub, name="_tmp_pub")
compact(f2, "publisher_metrics_compact")

for n in ("_tmp_med", "_tmp_pub"):
    (C.FIGURES_DIR / f"{n}.pdf").unlink(missing_ok=True)
