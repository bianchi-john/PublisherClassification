"""Analysis pipeline for the paper
"From Articles to Publishers: Aggregating Language Model Predictions for
News Source Reliability Inference".

Everything the paper reports is produced by `analysis/paper_results.ipynb`,
which is a thin orchestration layer over these modules:

    config     paths, model registry, labels, colours -- the single source of truth
    data       reads the heterogeneous result files into one tidy DataFrame
    metrics    metric definitions (each defined exactly once)
    aggregate  article -> publisher aggregation and the aggregation curve
    bias       political-leaning analysis, including cluster-aware tests
    plots      one function per figure

Design note. The old notebooks bound a model's display name to its data file in
several places independently, and in `4 - Plotter.ipynb` two of those bindings
were crossed, which is how RoBERTa and Llama ended up swapped in the published
article-level F1 and tolerant-accuracy figures. Here that binding exists once,
in `config.FINETUNED_MODELS`, and the name travels with the data.
"""

from . import aggregate, bias, config, data, metrics, plots  # noqa: F401

__all__ = ["config", "data", "metrics", "aggregate", "bias", "plots"]
