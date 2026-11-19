# Code behind this post's chart

The interactive chart (`eci_model_panels.html`) and the data files next to `index.qmd` are built by the code in
this folder, copied here automatically from the project's working repository each time the post's data is rebuilt.
Paths are as in that repository; run the scripts from its root.

Inputs: Epoch AI's benchmark data and published ECI, plus two fits whose code is published separately:
Epoch's Bayesian ECI model refit for the per-benchmark noise level that Epoch's release omits and for posterior
draws (`scripts/bayes_eci.py` in [hugetim/eci-clip-replication](https://github.com/hugetim/eci-clip-replication),
using Epoch's [eci-public](https://github.com/epoch-research/eci-public) at commit e04929a), and XCI v0.1
([hugetim/xci](https://github.com/hugetim/xci)).

`posts/eci_assets.py` builds the post's files, running

- `scripts/eci_weights_beta.py`: each benchmark's weight in a model's ECI (benchmark curves held fixed) and the
  ECI each score implies;
- `scripts/eci_full_weights.py`: the full-refit weights (the ticks on the weight bars), from the posterior draws;
- `scripts/xci_weights.py` (with `median_beta.py`, `logit_beta.py`, `xb_kappa.py`) and `scripts/xci_full_weights.py`: the
  same for the XCI v0.1 view;
- `viz/build_models.py` (with `viz/_style.py`): the chart itself. Benchmark order: `viz/benchmark_order.csv`;
  names and Epoch page links: `viz/epoch_benchmark_pages.csv`.
