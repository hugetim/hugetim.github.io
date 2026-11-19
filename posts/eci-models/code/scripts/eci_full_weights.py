"""Full-refit weights for post 1A's chart (Tim, Nov 19, 2026): how much a model's ECI moves, after a full refit of
Epoch's Bayesian model (every benchmark's curve re-estimated too), per point of change in one score's implied ECI.

From the posterior draws alone (no refit), by the data-perturbation identity
    d E[ECI_m] / d y_mb = Cov_posterior(ECI_m, d log p(y_mb | theta) / d y_mb),
taken on the logit scale of the (clipped) score and divided by d chat / d logit(y), where chat is the chart's
implied ECI (scripts/eci_weights_beta.py: the ECI at which the score is most likely, curves held fixed).
Checked against full refits on Nov 19 (results/weights_validation_2026-11-19/): within the Monte Carlo error.
The chart's weights (eci_benchmark_weights.csv, curves fixed) sum to 1 per model; these need not.

Inputs (environment): ECI_BAYES (draws.npz of scripts/bayes_eci.py), ECI_DATA (processed_data_for_eci.csv);
current directory: eci_benchmark_weights.csv, benchmark_curves_eci.csv.
Output (current directory): eci_full_weights.csv  model, benchmark, full_w, full_se (4 chain blocks)."""
import os
import numpy as np
import pandas as pd
from scipy.special import digamma, expit, logit

CLIP, MU_EPS = 1e-3, 1e-6
z = np.load(os.path.join(os.environ["ECI_BAYES"], "draws.npz"))
M = {m: i for i, m in enumerate(z["models"])}
Bi = {b: j for j, b in enumerate(z["benchmarks"])}
E, D, A, S = (z[k].astype(float) for k in ("eci", "edi", "slope", "sigma"))
w = pd.read_csv("eci_benchmark_weights.csv")
cv = pd.read_csv("benchmark_curves_eci.csv").set_index("benchmark")


def chat(l, mid, alpha, k):
    """The chart's implied ECI for logit score l (bisection on the beta score equation; vectorised over rows)."""
    def s(C):
        mu = np.clip(expit(alpha * (C - mid)), MU_EPS, 1 - MU_EPS)
        return l - digamma(mu * k) + digamma((1 - mu) * k)
    lo, hi = np.full_like(l, -500.0), np.full_like(l, 800.0)
    for _ in range(80):  # s is decreasing in C
        m_ = (lo + hi) / 2
        pos = s(m_) > 0
        lo, hi = np.where(pos, m_, lo), np.where(pos, hi, m_)
    return (lo + hi) / 2


w = w[w.model.isin(M) & w.benchmark.isin(Bi) & w.benchmark.isin(cv.index)].reset_index(drop=True)
c = cv.loc[w.benchmark]
mid, alpha, kap = c.mid_eci.to_numpy(), c.alpha_eci.to_numpy(), c.kappa.to_numpy()
yc = w.performance.clip(CLIP, 1 - CLIP).to_numpy()
l = logit(yc)
dchat = (chat(l + 1e-4, mid, alpha, kap) - chat(l - 1e-4, mid, alpha, kap)) / 2e-4
i = w.model.map(M).to_numpy(); j = w.benchmark.map(Bi).to_numpy()
mu = np.clip(expit(A[:, j] * (E[:, i] - D[:, j])), MU_EPS, 1 - MU_EPS)            # draws x rows
k = 1 / (4 * S[:, j] ** 2) - 1
g = ((mu * k - 1) / yc - ((1 - mu) * k - 1) / (1 - yc)) * (yc * (1 - yc) / dchat)  # per implied-ECI point
Ei = E[:, i]


def cov(a, b):
    return ((a - a.mean(0)) * (b - b.mean(0))).mean(0)


est = cov(Ei, g)
blocks = np.array_split(np.arange(E.shape[0]), 4)
se = np.std([cov(Ei[b], g[b]) for b in blocks], axis=0, ddof=1) / 2
rows = pd.DataFrame(dict(model=w.model, benchmark=w.benchmark, full_w=est, full_se=se))
rows.to_csv("eci_full_weights.csv", index=False)
