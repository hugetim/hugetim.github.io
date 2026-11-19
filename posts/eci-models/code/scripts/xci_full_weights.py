"""Full-refit weights for post 1A's XCI 0.1 view (Tim, Nov 19, 2026): the counterpart of scripts/eci_full_weights.py.
How much a model's XCI (posterior mean) moves after a full refit, per point of change in one score's implied XCI,
from the XCI 0.1 fit's posterior draws alone (no refit).

Interior scores: the data-perturbation identity d E[XCI_m] / d y = Cov_posterior(XCI_m, d log p(y | theta) / d y),
divided by d chat / d y (chat: the implied XCI, scripts/xci_weights.py's readings(), posterior-mean curves).

0% and 100% scores (Tim's construction): the change in E[XCI_m] if the score were instead the interior score whose
implied XCI is 1 point inside the bracket (100%: bracket - 1; 0%: bracket + 1), by importance-reweighting the
posterior draws with p(y' | theta) / p(y | theta), divided by the distance from the score's reading (the open
circle) to that point. Draws' effective sample size under the reweighting is reported per row (ess). Uses every ECI_FULL_THIN-th draw
(default 2); only the models the chart lists.

Inputs (environment): ECI_XCI (draws.npz, diagnostics.json). Current directory: eci_benchmark_weights_xci.csv,
benchmark_curves_xci.csv (scripts/xci_weights.py). Output: eci_full_weights_xci.csv  model, benchmark, full_w,
full_se (4 blocks of draws), ess (0%/100% only).
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xci_weights as xw  # noqa: E402

XCI = os.environ["ECI_XCI"]
xw.MU_EPS = float(json.load(open(os.path.join(XCI, "diagnostics.json"))).get("mu_floor", 1e-6))
z = np.load(os.path.join(XCI, "draws.npz"))
Mi = {m: i for i, m in enumerate(z["models"])}
Bi = {b: j for j, b in enumerate(z["benchmarks"])}
THIN = int(os.environ.get("ECI_FULL_THIN", "2"))  # every 2nd draw (4000 of 8000): keeps the daily build within its time limit
E, D, A, S, T = (z[k].astype(float)[::THIN] for k in ("eci", "edi", "slope", "sigma", "tau_b"))
braw, dl = z["b"].astype(float)[::THIN], z["hyper_delta"].astype(float)[::THIN]
w = pd.read_csv("eci_benchmark_weights_xci.csv")
cv = pd.read_csv("benchmark_curves_xci.csv").set_index("benchmark")
blocks = np.array_split(np.arange(E.shape[0]), 4)
EPS = 1e-4


def ll(y, i, j):
    """log p(y | theta_s) over the draws for model i, benchmark j (y = 0 or 1 censored, as in the fit)."""
    return xw.loglik(np.full(E.shape[0], y), E[:, i], D[:, j], A[:, j], S[:, j], A[:, j] * braw * T[:, j], dl)


def chat(y, c):
    """implied XCI of score y on the posterior-mean curve c (a benchmark_curves_xci.csv row)."""
    return float(xw.readings(np.array([y]), np.array([c.mid_eci]), np.array([c.alpha_eci]), np.array([c.sigma]),
                             np.array([c.g]), float(c.delta))[0])


def cov(a, b):
    return np.mean((a - a.mean()) * (b - b.mean()))


def chat_v(y, c):
    """implied XCI for arrays of scores y on curves c (rows of benchmark_curves_xci.csv, aligned with y)."""
    return xw.readings(y, c.mid_eci.to_numpy(), c.alpha_eci.to_numpy(), c.sigma.to_numpy(), c.g.to_numpy(),
                       float(c.delta.iloc[0]))


# Only the models post 1A's chart lists (its rule in viz/build_models.py: frontier models since GPT-4, plus models
# released since GPT-5 within 8 points of the frontier at release), to keep the daily build fast.
m = w.groupby("model").agg(ECI=("ECI", "first"), date=("date", "first")).reset_index()
m = m[m.date >= "2023-03-14"].sort_values(["date", "ECI"], ascending=[True, False])
fr = m[m.ECI > m.ECI.shift(1).cummax().fillna(-np.inf)]
m["gap"] = m.ECI.cummax() - m.ECI
shown = set(fr.model) | set(m[(m.date >= "2025-08-07") & (m.gap <= 8.0)].model)
w = w[w.model.isin(Mi) & w.benchmark.isin(Bi) & w.model.isin(shown)].reset_index(drop=True)
inner = w.bound.isna().to_numpy()
cw = cv.loc[w.benchmark]
yi = w.performance.clip(2 * EPS, 1 - 2 * EPS).to_numpy()
dchat = np.full(len(w), np.nan)
dchat[inner] = (chat_v(yi[inner] + EPS, cw[inner]) - chat_v(yi[inner] - EPS, cw[inner])) / (2 * EPS)
# 0% / 100%: the interior score whose implied XCI is 1 point inside the bracket (bisection on y; chat increases in y)
bd = ~inner
up = (w.bound == 1).to_numpy()
target = np.where(up, w.implied_ECI - 1, w.implied_ECI + 1)
lo, hi = np.full(bd.sum(), 1e-9), np.full(bd.sum(), 1 - 1e-9)
for _ in range(50):
    mid = (lo + hi) / 2
    below = chat_v(mid, cw[bd]) < target[bd]
    lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
yp = np.full(len(w), np.nan)
yp[bd] = (lo + hi) / 2

def ll_v(y, ii, jj):
    """log p(y_n | theta_s) for draws x rows (rows: model ii, benchmark jj, score y)."""
    yy = np.broadcast_to(y, (E.shape[0], len(ii))).copy()
    return xw.loglik(yy, E[:, ii], D[:, jj], A[:, jj], S[:, jj], A[:, jj] * braw[:, None] * T[:, jj], dl[:, None])


def covc(a, b):  # column-wise covariance over draws
    return ((a - a.mean(0)) * (b - b.mean(0))).mean(0)


ii = w.model.map(Mi).to_numpy(); jj = w.benchmark.map(Bi).to_numpy()
full_w, full_se, ess = np.full(len(w), np.nan), np.full(len(w), np.nan), np.full(len(w), np.nan)
for ch in np.array_split(np.arange(len(w)), max(1, len(w) // 200)):
    Ei = E[:, ii[ch]]
    ci, cb = ch[inner[ch]], ch[~inner[ch]]
    if len(ci):
        k = inner[ch]
        g = (ll_v(yi[ci] + EPS, ii[ci], jj[ci]) - ll_v(yi[ci] - EPS, ii[ci], jj[ci])) / (2 * EPS)
        full_w[ci] = covc(Ei[:, k], g) / dchat[ci]
        full_se[ci] = np.std([covc(Ei[b][:, k], g[b]) for b in blocks], axis=0, ddof=1) / 2 / np.abs(dchat[ci])
    if len(cb):
        k = ~inner[ch]
        lw = ll_v(yp[cb], ii[cb], jj[cb]) - ll_v(np.where(up[cb], 1.0, 0.0), ii[cb], jj[cb])
        wt = np.exp(lw - lw.max(0))
        ess[cb] = wt.sum(0) ** 2 / (wt ** 2).sum(0)
        Eb = Ei[:, k]
        dE = lambda ix: (wt[ix] * Eb[ix]).sum(0) / wt[ix].sum(0) - Eb[ix].mean(0)
        dist = target[cb] - w.reading_ECI.to_numpy()[cb]
        full_w[cb] = dE(slice(None)) / dist
        full_se[cb] = np.std([dE(b) for b in blocks], axis=0, ddof=1) / 2 / np.abs(dist)
pd.DataFrame(dict(model=w.model, benchmark=w.benchmark, full_w=full_w, full_se=full_se, ess=ess)).to_csv(
    "eci_full_weights_xci.csv", index=False)
