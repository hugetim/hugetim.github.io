"""Weighted-average decomposition of each model's capability under XCI 0.1 (Nov 17, 2026), the counterpart of
scripts/eci_weights_beta.py for post 1A's Epoch / XCI 0.1 toggle.

XCI 0.1 (xci.py in github.com/hugetim/xci, vendored by scripts/xci_vendor.py; "XB+ with benchmark-specific
capability noise tau_b") scores one benchmark score y_mb by a censored stretched beta:
    x ~ Beta(a, k - a), a = the median-centred shape (I_mu(a, k - a) = 1/2; median_table.herm_a),
    mu = sigmoid(alpha_b (C - D_b)), k + 1 = 1 / (4 sigma_b^2 + g_b^2 mu (1 - mu))   (xci.kappa_formula),
    g_b = alpha_b tau_b (capability noise in logit units), y* = -delta + (1 + 2 delta) x,
    interior y: density of x at (y + delta) / (1 + 2 delta), over 1 + 2 delta;
    y = 0: P(x <= x0); y = 1: P(x >= 1 - x0), x0 = delta / (1 + 2 delta).
Every parameter is held at its posterior mean (ECI scale), as eci_weights_beta.py holds Epoch's published values.

Readings. An interior score's reading chat_mb is the ECI at which that score is likeliest (the root of its score
function s_mb(C) = d log p(y_mb | C) / dC), as in eci_weights_beta.py. A 0% or 100% score has no such root under
XCI 0.1: its probability keeps rising as C -> -inf or +inf. Its dot is drawn as a bracket at the end of the range
of ECIs it is consistent with (Tim's choice, Nov 17): for 100%, the lowest ECI any score short of 100% could imply,
lim_{y -> 1} chat(y) (finite, since the stretched beta puts y = 1 inside its support); for 0%, lim_{y -> 0} chat(y).

Weights. Ct_m solves sum_b s_mb(Ct) = 0. Writing s_mb(Ct) = w_mb (r_mb - Ct), any positive w_mb with that r_mb makes
Ct = sum_b w_mb r_mb / sum_b w_mb exactly. Interior: r = chat, w = s(Ct) / (chat - Ct) (exact weights, as for
Epoch's model). 0% and 100%: w = the observed information -ds/dC at Ct (Nov 19, 2026; was the Fisher information), and r = Ct + s / w
(a Newton step; it lies beyond Ct in the score's direction), clamped so a 100% is never counted below its bracket
(a 0% never above) and w = s / (r - Ct) then. I_b(C) also draws the
weight ribbons in the chart (fisher_curve).

Inputs (environment): ECI_DATA (an archive's epoch_capabilities_index/), ECI_XCI (the XCI 0.1 fit's folder:
draws.npz, diagnostics.json, eci.csv). Outputs (current directory):
  eci_benchmark_weights_xci.csv  model, date, ECI (XCI 0.1 posterior mean), Ct, benchmark, performance, implied_ECI
                                 (the dot: chat, or the bracket for 0% / 100%), reading_ECI (r), bound (0, 1 or
                                 blank), share, w
  benchmark_curves_xci.csv       benchmark, mid_eci, alpha_eci, sigma, g, delta
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as Beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xci_vendor  # noqa: E402
sys.path.insert(0, xci_vendor.ensure())  # the published XCI code at its pinned commit
from median_table import herm_z  # noqa: E402
from xci import kappa_formula  # noqa: E402

MU_EPS, H = 1e-6, 1e-4  # clip on mu (set from the fit by curves(): Epoch's 1e-6, XCI v0.1 1e-12); step in C (ECI points)


def curves(fit_dir):
    """Posterior-mean parameters per benchmark (ECI scale) from an XCI 0.1 fit."""
    dr = np.load(os.path.join(fit_dir, "draws.npz"))
    diag = json.load(open(os.path.join(fit_dir, "diagnostics.json")))
    assert diag.get("tau_bench") and diag.get("median_centre") and diag.get("var_match") == 1, "not an XCI 0.1 fit"
    global MU_EPS
    MU_EPS = float(diag.get("mu_floor", 1e-6))  # the fit's own clip on mu, used by shapes() and readings()
    b = dr["b"].astype(float)[:, None]
    g = (dr["slope"].astype(float) * b) * dr["tau_b"].astype(float)  # raw slope x raw tau = logit-scale noise sd
    return pd.DataFrame({"benchmark": dr["benchmarks"], "mid_eci": dr["edi"].astype(float).mean(0),
                         "alpha_eci": dr["slope"].astype(float).mean(0), "sigma": dr["sigma"].astype(float).mean(0),
                         "g": g.mean(0), "delta": float(dr["hyper_delta"].astype(float).mean()),
                         "tau_eci": (dr["tau_b"].astype(float) * b).mean(0)})  # tau_b in ECI points (shown, not used)


def shapes(C, D, a, sig, g):
    """The latent beta's shapes at capability C. Works from the logit u = alpha (C - D) (clipped at the fit's floor)
    rather than from mu: near mu = 1 - 1e-12 a change in C below double precision's resolution of mu would otherwise
    read as no change at all (the numerical score function was exactly 0 there)."""
    le = np.log((1 - MU_EPS) / MU_EPS)
    u = np.clip(a * (C - D), -le, le)
    mu = expit(u)
    k = kappa_formula(sig, g, 1.0, mu)  # slope_raw ** 1 * tau = g (XCI v0.1: lambda = 1); mu (1 - mu) enters only as a small term
    z = herm_z(u, np.log(k))                 # the median table, indexed by logit(mu) directly
    return k * expit(z), k * expit(-z)


def loglik(y, C, D, a, sig, g, dl, bound=None):
    """log p(y | C) for arrays; bound: 0 or 1 marks a censored score, None means use y (exact 0 and 1 censored)."""
    A, Bb = shapes(C, D, a, sig, g)
    x0 = dl / (1 + 2 * dl)
    cat = np.where(y <= 0, 0, np.where(y >= 1, 2, 1)) if bound is None else np.full(np.shape(y), 1)
    xin = (np.clip(y, 0, 1) + dl) / (1 + 2 * dl)
    return np.where(cat == 0, Beta.logcdf(x0, A, Bb), np.where(cat == 2, Beta.logsf(1 - x0, A, Bb),
                    Beta.logpdf(xin, A, Bb) - np.log1p(2 * dl)))


def score_fn(y, C, D, a, sig, g, dl, interior=False):
    """d log p / dC by central differences; interior=True scores y as a density even at 0 or 1 (for the brackets)."""
    kw = {"bound": 1} if interior else {}
    return (loglik(y, C + H, D, a, sig, g, dl, **kw) - loglik(y, C - H, D, a, sig, g, dl, **kw)) / (2 * H)


def bisect(f, lo, hi, n=80):
    lo, hi = np.array(lo, float), np.array(hi, float)
    for _ in range(n):
        mid = (lo + hi) / 2
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return (lo + hi) / 2


def readings(y, D, a, sig, g, dl, n=60):
    """chat for each score (arrays of equal shape): the root of its score function, scoring 0% and 100% as the density
    at the ends of the stretched support (the brackets). The root lies where mu is inside the fit's clip, between
    D -+ logit(1 - MU_EPS) / a; outside, the likelihood is flat."""
    le = np.log((1 - MU_EPS) / MU_EPS)
    lo, hi = D - 0.999 * le / a, D + 0.999 * le / a
    f = lambda C: score_fn(y, C, D, a, sig, g, dl, interior=True)
    ok = (f(lo) > 0) & (f(hi) < 0)
    assert ok.all(), f"{(~ok).sum()} readings not bracketed"
    return bisect(f, lo, hi, n)


def fisher(C, D, a, sig, g, dl, nq=400):
    """Expected information E[s(Y; C)^2] of one score at capability C (arrays of equal shape), Y from the model:
    the two censored masses plus the interior, integrated over quantiles of the latent beta."""
    C, D, a, sig, g = np.broadcast_arrays(*map(np.asarray, (C, D, a, sig, g)))
    A, Bb = shapes(C, D, a, sig, g)
    x0 = dl / (1 + 2 * dl)
    F0, F1 = Beta.cdf(x0, A, Bb), Beta.cdf(1 - x0, A, Bb)
    s0 = score_fn(np.zeros_like(C, float), C, D, a, sig, g, dl)
    s1 = score_fn(np.ones_like(C, float), C, D, a, sig, g, dl)
    # an outcome whose probability underflows to 0 adds nothing (its log-probability is -inf and its slope NaN)
    out = np.where(F0 > 0, F0 * s0 ** 2, 0.0) + np.where(F1 < 1, (1 - F1) * s1 ** 2, 0.0)
    for q in (np.arange(nq) + 0.5) / nq:
        x = Beta.ppf(F0 + q * (F1 - F0), A, Bb)
        yy = np.clip(x * (1 + 2 * dl) - dl, 1e-12, 1 - 1e-12)
        out = out + (F1 - F0) / nq * score_fn(yy, C, D, a, sig, g, dl) ** 2
    return out


def fisher_curve(row, C):
    """I_b(C) for one benchmark (a benchmark_curves_xci.csv row) over an array of ECIs."""
    return fisher(C, row.mid_eci, row.alpha_eci, row.sigma, row.g, row.delta)


def main():
    DATA, XCI = os.environ["ECI_DATA"], os.environ["ECI_XCI"]
    bc = curves(XCI)
    dl = float(bc.delta.iloc[0])
    xe = pd.read_csv(os.path.join(XCI, "eci.csv"))
    pub = pd.read_csv(os.path.join(DATA, "eci_scores.csv"))[["Model", "date"]]
    d = pd.read_csv(os.path.join(DATA, "processed_data_for_eci.csv")).drop(columns=["date"], errors="ignore")
    d = d.merge(bc, on="benchmark", how="left")
    assert d.mid_eci.notna().all(), "scores on benchmarks the XCI 0.1 fit lacks"
    d["model"] = d["Model"]
    d = d.merge(xe[["Model", "eci"]].rename(columns={"eci": "ECI"}), on="Model", how="left").merge(pub, on="Model", how="left")
    assert d.ECI.notna().all(), "scores for models the XCI 0.1 fit lacks"
    y = d.performance.to_numpy(float)
    P = [d[c].to_numpy(float) for c in ("mid_eci", "alpha_eci", "sigma", "g")]
    bound = np.where(y <= 0, 0, np.where(y >= 1, 1, -1))
    # dots: chat for interior scores; for 0% and 100%, the limit of chat as the score approaches them (y scored as
    # the density at the end of the stretched support)
    d["implied_ECI"] = readings(y, *P, dl)
    # Ct: each model's root of its summed score function (0% / 100% censored), bracketed around the posterior mean
    # (far outside it every curve is at Epoch's clip on mu, where the likelihood is flat)
    codes, models = pd.factorize(d.model)
    tot = lambda Cm: np.bincount(codes, weights=score_fn(y, Cm[codes], *P, dl), minlength=len(models))
    e0 = d.groupby(codes).ECI.first().to_numpy()
    lo, hi = e0 - 60, e0 + 60
    ok = (tot(lo) > 0) & (tot(hi) < 0)
    assert ok.all(), "no capability within 60 points of the posterior mean for: %s" % list(models[~ok][:8])
    Ct = bisect(tot, lo, hi)
    d["Ct"] = Ct[codes]
    C = d.Ct.to_numpy()
    s = score_fn(y, C, *P, dl)
    gap = d.implied_ECI.to_numpy() - C
    info = fisher(C, *P, dl)
    # 0% and 100% (Tim, Nov 19, 2026): the OBSERVED information -ds/dC at Ct (the local slope of this score's own
    # pull), not the expected (Fisher) information, so that every weight is an observed steepness: interior scores'
    # weights are its average between Ct and chat. Reading = Ct + s / w (a Newton step), on the score's side of Ct.
    hh = 0.05
    obs = -(score_fn(y, C + hh, *P, dl) - score_fn(y, C - hh, *P, dl)) / (2 * hh)
    assert (obs[bound >= 0] > 0).all(), "non-positive observed information for a 0% or 100% score"
    # Clamped at the bracket (Tim, Nov 19): a 100% score is never counted below its bracket (a 0% never above), so the
    # reading never contradicts "at least / at most this"; the weight then follows from the identity, w = s / (r - Ct).
    # (Clamping only moves r further from Ct on the score's side, so w stays positive and only shrinks.)
    imp = d.implied_ECI.to_numpy()
    r_b = C + s / np.where(bound >= 0, obs, 1.0)
    r_b = np.where(bound == 1, np.maximum(r_b, imp), np.where(bound == 0, np.minimum(r_b, imp), r_b))
    exact = np.abs(gap) > 1e-7
    w = np.where(bound >= 0, s / np.where(bound >= 0, r_b - C, 1.0), np.where(exact, s / np.where(exact, gap, 1.0), info))
    assert (w > 0).all(), "non-positive weights"
    d["w"] = w
    d["reading_ECI"] = np.where(bound >= 0, r_b, d.implied_ECI)
    d["bound"] = np.where(bound >= 0, bound, -1)
    d["share"] = d.w / d.groupby("model").w.transform("sum")
    recon = (d.share * d.reading_ECI).groupby(d.model).sum() - d.groupby("model").Ct.first()
    dev = (d.groupby("model").Ct.first() - d.groupby("model").ECI.first()).abs()
    print(f"identity check (sum share*reading - Ct): max {recon.abs().max():.2e}")
    print(f"|Ct - XCI 0.1 posterior mean|: median {dev.median():.3f}, max {dev.max():.3f} ({dev.idxmax()})")
    out = d[["model", "date", "ECI", "Ct", "benchmark", "performance", "implied_ECI", "reading_ECI", "bound", "share", "w"]].copy()
    out["bound"] = out.bound.map({0: "0", 1: "1", -1: ""})
    out.sort_values(["date", "model", "share"], ascending=[True, True, False]).to_csv("eci_benchmark_weights_xci.csv", index=False)
    bc.to_csv("benchmark_curves_xci.csv", index=False)
    print("Wrote eci_benchmark_weights_xci.csv and benchmark_curves_xci.csv")


if __name__ == "__main__":
    main()
