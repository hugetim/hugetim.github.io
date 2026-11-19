"""Exact weighted-average decomposition of each model's ECI under Epoch's Bayesian model (Nov 4, 2026).

Inputs (environment):
  ECI_DATA   an archive's epoch_capabilities_index/ folder: Epoch's published ECI (eci_scores.csv),
             benchmark difficulties and slopes (edi_scores.csv) and fit input (processed_data_for_eci.csv)
  ECI_BAYES  output of scripts/bayes_eci.py for the same archive: edi.csv supplies the per-benchmark noise
             level sigma_b (posterior mean), which Epoch's release omits

Epoch's likelihood: score y_mb ~ Beta(mu k_b, (1 - mu) k_b), mu = sigmoid(a_b (C_m - D_b)), k_b = 1/(4 sigma_b^2) - 1.
Holding every benchmark's D_b, a_b (Epoch's published values, ECI scale) and sigma_b fixed, the
log-likelihood's derivative in C from one score is
    s_mb(C) = a_b mu (1 - mu) k_b [logit(y) - psi(mu k_b) + psi((1 - mu) k_b)]
(psi = digamma). It changes sign once, at
    chat_mb  = the ECI at which score y_mb is most likely: the capability that benchmark alone implies.
(For mid-range scores chat is close to inverting the curve, D_b + logit(y)/a_b; near 0% or 100% the beta's
skew moves it by a few points.) The model's capability given the benchmarks, Ct_m, solves sum_b s_mb = 0.
Writing s_mb(Ct) = w_mb (chat_mb - Ct) gives the exact identity
    Ct_m = sum_b w_mb chat_mb / sum_b w_mb,   w_mb = s_mb(Ct_m) / (chat_mb - Ct_m) > 0.
Its linearized form is the Fisher information of one score,
    w_mb ~ [a_b mu (1 - mu)]^2 k_b^2 [psi'(mu k_b) + psi'((1 - mu) k_b)]  ~  a_b^2 mu (1 - mu) / (4 sigma_b^2),
against [a_b mu (1 - mu)]^2 under least squares. Ct_m is close to Epoch's published ECI (a posterior mean
that also averages over the benchmarks' uncertainty) but not equal to it; the files carry both.

Outputs (current directory), same layout as scripts/eci_weights.py:
  eci_benchmark_weights.csv  model, date, ECI (published), Ct, benchmark, performance, phat (mu at the
                             published ECI), implied_ECI (chat), curve_ECI (D + logit(y)/a), share, w_lin
  benchmark_curves_eci.csv   benchmark, mid_eci, alpha_eci, sigma, kappa, eci_at_10/25/75/90
"""
import os
import numpy as np
import pandas as pd
from scipy.special import digamma, expit, logit, polygamma

CLIP, MU_EPS = 1e-3, 1e-6  # Epoch's score clip and its clip on mu (eci.bayesian)
DATA, BAYES = os.environ["ECI_DATA"], os.environ["ECI_BAYES"]

pub = pd.read_csv(os.path.join(DATA, "eci_scores.csv"))
edi = pd.read_csv(os.path.join(DATA, "edi_scores.csv")).rename(
    columns={"benchmark_name": "benchmark", "edi": "mid_eci", "estimated_slope_scaled": "alpha_eci"})
sig = pd.read_csv(os.path.join(BAYES, "edi.csv"))[["benchmark", "sigma"]]
bc = edi.merge(sig, on="benchmark", how="left")
assert bc.sigma.notna().all(), "no sigma for: %s" % list(bc.benchmark[bc.sigma.isna()])
bc["kappa"] = 1 / (4 * bc.sigma ** 2) - 1

d = pd.read_csv(os.path.join(DATA, "processed_data_for_eci.csv")).drop(columns=["date"], errors="ignore")
d = d.merge(bc[["benchmark", "mid_eci", "alpha_eci", "kappa"]], on="benchmark", how="left")
assert d.kappa.notna().all(), "scores on benchmarks without curves"
d["model"] = d["Model"]
d = d.merge(pub[["Model", "eci", "date"]].rename(columns={"eci": "ECI"}), on="Model", how="left")
assert d.ECI.notna().all(), "scores for models without a published ECI"
y = d.performance.clip(CLIP, 1 - CLIP).to_numpy()
a, D, k = d.alpha_eci.to_numpy(), d.mid_eci.to_numpy(), d.kappa.to_numpy()
ly = logit(y)


def score(C):
    mu = np.clip(expit(a * (C - D)), MU_EPS, 1 - MU_EPS)
    return a * mu * (1 - mu) * k * (ly - digamma(mu * k) + digamma((1 - mu) * k))


def bisect(f, lo, hi, n=200):
    """Vectorized bisection for roots of f, assumed positive at lo and negative at hi."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    for _ in range(n):
        mid = (lo + hi) / 2
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return (lo + hi) / 2


# chat: each score's own root (bracket checked)
LO, HI = -2000.0, 2000.0
assert (score(np.full_like(y, LO)) > 0).all() and (score(np.full_like(y, HI)) < 0).all()
d["implied_ECI"] = bisect(score, np.full_like(y, LO), np.full_like(y, HI))
d["curve_ECI"] = D + ly / a

# Ct: each model's root of the summed score
codes, models = pd.factorize(d.model)
tot = lambda Cm: np.bincount(codes, weights=score(Cm[codes]), minlength=len(models))
lo, hi = np.full(len(models), -1000.0), np.full(len(models), 1500.0)
assert (tot(lo) > 0).all() and (tot(hi) < 0).all()
Ct = bisect(tot, lo, hi)
d["Ct"] = Ct[codes]
s = score(d.Ct.to_numpy())
gap = d.implied_ECI.to_numpy() - d.Ct.to_numpy()
mu_t = np.clip(expit(a * (d.Ct - D)), MU_EPS, 1 - MU_EPS)
fisher = (a * mu_t * (1 - mu_t)) ** 2 * k ** 2 * (polygamma(1, mu_t * k) + polygamma(1, (1 - mu_t) * k))
w = np.where(np.abs(gap) > 1e-7, s / np.where(np.abs(gap) > 1e-7, gap, 1.0), fisher)  # limit at chat = Ct
assert (w > 0).all(), "non-positive exact weights"
d["w_exact"], d["w_lin"] = w, fisher
d["share"] = d.w_exact / d.groupby("model").w_exact.transform("sum")
d["phat"] = expit(a * (d.ECI - D))

recon = d.groupby("model").apply(lambda g: abs((g.share * g.implied_ECI).sum() - g.Ct.iloc[0]), include_groups=False)
dev = d.groupby("model").agg(Ct=("Ct", "first"), ECI=("ECI", "first"))
dev = (dev.Ct - dev.ECI).abs()
print(f"identity check (sum share*chat - Ct): max {recon.max():.2e}")
print(f"|Ct - published ECI|: median {dev.median():.3f}, max {dev.max():.3f} ({dev.idxmax()})")

out = d[["model", "date", "ECI", "Ct", "benchmark", "performance", "phat", "implied_ECI", "curve_ECI", "share", "w_lin"]]
out.sort_values(["date", "model", "share"], ascending=[True, True, False]).to_csv("eci_benchmark_weights.csv", index=False)
for q in (0.1, 0.25, 0.75, 0.9):
    bc[f"eci_at_{int(q*100)}"] = bc.mid_eci + np.log(q / (1 - q)) / bc.alpha_eci
bc.to_csv("benchmark_curves_eci.csv", index=False)
print("Wrote eci_benchmark_weights.csv and benchmark_curves_eci.csv")
