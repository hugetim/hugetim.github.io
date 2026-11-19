"""Variance-matched capability noise (Nov 16, 2026, Tim): instead of averaging the XB likelihood over capability noise
eps ~ N(0, tau) by quadrature, give the beta the same mean and approximately the same variance,
    Var(Y) ~ 4 sigma_b^2 mu (1 - mu) + (alpha_b^lambda tau)^2 (mu (1 - mu))^2,
i.e. kappa + 1 = mu (1 - mu) / Var = 1 / (4 sigma_b^2 + (alpha_b^lambda tau)^2 mu (1 - mu)). alpha_b is the raw-scale
slope (geometric mean 1) and tau is on the raw capability scale; lambda (fixed per fit) = 1 is capability-scale noise
(logit-scale sd alpha_b tau), lambda = 0 logit-scale noise the same for every benchmark. s2 is capped smoothly (kappa_formula).
vm_kappa: the numpy version for the scoring scripts, from a fit's draws (slopes stored on the ECI scale, slope_raw =
slope_eci * b, b the draw's ECI points per raw unit).
"""
import numpy as np

S2_MAX, SHARP = 0.7, 50.0  # smooth cap on the variance factor s2: kappa >= 1 / 0.7 - 1 = 0.43 (table edge 0.3)


def kappa_formula(sigma, slope_raw, tau, lam, mu, xp=np):
    """kappa + 1 = 1 / s2, s2 = 4 sigma^2 + (slope^lam tau)^2 mu (1 - mu), with s2 capped smoothly at S2_MAX by a
    soft minimum, -log(exp(-SHARP s2) + exp(-SHARP S2_MAX)) / SHARP. A hard floor on kappa (the first version, Nov 16)
    gave flat, kinked regions where chains that wandered to a large tau got stuck (162 divergences, step size 1e-4);
    the soft minimum equals s2 to ~1e-10 for s2 <= 0.25 (plausible values reach ~0.2) and is smooth everywhere."""
    s2 = 4 * sigma ** 2 + (slope_raw ** lam * tau) ** 2 * mu * (1 - mu)
    s2 = -xp.logaddexp(-SHARP * s2, -SHARP * S2_MAX) / SHARP
    return 1 / s2 - 1


def vm_kappa(dr, diag, idx, bi, sigma, slope_eci, mu):
    """kappa per (draw, score). sigma: per score (draws, scores); slope_eci: (draws, benchmarks) on the ECI scale;
    mu: (draws, scores); lambda from the fit's diagnostics (var_match)."""
    b = dr["b"][idx].astype(float)[:, None]
    tau = (dr["tau_b"][idx][:, bi] if "tau_b" in dr.files else dr["hyper_tau_cap"][idx][:, None]).astype(float)
    lam = float(diag["var_match"])
    slope_raw = slope_eci[:, bi] * b
    return kappa_formula(sigma, slope_raw, tau, lam, mu)
