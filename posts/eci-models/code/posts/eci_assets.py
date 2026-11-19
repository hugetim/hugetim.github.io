"""Build everything the two ECI posts need. Run from anywhere:
    python posts/eci_assets.py
Runs the weight decomposition and chart builders, copies the interactive charts into each post's folder,
and writes every file the posts' cells read into the post folders, so a post folder is self-contained
(it can be copied into the blog repo and rendered there).
  posts/eci-models    (post 1A, comparing models): model panels, weights, ECI scores, rescaled benchmarks,
                      benchmark pages and domains, stats, data date
  posts/eci-point (post 1B, what a point means): weight curves, units table, METR conversion,
                      ECI scores, stats, data date

Weights: if the archive folder holds bayes/edi.csv (scripts/bayes_eci.py's fit of Epoch's Bayesian model on that
archive, which supplies the per-benchmark scatter sigma_b Epoch's release omits), the decomposition is
scripts/eci_weights_beta.py (Epoch's beta likelihood, published ECI, difficulties and slopes); otherwise
scripts/eci_weights.py (least squares, the method before Nov 4, 2026), as for data/current.

Environment (all optional):
  ECI_DATA_ROOT  unpacked Epoch archive. Default: data/live (kept current by scripts/update_eci_posts.py)
                 if it exists, else data/current (the research copy).
  ECI_BUILD      working directory for intermediate files. Default: the repo root when building from
                 data/current, else build/live, so research outputs at the repo root
                 (eci_benchmark_weights.csv etc.) are never overwritten with newer data.
  ECI_POSTS_OUT  where to write the post folders' files (default: posts/); the updater builds into a
                 staging copy and installs only after every check and render has passed.
"""
import json, os, shutil, subprocess, sys
import numpy as np
import pandas as pd

POSTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(POSTS, ".."))
CURRENT = os.path.join(ROOT, "data", "current")
_live = os.path.join(ROOT, "data", "live")
_live = _live if os.path.exists(_live) else CURRENT
DATA = os.path.abspath(os.environ.get("ECI_DATA_ROOT", _live))
BUILD = os.path.abspath(os.environ.get("ECI_BUILD", ROOT if DATA == os.path.abspath(CURRENT) else os.path.join(ROOT, "build", "live")))
OUT = os.path.abspath(os.environ.get("ECI_POSTS_OUT", POSTS))
A = os.path.join(OUT, "eci-models")
B = os.path.join(OUT, "eci-point")
os.makedirs(A, exist_ok=True); os.makedirs(B, exist_ok=True)
RATE = 14.0  # Epoch's reasoning-era frontier rate, ECI points per year
SHOW = ["FrontierCode", "GPQA diamond", "METR Time Horizons", "OTIS Mock AIME 2024-2025",
        "FrontierMath-Tiers-1-3-v2-Private", "ARC-AGI-2", "Terminal Bench", "MMLU",  # HLE dropped (Epoch removed it from ECI 2026-10-20); SWE-Bench verified -> FrontierCode (removed 2026-10-31)
        "DeepResearch Bench", "Furniture Assembly"]
os.makedirs(BUILD, exist_ok=True)
# Ridge strength: Epoch's fit.py default, 0.1. (Oct 20-30, 2026 releases were reproduced better with 0.1032;
# dropped Oct 31, when 0.1 matched better again. See ECI_PROJECT_BRIEF.md.)
RIDGE = os.environ.get("ECI_RIDGE", "0.1")
env = dict(os.environ, ECI_DATA_ROOT=DATA, ECI_DATA=os.path.join(DATA, "epoch_capabilities_index"), ECI_RAW=DATA,
           ECI_REPO=os.path.join(ROOT, "benchmark-stitching"), PYTHONPATH=os.path.join(ROOT, "scripts"), ECI_RIDGE=RIDGE)
BAYES = os.path.join(DATA, "bayes")
BETA = os.path.exists(os.path.join(BAYES, "edi.csv"))
env["ECI_BAYES"] = BAYES
weights = "scripts/eci_weights_beta.py" if BETA else "scripts/eci_weights.py"
# XCI 0.1 (Nov 17, 2026): post 1A's tool can switch to it when the data folder has its fit (scripts/xci_weights.py)
XCI = os.path.join(DATA, "xci_01")
env["ECI_XCI"] = XCI
for f in ("eci_benchmark_weights_xci.csv", "benchmark_curves_xci.csv"):  # never a stale XCI file from older data
    if os.path.exists(os.path.join(BUILD, f)):
        os.replace(os.path.join(BUILD, f), os.path.join(BUILD, f + ".old"))
xci = ["scripts/xci_weights.py"] if os.path.exists(os.path.join(XCI, "draws.npz")) else []
env["ECI_PANEL_ORDER"] = "custom"  # Tim, Nov 18: rows in viz/benchmark_order.csv (related benchmarks grouped; was difficulty, Nov 17)
# Full-refit weights (Nov 19, 2026): the ticks on post 1A's weight bars, from our fit's posterior draws
full = ["scripts/eci_full_weights.py"] if BETA and os.path.exists(os.path.join(BAYES, "draws.npz")) else []
if os.path.exists(os.path.join(BUILD, "eci_full_weights.csv")):  # never stale ticks from older data
    os.replace(os.path.join(BUILD, "eci_full_weights.csv"), os.path.join(BUILD, "eci_full_weights.csv.old"))
for cmd in [weights] + full + xci + ["viz/build_viz.py", "viz/build_models.py", "scripts/unit_meaning.py"]:
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, cmd)], cwd=BUILD, env=env, check=True, stdout=subprocess.DEVNULL)
# The code behind post 1A's chart goes into the post's folder too (Tim, Nov 19, 2026: any code used to make a post that
# is not in its .qmd or in a separate repo is published with it), under its paths in this repo.
# Not included, being available elsewhere (Tim): the daily updater, and the fits (Epoch's model, scripts/bayes_eci.py:
# github.com/hugetim/eci-clip-replication; XCI v0.1: github.com/hugetim/xci).
CODE_1A = ["posts/eci_assets.py", "scripts/eci_weights_beta.py", "scripts/eci_full_weights.py", "scripts/xci_weights.py",
           "scripts/median_beta.py", "scripts/xb_kappa.py", "viz/build_models.py", "viz/_style.py",
           "viz/benchmark_order.csv", "viz/epoch_benchmark_pages.csv"]
for f in CODE_1A:
    os.makedirs(os.path.join(A, "code", os.path.dirname(f)), exist_ok=True)
    shutil.copy(os.path.join(ROOT, f), os.path.join(A, "code", f))
shutil.copy(os.path.join(ROOT, "viz", "post_code_readme_eci-models.md"), os.path.join(A, "code", "README.md"))
for f, d in (("eci_model_panels.html", A), ("eci_weight_curves.html", B)):
    shutil.copy(os.path.join(BUILD, f), d)

# Files post 1A reads
w = pd.read_csv(os.path.join(BUILD, "eci_benchmark_weights.csv"))
w[["model", "date", "ECI", "benchmark", "performance", "implied_ECI", "share"]].to_csv(os.path.join(A, "weights.csv"), index=False)
eci = pd.read_csv(os.path.join(DATA, "epoch_capabilities_index", "eci_scores.csv"))
for d in (A, B):
    eci.to_csv(os.path.join(d, "eci_scores.csv"), index=False)
meta = pd.read_csv(os.path.join(DATA, "benchmark_metadata.csv"))
meta = meta[meta.benchmark.isin(w.benchmark.unique())]
adj = meta[(meta.random_baseline >= 0.01) | (meta.score_ceiling < 1)][["benchmark", "random_baseline", "score_ceiling"]]
adj.to_csv(os.path.join(A, "rescaled_benchmarks.csv"), index=False)
shutil.copy(os.path.join(ROOT, "viz", "epoch_benchmark_pages.csv"), os.path.join(A, "benchmark_pages.csv"))
shutil.copy(os.path.join(ROOT, "influence", "benchmark_domains.csv"), os.path.join(A, "benchmark_domains.csv"))

# Scalars quoted in the text: readings per model, and how closely the exact weights reproduce published ECI
# (least squares: up to Epoch's small ridge term; beta: the weighted average is the model's capability given the
# benchmarks' fitted curves, which differs slightly from Epoch's posterior mean)
n_read = w.groupby("model").size()
recon = w.groupby("model")[["share", "implied_ECI", "ECI"]].apply(
    lambda x: abs((x.share * x.implied_ECI).sum() - x.ECI.iloc[0]))
stats = pd.DataFrame([{"rate": RATE, "n_read_min": n_read.min(), "n_read_max": n_read.max(),
                       "recon_median": recon.median(), "recon_max": recon.max(), "recon_max_model": recon.idxmax(),
                       "recon_n_over_0_1": int((recon > 0.1).sum()), "method": "beta" if BETA else "least_squares"}])
if BETA:  # per-benchmark scatter around the curve, from our fit of Epoch's model
    fit = pd.read_csv(os.path.join(BAYES, "edi.csv"))
    fit = fit[fit.benchmark.isin(w.benchmark.unique())]
    fit[["benchmark", "sigma", "sigma_q05", "sigma_q95"]].to_csv(os.path.join(A, "benchmark_fit.csv"), index=False)
    lo, hi = fit.loc[fit.sigma.idxmin()], fit.loc[fit.sigma.idxmax()]
    stats = stats.assign(sigma_min=lo.sigma, sigma_min_bench=lo.benchmark, sigma_max=hi.sigma, sigma_max_bench=hi.benchmark,
                         sigma_median=fit.sigma.median(), recon_max_tool=np.nan)
# Data date: written by scripts/update_eci_posts.py; otherwise the date of the newest model in the archive
info_path = os.path.join(DATA, "ARCHIVE_INFO.json")
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
info.setdefault("data_date", str(pd.to_datetime(eci.date).max().date()))
for d in (A, B):
    stats.to_csv(os.path.join(d, "post_stats.csv"), index=False)
    json.dump(info, open(os.path.join(d, "data_info.json"), "w"), indent=1)

# Files post 1B reads
edi = pd.read_csv(os.path.join(DATA, "epoch_capabilities_index", "edi_scores.csv"))
sig = lambda x: 1 / (1 + np.exp(-x))
t = edi.rename(columns={"benchmark_name": "benchmark", "edi": "eci_at_50", "estimated_slope_scaled": "slope"})
t["points_10_to_90"] = 2 * np.log(9) / t.slope
t["years_10_to_90"] = t.points_10_to_90 / RATE
t["points_25_to_75"] = 2 * np.log(3) / t.slope  # the near-linear middle of the curve
t["pp_per_point"] = 50 / t.points_25_to_75      # percentage points of score per ECI point, 25%-75%
t["after_one_year_from_50"] = sig(RATE * t.slope)
t["after_one_year_from_10"] = sig(np.log(1 / 9) + RATE * t.slope)
t["show"] = t.benchmark.isin(SHOW)
t.sort_values("slope").to_csv(os.path.join(B, "benchmark_units.csv"), index=False)
missing = sorted(set(SHOW) - set(t.benchmark))
if missing:
    print("WARNING: benchmarks for the units table not in the data:", missing)

m = pd.read_csv(os.path.join(BUILD, "unit_metr_vs_eci.csv"))
b = np.polyfit(m.eci, m.l2h, 1)[0]
pd.DataFrame([{"n_models": len(m), "doublings_per_point": b, "doublings_per_year": RATE * b,
               "doubling_months": 12 / (RATE * b)}]).to_csv(os.path.join(B, "metr_units.csv"), index=False)
print("assets built in", A, "and", B, "from", DATA)
