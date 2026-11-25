"""Interactive per-model panels (Plotly) for every frontier model (running-max ECI from GPT-4, Mar 2023)
plus near-frontier models: released on or after GPT-5 (2025-08-07) and within NEAR_GAP ECI points of the
best ECI reached by their release date (same-day releases count). Prev/next steps through frontier models.

Each row is a benchmark the model (or the comparison model) was scored on, sorted by the ECI its scores imply
(averaged over the models shown; highest at the top). Green
ribbon: the weight the benchmark would carry at each ECI level, from a 2.5% to a 97.5% score (ticks every
10%). Dots: the ECI each score implies by itself (hover or tap for the score, the raw score where Epoch
adjusts for a chance floor or a ceiling, and the weight). Dashed lines: the models' ECIs. Bars: weight shares.
Benchmark names link to Epoch's benchmark pages; a dagger marks benchmarks whose scores Epoch adjusts.
Plotly (basic bundle) is inlined so the page works offline and in strict viewers.

Run after scripts/eci_weights.py.  Usage: python viz/build_models.py [extra model ...]

Fixed-pair version (for a post that discusses two models): set ECI_PANEL_PAIR="Model A|Model B" to include only
those two and open on them, ECI_PANEL_SIMPLE=1 to drop the menus, the weight ribbons, the weight bars and the
gap column and show the two models' scores as numbers instead (the benchmark scales stay, drawn out to Epoch's
0.1%-99.9% clip, with ticks at 10%-90%), and place each dot where its score falls on the benchmark's
fitted curve rather than at the ECI the score implies under the beta likelihood. With ECI_PANEL_RULER=implied
the simple chart's scales are redrawn so that both agree: each tick and each dot sits at the ECI that score
implies (where it is most likely), so the scale's ends are where a 0.1% and a 99.9% score point. In pair mode
the benchmarks' curves are also written next to the chart (<output name>_curves.csv), and the two models' scores
(<output name>_scores.csv).
Further options:
ECI_PANEL_LABELS="A|B" for short column headings over the scores, ECI_PANEL_ORDER=difficulty to sort the rows by
the benchmarks' difficulty D_b (the ECI of a 50% score; hardest at the top) instead of by the ECI the scores imply,
and ECI_PANEL_OUT for the output file (default eci_model_panels.html).
"""
import json, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from _style import HEAD

HERE = os.path.dirname(os.path.abspath(__file__))
w = pd.read_csv("eci_benchmark_weights.csv", parse_dates=["date"])
if os.path.exists("eci_full_weights.csv"):  # full-refit weights (scripts/eci_full_weights.py): the bars' ticks
    w = w.merge(pd.read_csv("eci_full_weights.csv"), on=["model", "benchmark"], how="left")
bc = pd.read_csv("benchmark_curves_eci.csv").set_index("benchmark")
DATA = os.environ.get("ECI_DATA_ROOT", os.path.join("data", "current"))  # unpacked Epoch archive
ci = pd.read_csv(os.path.join(DATA, "epoch_capabilities_index", "eci_scores.csv")).set_index("Model")
meta = pd.read_csv(os.path.join(DATA, "benchmark_metadata.csv")).set_index("benchmark")
pages = pd.read_csv(os.path.join(HERE, "epoch_benchmark_pages.csv")).set_index("benchmark")
P_LO, P_HI = 0.01, 0.99  # simple chart's weight-shape ends; the full chart's shapes span the whole line (Tim, Nov 18)
RIB_N = 160  # segments along each weight shape
DMAX = round(w.ECI.max(), 1)
m = w.groupby("model").agg(ECI=("ECI", "first"), date=("date", "first")).reset_index()
m = m[m.date >= "2023-03-14"].sort_values(["date", "ECI"], ascending=[True, False])
fr = m[m.ECI > m.ECI.shift(1).cummax().fillna(-np.inf)]
NEAR_GAP, NEAR_FROM = 8.0, "2025-08-07"  # within 8 points of the frontier at release; released since GPT-5
m["gap"] = m.ECI.cummax() - m.ECI  # sorted by date, then ECI descending, so same-day releases count
near = m[~m.model.isin(fr.model) & (m.date >= NEAR_FROM) & (m.gap <= NEAR_GAP)]
gap = dict(zip(m.model, m.gap))
models = list(fr.model) + list(near.model) + [x for x in sys.argv[1:] if x not in set(fr.model) | set(near.model)]
PAIR = [x for x in os.environ.get("ECI_PANEL_PAIR", "").split("|") if x]
SIMPLE = os.environ.get("ECI_PANEL_SIMPLE") == "1"
RULER = os.environ.get("ECI_PANEL_RULER", "curve")
assert RULER in ("curve", "implied"), f"ECI_PANEL_RULER: 'curve' or 'implied', got {RULER!r}"
ORDER = os.environ.get("ECI_PANEL_ORDER", "implied")
assert ORDER in ("implied", "difficulty", "custom"), f"ECI_PANEL_ORDER: 'implied', 'difficulty' or 'custom', got {ORDER!r}"
OUT = os.environ.get("ECI_PANEL_OUT", "eci_model_panels.html")
LABELS = [x for x in os.environ.get("ECI_PANEL_LABELS", "").split("|") if x] or PAIR  # column headings for the scores
if PAIR:
    assert len(PAIR) == 2 and all(x in set(w.model) for x in PAIR), f"ECI_PANEL_PAIR: need two models in the weights file, got {PAIR}"
    models = PAIR

# Benchmark info: display name, Epoch page, curve, and Epoch's score adjustment (chance floor, ceiling)
used = sorted(set(w[w.model.isin(models)].benchmark))
bench = {}
for b in used:
    rb = float(meta.loc[b, "random_baseline"]) if b in meta.index and pd.notna(meta.loc[b, "random_baseline"]) else 0.0
    c = float(meta.loc[b, "score_ceiling"]) if b in meta.index and pd.notna(meta.loc[b, "score_ceiling"]) else 1.0
    rb = rb if rb >= 0.01 else 0.0  # OTIS's 0.1% baseline changes nothing visible
    pg = pages.loc[b] if b in pages.index else None
    bench[b] = {"name": pg.display if pg is not None else b,
                "short": pg.short if pg is not None and isinstance(pg.short, str) else (pg.display if pg is not None else b),
                "url": f"https://epoch.ai/benchmarks/{pg.slug}" if pg is not None else "https://epoch.ai/benchmarks/search",
                "D": round(bc.loc[b, "mid_eci"], 2), "a": round(bc.loc[b, "alpha_eci"], 5),
                # beta-likelihood concentration (scripts/eci_weights_beta.py); 0 = least-squares weights
                "k": round(float(bc.loc[b, "kappa"]), 4) if "kappa" in bc.columns else 0,
                "s": round(float(bc.loc[b, "sigma"]), 4) if "sigma" in bc.columns else None,  # sigma_b (hover on the name)
                "rb": round(rb, 4), "c": round(c, 4)}
RULER_P = [0.001] + [q / 10 for q in range(1, 10)] + [0.999]  # the scale's ends (Epoch's clip) and its ticks
# Post 1A's full chart (Nov 17, Tim, from the clip post's chart): its 10%-90% ticks also sit at the ECI each score
# implies, so a dot lands on the tick of its score
if RULER == "implied" or (not SIMPLE and "kappa" in bc.columns):
    from scipy.optimize import brentq
    from scipy.special import digamma, expit, logit

    def implied_eci(p, D, a, k):
        """ECI at which a score p is most likely under the beta likelihood (chat in scripts/eci_weights_beta.py)."""
        def score(C):
            mu = np.clip(expit(a * (C - D)), 1e-6, 1 - 1e-6)
            return logit(p) - digamma(mu * k) + digamma((1 - mu) * k)
        return brentq(score, -2000.0, 2000.0)

    for b, v in bench.items():
        assert v["k"] > 0, f"ECI_PANEL_RULER=implied needs the beta weights (kappa) for {b}"
        v["t"] = [round(implied_eci(p, bc.loc[b, "mid_eci"], bc.loc[b, "alpha_eci"], bc.loc[b, "kappa"]), 3) for p in RULER_P]
missing = [b for b in used if b not in pages.index]
if missing:
    print("WARNING: no Epoch page mapped for:", missing, "(add them to viz/epoch_benchmark_pages.csv)")

def rows_for(g, bench):
    """One row per score: benchmark, weight share, score, dot (implied ECI, or a 0%/100% bracket under XCI 0.1), the
    reading the weighted average uses (= the dot except for XCI 0.1's 0% and 100% scores), and the bound ('0', '1' or '')."""
    out = []
    for r in g.itertuples():
        rd = getattr(r, "reading_ECI", r.implied_ECI)
        bd = getattr(r, "bound", None)
        bd = "" if bd is None or pd.isna(bd) or str(bd) == "" else str(int(float(bd)))
        fw, fs, an = getattr(r, "full_w", None), getattr(r, "full_se", None), getattr(r, "anchor", 0)
        full = [round(fw, 4), round(fs, 4), int(an == 1)] if fw is not None and not pd.isna(fw) else []
        out.append([r.benchmark, round(r.share, 4), round(r.performance, 4), round(r.implied_ECI, 3), round(rd, 3), bd] + full)
    return sorted(out, key=lambda r: bench[r[0]]["D"])


def model_data(w, ci, models, fr, gap, bench, lo_col, hi_col):
    data = {}
    for name in models:
        g = w[w.model == name]
        c = ci.loc[name] if name in ci.index else None
        data[name] = {"eci": round(g.ECI.iloc[0], 1), "date": str(g.date.iloc[0].date()),
                      "lo": None if c is None or pd.isna(c[lo_col]) else round(c[lo_col], 1),
                      "hi": None if c is None or pd.isna(c[hi_col]) else round(c[hi_col], 1),
                      "neff": round(1 / (g.share ** 2).sum(), 1), "rows": rows_for(g, bench),
                      "fr": name in set(fr.model), "gap": round(gap.get(name, 0.0), 1)}
    return data


data = model_data(w, ci, models, fr, gap, bench, "eci_ci_low", "eci_ci_high")

# XCI 0.1 (Nov 17, 2026): the same tool under XCI 0.1's fit and likelihood (scripts/xci_weights.py), behind a toggle.
# Its own frontier and near-frontier models, curves, readings, weights and ribbons (each benchmark's Fisher
# information along the ribbon, scaled so the largest peak is 1). Not in the fixed-pair version.
XDATA, XB = {}, {}
_xw, _xc = "eci_benchmark_weights_xci.csv", "benchmark_curves_xci.csv"
if not PAIR and os.path.exists(_xw) and os.path.exists(_xc):
    sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
    from xci_weights import fisher_curve, readings as xci_readings
    xw = pd.read_csv(_xw, parse_dates=["date"], dtype={"bound": str})
    if os.path.exists("eci_full_weights_xci.csv"):  # XCI 0.1's full-refit weights (scripts/xci_full_weights.py): ticks
        xw = xw.merge(pd.read_csv("eci_full_weights_xci.csv")[["model", "benchmark", "full_w", "full_se", "anchor"]],
                      on=["model", "benchmark"], how="left")
    xc = pd.read_csv(_xc).set_index("benchmark")
    xci_ci = pd.read_csv(os.path.join(DATA, "xci_01", "eci.csv")).set_index("Model")
    xm = xw.groupby("model").agg(ECI=("ECI", "first"), date=("date", "first")).reset_index()
    xm = xm[xm.date >= "2023-03-14"].sort_values(["date", "ECI"], ascending=[True, False])
    xfr = xm[xm.ECI > xm.ECI.shift(1).cummax().fillna(-np.inf)]
    xm["gap"] = xm.ECI.cummax() - xm.ECI
    xnear = xm[~xm.model.isin(xfr.model) & (xm.date >= NEAR_FROM) & (xm.gap <= NEAR_GAP)]
    xmodels = list(xfr.model) + list(xnear.model)
    for b in sorted(set(xw[xw.model.isin(xmodels)].benchmark)):
        base = bench.get(b) or {"name": pages.loc[b].display if b in pages.index else b,
                                "short": (pages.loc[b].short if isinstance(pages.loc[b].short, str) else pages.loc[b].display)
                                if b in pages.index else b,
                                "url": f"https://epoch.ai/benchmarks/{pages.loc[b].slug}" if b in pages.index else "https://epoch.ai/benchmarks/search",
                                "rb": round(float(meta.loc[b, "random_baseline"]) if b in meta.index and pd.notna(meta.loc[b, "random_baseline"]) and meta.loc[b, "random_baseline"] >= 0.01 else 0.0, 4),
                                "c": round(float(meta.loc[b, "score_ceiling"]) if b in meta.index and pd.notna(meta.loc[b, "score_ceiling"]) else 1.0, 4)}
        row = xc.loc[b]
        # ticks at the ECI each score implies under XCI 0.1 (ends: 0% and 100% brackets)
        t = [round(float(x), 3) for x in xci_readings(np.array([0.0] + RULER_P[1:-1] + [1.0]), *(np.full(11, row[c])
             for c in ("mid_eci", "alpha_eci", "sigma", "g")), row.delta)]
        XB[b] = {**{k: base[k] for k in ("name", "short", "url", "rb", "c")}, "D": round(row.mid_eci, 2), "a": round(row.alpha_eci, 5),
                 # the weight shape spans the whole line, bracket to bracket (Tim, Nov 18)
                 "k": 0, "rib": fisher_curve(row, np.linspace(t[0], t[-1], RIB_N + 1)), "t": t,
                 "s": round(float(row.sigma), 4), "tau": round(float(row.tau_eci), 3)}
    fmax = max(float(np.max(v["rib"])) for v in XB.values())
    for v in XB.values():
        v["rib"] = [round(float(x) / fmax, 4) for x in v["rib"]]
    XDATA = model_data(xw, xci_ci, xmodels, xfr, dict(zip(xm.model, xm.gap)), XB, "eci_q05", "eci_q95")
def pair_probs(names, draws):
    """P(row model's ECI > column model's ECI) over posterior draws, for every pair of the tool's models (Tim, Nov 18).
    draws: model name -> array of ECI draws (same draws for all models)."""
    names = [n for n in names if n in draws]
    S = np.array([draws[n] for n in names])                         # models x draws
    P = (S[:, None, :] > S[None, :, :]).mean(2)
    return {"names": names, "P": np.round(P, 3).tolist()}


_info = os.path.join(DATA, "ARCHIVE_INFO.json")
DATA_DATE = json.load(open(_info)).get("data_date", "") if os.path.exists(_info) else ""
PROV = json.load(open(_info)).get("provisional", 0) if os.path.exists(_info) else 0  # provisional update: our refit (Nov 20)

# Epoch view: Epoch's own released posterior draws (eci_bootstraps.json, already on the published ECI scale), so the
# probabilities go with the published ECIs; in a provisional update, whose ECIs are all our refit of Epoch's model,
# the refit's draws, for every pair (Tim, Nov 23). XCI view: the XCI fit's draws.
EPH, XPH = {}, {}
_boot = os.path.join(DATA, "epoch_capabilities_index", "eci_bootstraps.json")
_refit = os.path.join(DATA, "bayes", "draws.npz")
if not PAIR and PROV and os.path.exists(_refit):
    _bd = np.load(_refit)
    EPH = pair_probs(list(data), dict(zip(list(_bd["models"]), _bd["eci"].astype(float).T)))
elif not PAIR and os.path.exists(_boot):
    _j = json.load(open(_boot))
    _caps = np.array(_j["model"]["capability_samples"], float)       # draws x models (ECI scale)
    if _j.get("eci_scaled", True):
        EPH = pair_probs(list(data), dict(zip(_j["model"]["names"], _caps.T)))
if XDATA:
    _xd = np.load(os.path.join(DATA, "xci_01", "draws.npz"))
    XPH = pair_probs(list(XDATA), dict(zip(list(_xd["models"]), _xd["eci"].astype(float).T)))
AMAX = float(max(v["a"] for v in bench.values()))
# Peak of the beta-likelihood weight a^2 k^2 p^2 (1-p)^2 [psi'(pk) + psi'((1-p)k)] over the shown score range
# and benchmarks, which scales the ribbons (the least-squares ribbons scale by AMAX instead)
from scipy.special import polygamma
_pg = np.linspace(P_LO, P_HI, 401)
FMAX = max([float(np.max(v["a"]**2 * v["k"]**2 * (_pg * (1 - _pg))**2 * (polygamma(1, _pg * v["k"]) + polygamma(1, (1 - _pg) * v["k"]))))
            for v in bench.values() if v["k"] > 0] or [1.0])
PLOTLY = open(os.path.join(HERE, "vendor", "plotly-basic-2.35.2.min.js"), encoding="utf-8").read()

BETA = "kappa" in bc.columns
note = (("A model's ECI is, to within a few tenths of a point, a weighted average of the ECIs its individual benchmark scores imply (dots). "
         "Weights depend on how steep each benchmark's curve is where the model sits and how widely its scores scatter around the curve; "
         "flat, saturated or widely scattered benchmarks count less. ") if BETA else
        ("A model's ECI is a weighted average of the ECIs its individual benchmark scores imply (dots). "
         "Weights depend on how steep each benchmark's curve is where the model sits; flat or saturated benchmarks count little. ")) + (
        "Both menus list frontier models and near-frontier models released since GPT-5 "
        "(within 8 ECI points of the frontier at release; the number in parentheses is how far behind).")
if SIMPLE:
    note = ("Each row is a benchmark, with both models' scores on it, placed " +
            ("at the ECI each score implies under Epoch's model, " if RULER == "implied" else "where they fall on the benchmark's fitted curve, ") +
            "so that every benchmark is on the same ECI scale."
            + (" Hardest benchmarks (highest ECI at a 50% score) at the top."  if ORDER == "difficulty" else " Benchmarks are grouped by type, with related ones next to each other." if ORDER == "custom" else ""))
html = HEAD.format(title="Compare what each model's ECI rests on", note=note) + """
<style>
button{font-size:14px;padding:2px 10px}
.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
.ctl2{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:2px 0 6px;font-size:13px;color:var(--text-secondary)}
.ex{font-size:13px;color:var(--text-secondary);margin:2px 0 4px}
.ex a{color:inherit;cursor:pointer;text-decoration:underline dotted}
.hdr{font-size:14px;font-weight:500;margin:8px 0 0}.sub{font-size:11.5px;color:var(--text-muted);margin:0 0 2px}
#plot a{fill:inherit}
.foot{font-size:12px;color:var(--text-muted);margin:4px 0}
.ctl button:disabled{opacity:.35;cursor:default}.mth{display:flex;gap:0;margin:6px 0 2px}.mth button{border:1px solid var(--border);background:transparent;color:var(--text-secondary);cursor:pointer}
.mth button:first-child{border-radius:4px 0 0 4px}.mth button:last-child{border-radius:0 4px 4px 0;border-left:0}
.mth button[aria-pressed=true]{background:var(--surface-1);color:var(--text-primary);font-weight:600}
</style>
<div class="mth" id="mth" style="display:none"><button id="mE" aria-pressed="true">Epoch ECI</button><button id="mX" aria-pressed="false" title="XCI v0.1, an extension of Epoch AI&#39;s Bayesian ECI methodology">XCI v0.1</button></div>
<div class="ctl"><button id="prev" aria-label="Previous frontier model">&larr;</button><select id="sel" style="font-size:14px;max-width:70vw"></select><button id="next" aria-label="Next frontier model">&rarr;</button></div>
<div class="ctl2">Compare with <select id="sel2" style="font-size:14px;max-width:60vw"><option value="">(none)</option></select><button id="share" style="font-size:12px">Copy link to this view</button></div>
<div id="plot"></div>
<div class="legend" id="legend"></div>
<div id="hdr"></div>
<p id="info" style="font-size:13px;color:var(--text-secondary);min-height:40px">Hover over or tap a dot for the score behind it; click a benchmark name for Epoch's page on it.</p>
<p class="foot" id="xnote" style="display:none">Under XCI v0.1, a score of 0% or 100% doesn&#39;t point to a single XCI: the higher a model&#39;s capability, the likelier a perfect score, without limit. So instead of a dot, a 100% score gets a bracket, [, at the XCI that a score just short of 100% would imply, meaning &ldquo;at least this&rdquo;; a 0% score gets ], meaning &ldquo;at most this&rdquo;. In the weighted average, such a score is still counted, at a reading beyond the model&#39;s XCI in the score&#39;s direction, marked by an open circle.</p>
<p class="foot">&dagger; Epoch rescales this benchmark's scores before fitting: random guessing counts as 0% and, for some, a maximum below 100% counts as 100%. Scores here are the rescaled ones; hover a dot for the raw score. Data: <a href="https://epoch.ai/benchmarks" target="_blank" style="color:inherit">Epoch AI Benchmarking Hub</a><span id="pub">, __PUB__</span>__ASOF__.</p>
<script>__PLOTLY__</script>
<script>
const EDATA=__D__,EB=__B__,XDATA=__XD__,XB=__XB__,EPH=__EPH__,XPH=__XPH__,DMAX=__DMAX__,AMAX=__AMAX__,FMAX=__FMAX__,PLO=__PLO__,PHI=__PHI__,RIB_N=__RIBN__,PAIR=__PAIR__,SIMPLE=__SIMPLE__,IMPL=__IMPL__,ORDER=__ORDER__,RANK=__RANK__,PROVN=__PROVN__,LABELS=__LABELS__;
/* MX: XCI 0.1 shown (the toggle); DATA and B are the shown method's models and benchmarks */
let MX=false,DATA=EDATA,B=EB,names,FR,NR;const VAL=()=>MX?'XCI':'ECI';
function lists(){names=Object.keys(DATA);FR=names.filter(n=>DATA[n].fr);NR=names.filter(n=>!DATA[n].fr).sort((a,b)=>DATA[a].date<DATA[b].date?-1:DATA[a].date>DATA[b].date?1:DATA[b].eci-DATA[a].eci);}
lists();
let X0=100,X1=Math.ceil((DMAX+5)/10)*10;/* ECI axis range, reset in render() to what the shown models need */
/* first model blue, second red, so the gap column's negative contributions (toward the second) are red (Tim, Nov 17) (both this chart and the clip post's fixed-pair chart) */
const COL=['#5B57D1','#D85A30'],GREEN='#1D9E75',DARKG='#085041';
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim()||'#52514e';
const eAt=(b,p)=>b.D+Math.log(p/(1-p))/b.a, pct=x=>(Math.round(x*1000)/10)+'%', wpc=x=>x<.005?'<1%':Math.round(x*100)+'%';
const adj=b=>b.rb>0||b.c<1;
/* trigamma: recurrence up to 6, then the asymptotic series */
const tg=x=>{let r=0;while(x<6){r+=1/(x*x);x+=1;}const f=1/(x*x);return r+1/x+f/2+f/x*(1/6-f*(1/30-f/42));};
/* weight a benchmark gives a model whose expected score is p, scaled so the largest ribbon peaks at 1:
   beta likelihood (Epoch's model since Nov 2026) when b.k > 0, else least squares */
const wt=(b,p)=>b.k>0?(b.a*b.k*p*(1-p))**2*(tg(p*b.k)+tg((1-p)*b.k))/FMAX:(b.a/AMAX)**2*(4*p*(1-p))**2;
/* where a score's dot sits: the ECI it implies (beta: where the score is most likely), else the curve inverse */
const dotE=(b,v)=>SIMPLE&&IMPL?v[3]:SIMPLE?eAt(b,Math.min(.999,Math.max(.001,v[2]))):(b.k>0||b.rib)?v[3]:eAt(b,Math.min(.995,Math.max(.005,v[2])));
/* simple mode shows scores, not the estimator: each dot sits where its score falls on the benchmark's fitted curve
   (scores clipped to Epoch's 0.1%-99.9%, so a 0% or 100% sits at the end of the line). With IMPL the scale is redrawn
   instead: ticks and dots sit at the ECI each score implies (b.t: 0.1%, 10%, ..., 90%, 99.9%) */
const tickE=(b,q)=>(IMPL||(!SIMPLE&&b.t))?b.t[q]:eAt(b,q===0?.001:q===10?.999:q/10);/* q = 0..10; the full chart uses b.t when built */
const sel=document.getElementById('sel'),sel2=document.getElementById('sel2'),info=document.getElementById('info');
const lab=(n,d)=>`${n} (${d?DATA[n].date+', ':''}${VAL()} ${DATA[n].eci.toFixed(1)}${DATA[n].fr?'':', \\u2212'+DATA[n].gap.toFixed(1)})`;
function fill(s,d){[['Frontier',FR],['Near-frontier at release',NR]].forEach(([t,ns])=>{if(!ns.length)return;const og=document.createElement('optgroup');og.label=t;
 ns.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=lab(n,d);og.appendChild(o);});s.appendChild(og);});}
function fillAll(){sel.innerHTML='';sel2.innerHTML='<option value="">(none)</option>';fill(sel,true);fill(sel2,false);}
fillAll();
/* Epoch view of the full chart: name Epoch's 0.1%-99.9% clip next to a score it moves (Tim, Nov 18); XCI 0.1 doesn't clip */
const clipNote=p=>SIMPLE||MX?'':p>0.999+1e-9?' (clipped to 99.9%)':p<0.001-1e-9?' (clipped to 0.1%)':'';
function scoreText(b,p){if(!adj(b))return `scored ${pct(p)}${clipNote(p)}`;
 const raw=b.rb+p*(b.c-b.rb);let s=`Epoch-adjusted score ${pct(p)}${clipNote(p)} (raw ${p<=0.0011?'at or below '+pct(b.rb):p>=0.9989&&b.c<1?'at or above '+pct(b.c):pct(raw)}`;
 const parts=[];if(b.rb>0)parts.push(`random guessing ${pct(b.rb)} counts as 0%`);if(b.c<1)parts.push(`${pct(b.c)} counts as 100%`);return s+'; '+parts.join(', ')+')';}
function dotText(n,bn,v){const b=B[bn],p=v[2],e=dotE(b,v);
 if(SIMPLE)return `<b>${n}</b> \\u00b7 ${b.name}<br>${scoreText(b,p)}<br>${IMPL?'implies ECI':'on the benchmark\\u2019s curve at ECI'} ${e.toFixed(1)}${p<=0.0011||p>=0.9989?(IMPL?' (as clipped by Epoch)':' (Epoch\\u2019s clip)'):''}`;
 if(v[5])return `<b>${n}</b> \\u00b7 ${b.name}<br>${scoreText(b,p)}<br>implied ${VAL()} ${v[5]==='1'?'at least':'at most'} ${e.toFixed(1)}; counted as ${v[4].toFixed(1)}`;
 return `<b>${n}</b> \\u00b7 ${b.name}<br>${scoreText(b,p)}<br>implied ${VAL()} ${p<=0.0011?'below '+e.toFixed(0):p>=0.9989?'above '+e.toFixed(0):e.toFixed(1)}`;}
/* posterior probability that the first model's ECI is higher than the second's (PH: the shown method's pairwise table) */
function pHigher(a,b){const T=MX?XPH:EPH;if(!T||!T.names)return '';const i=T.names.indexOf(a),j=T.names.indexOf(b);if(i<0||j<0)return '';
 const p=T.P[i][j];return ` (P(first model higher) ${p>=.995?'> 99%':p<=.005?'< 1%':'= '+Math.round(p*100)+'%'})`;}
function header(shown){const two=shown.length>1;document.getElementById('hdr').innerHTML=shown.map((n,k)=>{const m=DATA[n],ciT=m.lo?` (90% CI ${m.lo.toFixed(1)}\\u2013${m.hi.toFixed(1)})`:'';
 return `<p class="hdr" style="color:${two?COL[k]:'var(--text-primary)'}">${n} \\u00b7 ${VAL()} ${m.eci.toFixed(1)}${ciT}</p><p class="sub">released ${m.date} \\u00b7 ${m.rows.length} benchmarks \\u00b7 effective number ${m.neff}${m.fr?'':' \\u00b7 '+m.gap.toFixed(1)+' below the frontier at release'}</p>`;}).join('')+(two&&!SIMPLE?`<p class="sub" style="margin-top:4px">${VAL()} gap ${(DATA[shown[0]].eci-DATA[shown[1]].eci>=0?'+':'')+(DATA[shown[0]].eci-DATA[shown[1]].eci).toFixed(1)}, shown split by benchmark in rightmost column.${pHigher(shown[0],shown[1])}</p>`:'');
 const sw=(c,l)=>`<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${c}"/></svg><svg width="10" height="12"><line x1="5" y1="0" x2="5" y2="12" stroke="${c}" stroke-width="1.5" stroke-dasharray="3 2"/></svg><span style="width:12px;height:7px;border-radius:2px;background:${c}"></span>${l}</span>`;
 if(SIMPLE){const sw2=(c,l)=>`<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="${c}"/></svg><svg width="10" height="12"><line x1="5" y1="0" x2="5" y2="12" stroke="${c}" stroke-width="1.5" stroke-dasharray="3 2"/></svg>${l}</span>`;
  document.getElementById('legend').innerHTML=`<span><svg width="30" height="12"><line x1="1" y1="6" x2="29" y2="6" stroke="${GREEN}"/><line x1="8" y1="3" x2="8" y2="9" stroke="${DARKG}"/><line x1="15" y1="1" x2="15" y2="11" stroke="${DARKG}"/><line x1="22" y1="3" x2="22" y2="9" stroke="${DARKG}"/></svg>Benchmark scale${IMPL?': the ECI each score implies,':''} from 0.1% to 99.9% (Epoch\\u2019s clip), with ticks at 10%, 20%, \\u2026, 90% (long tick: 50%)</span>`+shown.map((n,k)=>sw2(COL[k],n+': score on the scale (dot), ECI (dashed)')).join('');return;}
 document.getElementById('legend').innerHTML=`<span><svg width="26" height="12"><path d="M1 6 Q13 -2 25 6 Q13 14 1 6Z" fill="${GREEN}"/></svg>Benchmark weight curve</span>`+shown.map((n,k)=>sw(COL[k],two?n:`implied ${VAL()} (dot) \\u00b7 ${VAL()} (dashed) \\u00b7 weight share (bar)`)).join('');}
function render(shown){const ms=shown.map(n=>DATA[n]),two=ms.length>1;
 {/* ECI axis: just the span of the shown models' dots and ECIs, within 30 points of their ECIs; padded, on a 5-point grid */
  /* the union of the ranges both methods need (Epoch ECI and XCI 0.1), so the axis doesn't change when toggling (Tim, Nov 18) */
  let lo=Infinity,hi=-Infinity;
  [[EDATA,EB],[XDATA,XB]].forEach(([D,BB])=>{if(!D)return;const mm=shown.map(n=>D[n]).filter(Boolean);if(!mm.length)return;
   const mid=mm.reduce((a,m)=>a+m.eci,0)/mm.length,v=[...mm.map(m=>m.eci),...mm.flatMap(m=>m.rows.map(r=>dotE(BB[r[0]],r))),...mm.flatMap(m=>m.rows.filter(r=>r[5]).map(r=>r[4]))];/* XCI 0.1: the open circles too (Tim, Nov 19) */
   lo=Math.min(lo,Math.max(Math.min(...v),mid-30));hi=Math.max(hi,Math.min(Math.max(...v),mid+30));});
  X0=Math.floor((lo-2)/5)*5;X1=Math.ceil((hi+2)/5)*5;}
 const byB={};ms.forEach((m,k)=>m.rows.forEach(r=>{(byB[r[0]]=byB[r[0]]||[null,null])[k]=r;}));
 const mE=bn=>{const v=byB[bn].filter(Boolean).map(r=>dotE(B[bn],r));return v.reduce((s,x)=>s+x,0)/v.length;};
 const DD=bn=>(EB[bn]||B[bn]).D;/* difficulty order: Epoch's D_b in both views, so rows stay put when toggling to XCI 0.1 (Tim, Nov 18) */
 const rows=Object.keys(byB).sort(ORDER==='custom'?(a,b)=>(RANK[b]??-1)-(RANK[a]??-1)||DD(a)-DD(b):ORDER==='difficulty'?(a,b)=>DD(a)-DD(b):(a,b)=>mE(a)-mE(b));/* custom: RANK 0 at the top *//* y counts up, so the hardest benchmark (difficulty order) or the one whose scores imply the highest ECI (averaged over the models shown) ends up at the top */
 const n=rows.length,W=document.getElementById('plot').clientWidth,narrow=W<700,y=i=>i,txt=css('--text-secondary'),mut=css('--text-muted'),grid=css('--border');
 const rx=[],ry=[],lx=[],ly=[],tx=[],ty=[],tw=[],hx=[],hy=[],ht=[];
 rows.forEach((bn,i)=>{const b=B[bn],e1=!SIMPLE&&b.t?b.t[0]:eAt(b,PLO),e9=!SIMPLE&&b.t?b.t[10]:eAt(b,PHI),top=[],bot=[],N=b.rib?b.rib.length-1:RIB_N;
  for(let k=0;k<=N;k++){const e=e1+(e9-e1)*k/N,p=1/(1+Math.exp(-b.a*(e-b.D))),h=0.36*(b.rib?b.rib[k]:wt(b,p));top.push([e,i+h]);bot.unshift([e,i-h]);}
  [...top,...bot,top[0]].forEach(([a,c])=>{rx.push(a);ry.push(c);});rx.push(null);ry.push(null);
  if(SIMPLE){const c0=tickE(b,0),c9=tickE(b,10);lx.push(c0,c9,null);ly.push(i,i,null);}/* simple: the scale runs to Epoch's clip, 0.1%-99.9%, with no end caps (Tim's choice) */
  else{const l0=b.t?tickE(b,0):e1,l1=b.t?tickE(b,10):e9;lx.push(l0,l1,null);ly.push(i,i,null);}/* the line runs to where a 0.1% and a 99.9% score imply (XCI: 0% and 100%; Tim, Nov 18); the weight shape spans the same range; the line shows where it gets too thin to see */
  /* XCI 0.1: also ticks at 0% and 100% (the line's ends), which it scores as such; Epoch's line ends at its 0.1% / 99.9% clip, unticked (Tim, Nov 19) */
  for(let q=MX&&!SIMPLE&&b.t?0:1;q<=(MX&&!SIMPLE&&b.t?10:9);q++){const e=(SIMPLE||b.t)?tickE(b,q):eAt(b,q/10),t=q===5?.22:.11;tx.push(e,e,null);ty.push(i-t,i+t,null);}
  hx.push(b.D);hy.push(i);ht.push(`<b>${b.name}</b>${adj(b)?' \\u2020':''}<br>`+(SIMPLE&&IMPL?`a 50% score implies ECI ${b.D.toFixed(0)}; 10% and 90% imply ${b.t[1].toFixed(0)} and ${b.t[9].toFixed(0)}`:`half solved at ${VAL()} ${b.D.toFixed(0)}; 10%\\u219290% from ${eAt(b,.1).toFixed(0)} to ${eAt(b,.9).toFixed(0)}`));});
 const tr=[{x:rx,y:ry,type:'scatter',mode:'lines',fill:'toself',fillcolor:GREEN,opacity:.85,line:{width:0},hoverinfo:'skip',visible:!SIMPLE},
  {x:lx,y:ly,type:'scatter',mode:'lines',line:{color:GREEN,width:1},opacity:SIMPLE?.6:.85,hoverinfo:'skip'},/* full chart: the shape's opacity, so line and shape read as one */
  {x:tx,y:ty,type:'scatter',mode:'lines',line:{color:DARKG,width:1},hoverinfo:'skip'},
  {x:hx,y:hy,type:'scatter',mode:'markers',marker:{size:14,opacity:0},text:ht,hovertemplate:'%{text}<extra></extra>'}];
 const off=two?[.14,-.14]:[0];
 const wmax=Math.max(...ms.flatMap(m=>m.rows.map(r=>Math.max(r[1],r.length>6?r[6]:0))));
 ms.forEach((m,k)=>{const X=[],Y=[],S=[],T=[],BX=[],BY=[],BT=[],BL=[],FX=[],FY=[],K={'0':{x:[],y:[],t:[]},'1':{x:[],y:[],t:[]}},OX=[],OY=[],OT=[];
  rows.forEach((bn,i)=>{const v=byB[bn][k];if(!v)return;const b=B[bn],e=dotE(b,v),t=dotText(shown[k],bn,v),inR=e>=X0&&e<=X1,xe=Math.min(X1-.8,Math.max(X0+.8,e));
   if(v[5]&&inR&&!SIMPLE){const q=K[v[5]];q.x.push(e);q.y.push(i+off[k]);q.t.push(t);if(v[4]>=X0&&v[4]<=X1){OX.push(v[4]);OY.push(i+off[k]);OT.push(t);}}/* XCI 0.1's 0% / 100%: a bracket at the end of the ECIs the score is consistent with */
   else{X.push(xe);Y.push(i+off[k]);S.push(e<X0?'triangle-left':e>X1?'triangle-right':'circle');T.push(t);}
   const hasF=v.length>6,bt=`<b>${shown[k]}</b> \\u00b7 ${b.name}<br>Weight: ${wpc(v[1])}`+(hasF?`<br>Full refit impact: ${Math.round(v[6]*100)}% (\\u00b1 ${Math.max(1,Math.round(2*v[7]*100))}%) of `+(v[5]?`a ${v[5]==='1'?'fall':'rise'} from circle to 1 ${VAL()} pt. ${v[5]==='1'?'below':'above'} bracket`:`this score\\u2019s implied ${VAL()} change`)+(v[8]?` (relative to models near its fixed ${VAL()} of ${Math.round(DATA[shown[k]].eci)})`:''):'');
   BX.push(v[1]);BY.push(i+off[k]);BT.push(bt);BL.push(wpc(v[1]));if(hasF){const y0=i+off[k],h=two?.13:.2,e=.1,lo=two?(k===0?y0-h:y0-h-e):y0-h-e,hi=two?(k===0?y0+h+e:y0+h):y0+h+e;const fx=Math.max(0,v[6]);FX.push(fx,fx,null);/* a negative refit weight (within noise of 0) gets its tick at 0; its hover text keeps the value (Tim, Nov 19) */FY.push(lo,hi,null);}});
  tr.push({x:X,y:Y,type:'scatter',mode:'markers',marker:{color:COL[k],size:9,symbol:S,line:{color:css('--surface-2')||'#fff',width:1.2}},text:T,hovertemplate:'%{text}<extra></extra>',name:shown[k]});
  /* the bracket's stroke sits on the line's end: Plotly sets text ~4px off its point, so shift it back by that much (Tim, Nov 19) */
  const u1=(X1-X0)/((W-(narrow?118:210))*(two?(narrow?.5:.6):(narrow?.72:.8)));
  [['1','['],['0',']']].forEach(([bd,g])=>{const q=K[bd];if(q.x.length)tr.push({x:q.x.map(x=>x+(bd==='1'?-1:1)*5.5*u1),y:q.y,type:'scatter',mode:'markers+text',marker:{size:2,opacity:0},
   text:q.x.map(()=>'<b>'+g+'</b>'),textposition:bd==='1'?'middle right':'middle left',textfont:{color:COL[k],size:14,family:'system-ui,sans-serif'},hovertext:q.t,hovertemplate:'%{hovertext}<extra></extra>',hoverlabel:{bgcolor:COL[k]},name:shown[k]});});
  /* XCI 0.1's 0% / 100%: an open circle where the score is counted in the weighted average (Tim, Nov 19) */
  if(OX.length)tr.push({x:OX,y:OY,type:'scatter',mode:'markers',marker:{symbol:'circle-open',size:7,color:COL[k],line:{width:1.5,color:COL[k]}},hovertext:OT,hovertemplate:'%{hovertext}<extra></extra>',hoverlabel:{bgcolor:COL[k]},name:shown[k]});
  if(!SIMPLE){tr.push({x:BX,y:BY,xaxis:'x2',type:'bar',orientation:'h',width:two?.26:.4,marker:{color:COL[k]},cliponaxis:false,hovertext:BT,hovertemplate:'%{hovertext}<extra></extra>',name:shown[k]});
   /* full-refit weight: a faint tick on the bar (Tim, Nov 19), drawn under the bar's % label, which stays at the bar's end */
   /* two models: each tick sticks out only on its own bar's outer side, so the two never overlap (Tim, Nov 19) */
   if(FX.length)tr.push({x:FX,y:FY,xaxis:'x2',type:'scatter',mode:'lines',line:{width:2.5,color:txt},opacity:.3,cliponaxis:false,hoverinfo:'skip',name:shown[k]});
   /* the number's hover point sits at the middle of the number (not its left edge), so it doesn't take the hover from the other model's bar (Tim, Nov 19) */
   const d2=two?(narrow?[.53,.72]:[.62,.77]):(narrow?[.76,.96]:[.82,1]),u=wmax*1.6/((W-(narrow?118:210))*(d2[1]-d2[0]));
   tr.push({x:BX.map((x,j)=>x+u*(3+2.4*BL[j].length)),y:BY.map(y=>y+(two?(k===0?.05:-.05):0)),xaxis:'x2',type:'scatter',mode:'text',text:BL,textposition:'middle center',textfont:{size:8,color:txt},cliponaxis:false,hovertext:BT,hovertemplate:'%{hovertext}<extra></extra>',hoverlabel:{bgcolor:COL[k]},name:shown[k]});}
  if(SIMPLE){const SX=[],SY=[],ST=[],SH=[];rows.forEach((bn,i)=>{const v=byB[bn][k];if(!v)return;const o=byB[bn][1-k],hiK=o&&v[2]>o[2];
    SX.push(k);SY.push(i);ST.push((hiK?'<b>':'')+pct(v[2])+(hiK?'</b>':''));SH.push(dotText(shown[k],bn,v));});
   tr.push({x:SX,y:SY,xaxis:'x4',type:'scatter',mode:'text',text:ST,textfont:{color:COL[k],size:narrow?11:12.5},hovertext:SH,hovertemplate:'%{hovertext}<extra></extra>'});}});
 let cmax=1;
 if(two&&!SIMPLE){const avg=(ms[0].eci+ms[1].eci)/2,CX=[],CY=[],CC=[],CT=[],CL=[];
  rows.forEach((bn,i)=>{const v=byB[bn],b=B[bn],part=k=>v[k]?v[k][1]*(v[k][4]-avg):0,c=part(0)-part(1);/* v[4]: the reading the weighted average uses */
   const line=k=>v[k]?`${shown[k]}: ${wpc(v[k][1])} weight \u00d7 (${v[k][5]?'counted as':'implied'} ${v[k][4].toFixed(1)} \u2212 midpoint ${avg.toFixed(1)}) = ${(part(k)>=0?'+':'')+part(k).toFixed(2)}`:`${shown[k]}: not scored`;
   CX.push(c);CY.push(i);CC.push(COL[c>=0?0:1]);CL.push(Math.abs(c)<.05?'':(c>0?'+':'')+c.toFixed(1));
   CT.push(`<b>${b.name}</b>: ${(c>=0?'+':'')+c.toFixed(2)} ${VAL()} toward ${shown[c>=0?0:1]}<br>${line(0)}<br>${line(1)}`);});
  cmax=Math.max(.5,...CX.map(Math.abs));
  tr.push({x:CX,y:CY,xaxis:'x3',type:'bar',orientation:'h',width:.5,marker:{color:CC},text:CL,textposition:'outside',textfont:{size:11,color:txt},cliponaxis:false,hovertext:CT,hovertemplate:'%{hovertext}<extra></extra>'});}
 const shapes=ms.map((m,k)=>({type:'line',xref:'x',yref:'y',x0:m.eci,x1:m.eci,y0:-.6,y1:n-.4,line:{color:COL[k],width:1.5,dash:'dot'}}));
 const tick=rows.map(bn=>{const b=B[bn];return `<a href="${b.url}" target="_blank">${narrow?b.short:b.name}${adj(b)?' \\u2020':''}</a>`;});
 const lay={height:n*24+70+(SIMPLE?16:0),margin:{l:narrow?96:200,r:narrow?22:10,t:SIMPLE?26:10,b:40},showlegend:false,hovermode:'closest',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:'system-ui,sans-serif',size:narrow?10.5:11.5,color:txt},shapes,dragmode:false,
  xaxis:{fixedrange:true,domain:SIMPLE?(narrow?[.38,1]:[.25,1]):two?(narrow?[0,.5]:[0,.6]):(narrow?[0,.72]:[0,.8]),range:[X0,X1],dtick:X1-X0<=30?5:10,gridcolor:grid,zeroline:false,title:{text:VAL(),font:{size:11,color:mut}},tickfont:{color:mut}},
  xaxis2:{visible:!SIMPLE,domain:two?(narrow?[.53,.72]:[.62,.77]):(narrow?[.76,.96]:[.82,1]),range:[0,wmax*1.6],showgrid:false,showticklabels:false,zeroline:false,fixedrange:true,title:{text:'weight',font:{size:11,color:mut}}},
  xaxis4:{domain:narrow?[0,.36]:[0,.22],range:[-.65,1.65],visible:false,fixedrange:true},
  annotations:SIMPLE?shown.map((nm,k)=>({xref:'x4',yref:'paper',x:k,y:1,yanchor:'bottom',showarrow:false,text:'<b>'+(LABELS[k]||nm)+'</b>',font:{color:COL[k],size:narrow?10.5:11.5}})):[],
  xaxis3:{domain:narrow?[.78,.97]:[.81,1],range:[-cmax*1.5,cmax*1.5],visible:two&&!SIMPLE,showgrid:false,showticklabels:false,zeroline:true,zerolinecolor:mut,fixedrange:true,title:{text:'share of gap',font:{size:11,color:mut}}},
  yaxis:{fixedrange:true,tickvals:rows.map((_,i)=>i),ticktext:tick,range:[-.7,n-.3],showgrid:false,zeroline:false,fixedrange:true,ticks:'',...(SIMPLE?{anchor:'free',position:0}:{})},
  hoverlabel:{align:'left',font:{size:12}}};
 Plotly.newPlot('plot',tr,lay,{displaylogo:false,responsive:true,displayModeBar:false});
 /* hovering a benchmark name shows its parameters (Tim, Nov 17): a native tooltip on each y-axis label */
 document.querySelectorAll('#plot .ytick text').forEach((el,i)=>{const b=B[rows[i]];if(!b)return;
  const p=[`D = ${b.D.toFixed(1)}`,`\\u03b1 = ${b.a.toFixed(3)}`];if(b.s!=null)p.push(`\\u03c3 = ${b.s.toFixed(3)}`);if(b.tau!=null)p.push(`\\u03c4 = ${b.tau.toFixed(2)}`);
  const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=b.name+'\\n'+p.join(', ');el.insertBefore(t,el.firstChild);});
 const pl=document.getElementById('plot');pl.removeAllListeners&&pl.removeAllListeners('plotly_click');
 pl.on('plotly_click',ev=>{const p=ev.points[0];const t=p.hovertext||p.text;if(t)info.innerHTML=t;});}
/* disable an arrow with no frontier model to step to: next at the latest frontier model, prev at the first (Tim, Nov 18) */
function arrows(){const c=sel.value,i=FR.indexOf(c),dt=DATA[c]&&DATA[c].date;
 document.getElementById('prev').disabled=i>=0?i===0:!FR.some(n=>DATA[n].date<=dt);
 document.getElementById('next').disabled=i>=0?i===FR.length-1:!FR.some(n=>DATA[n].date>dt);}
function renderAll(){const shown=[sel.value,...(sel2.value&&sel2.value!==sel.value?[sel2.value]:[])];header(shown);render(shown);arrows();
 info.textContent='Hover over or tap a dot for the score behind it; click a benchmark name for Epoch\\u2019s page on it.';}
sel.addEventListener('change',renderAll);sel2.addEventListener('change',renderAll);
const step=d=>{const c=sel.value,i=FR.indexOf(c);let j;
 if(i>=0)j=Math.min(FR.length-1,Math.max(0,i+d));
 else{const dt=DATA[c].date;j=d>0?FR.findIndex(n=>DATA[n].date>dt):FR.length-1-[...FR].reverse().findIndex(n=>DATA[n].date<=dt);if(j<0||j>=FR.length)j=d>0?FR.length-1:0;}
 sel.value=FR[j];sel2.value=j>0?FR[j-1]:'';renderAll();};/* arrows also compare with the previous frontier model (Tim, Nov 18) */
document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);
let q=new URLSearchParams(location.search);try{const ps=new URLSearchParams(parent.location.search);if(ps.get('m'))q=ps;}catch(e){}
if(SIMPLE||PAIR.length){document.querySelector('.ctl').style.display=SIMPLE?'none':'';document.querySelector('.ctl2').style.display=SIMPLE?'none':'';q=new URLSearchParams({m:PAIR[0],c:PAIR[1]});}
/* the Epoch / XCI 0.1 toggle: switch the data, rebuild the menus, keep the chosen models where the other method shows them */
function choose(m,c){if(m&&DATA[m]){sel.value=m;sel2.value=c&&DATA[c]?c:'';}
 else{sel.value=FR[FR.length-1];sel2.value=FR.length>1?FR[FR.length-2]:'';}}/* default: current frontier vs the previous one */
function setMethod(x){MX=x&&Object.keys(XDATA).length>0;DATA=MX?XDATA:EDATA;B=MX?XB:EB;lists();fillAll();
 document.getElementById('mE').setAttribute('aria-pressed',String(!MX));document.getElementById('mX').setAttribute('aria-pressed',String(MX));
 document.getElementById('pub').style.display=MX?'none':'';document.getElementById('xnote').style.display=MX?'':'none';}
if(Object.keys(XDATA).length&&!SIMPLE){document.getElementById('mth').style.display='';
 document.getElementById('mE').onclick=()=>{if(!MX)return;const m=sel.value,c=sel2.value;setMethod(false);choose(m,c);renderAll();};
 document.getElementById('mX').onclick=()=>{if(MX)return;const m=sel.value,c=sel2.value;setMethod(true);choose(m,c);renderAll();};}
if(q.get('x')==='1')setMethod(true);
choose(q.get('m'),q.get('c'));
renderAll();
document.getElementById('share').onclick=()=>{const st='m='+encodeURIComponent(sel.value)+(sel2.value?'&c='+encodeURIComponent(sel2.value):'')+(MX?'&x=1':'');let url=null;
 try{const pl=parent.location;if(/^https?:/.test(pl.protocol))url=pl.origin+pl.pathname+'?'+st+'#fig-panels';}catch(e){}
 if(!url&&/^https?:/.test(location.protocol))url=location.origin+location.pathname+'?'+st;
 if(!url){info.textContent='Links work once the post is online.';return;}
 (navigator.clipboard?navigator.clipboard.writeText(url):Promise.reject()).then(()=>{info.textContent='Link copied: '+url;},()=>{info.textContent='Copy this link: '+url;});};
let lastNarrow=null,rt=null;new ResizeObserver(()=>{clearTimeout(rt);rt=setTimeout(()=>{const nw=document.getElementById('plot').clientWidth<700;if(lastNarrow!==null&&nw!==lastNarrow)renderAll();lastNarrow=nw;},150);}).observe(document.getElementById('plot').parentElement);
window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderAll);

// tell an embedding page how tall this chart is (works across origins)
if(window.parent!==window){document.documentElement.style.cssText='color-scheme:light;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;--surface-1:#f3f2ee;--surface-2:#fff;--border:rgba(11,11,11,.10)';document.body.style.background='transparent';document.querySelectorAll('body>h1,body>h1+p').forEach(e=>e.style.display='none');document.body.style.margin='0 auto';renderAll();const post=()=>parent.postMessage({eciFrameHeight:Math.ceil(document.body.getBoundingClientRect().height)+24},'*');new ResizeObserver(post).observe(document.body);window.addEventListener('load',post);post();}
</script></body></html>"""
# ECI_PANEL_ORDER=custom (Tim, Nov 18, 2026): rows in the hand-edited order of viz/benchmark_order.csv (groups of related
# benchmarks, hardest first within a group), the same in both views. A benchmark not listed there goes right after the
# listed one whose implied ECIs (XCI 0.1 if available, else Epoch's; 0%/100% scores left out) correlate with it most
# across models (at least 8 in common), else at the end; a WARNING names it so it can be placed by hand.
RANK = {}
if ORDER == "custom":
    _o = list(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_order.csv")).benchmark)
    _src = xw if "xw" in globals() and xw is not None else w
    _pts = (_src[_src.bound.isna()] if "bound" in _src else _src).pivot_table(index="model", columns="benchmark", values="implied_ECI")
    for _b in sorted(set(w.benchmark) | (set(xw.benchmark) if "xw" in globals() and xw is not None else set())):
        if _b in _o: continue
        _best, _r = None, -2
        for _c in _o:
            if _b in _pts and _c in _pts:
                _x = _pts[[_b, _c]].dropna()
                if len(_x) >= 8 and _x[_b].corr(_x[_c]) > _r: _best, _r = _c, _x[_b].corr(_x[_c])
        _o.insert(_o.index(_best) + 1, _b) if _best else _o.append(_b)
        print(f"WARNING: benchmark {_b} is not in viz/benchmark_order.csv; placed after {_best or 'the last one'}", file=sys.stderr)
    RANK = {b: i for i, b in enumerate(_o)}
for k, v in {"__D__": json.dumps(data), "__B__": json.dumps(bench), "__XD__": json.dumps(XDATA), "__XB__": json.dumps(XB), "__EPH__": json.dumps(EPH), "__XPH__": json.dumps(XPH), "__DMAX__": str(DMAX), "__AMAX__": str(AMAX), "__FMAX__": repr(FMAX),
             "__PAIR__": json.dumps(PAIR), "__LABELS__": json.dumps(LABELS), "__SIMPLE__": "true" if SIMPLE else "false", "__IMPL__": "true" if RULER == "implied" else "false", "__ORDER__": json.dumps(ORDER), "__RANK__": json.dumps(RANK),
             "__PLO__": str(P_LO), "__PROVN__": str(int(PROV)), "__PUB__": f"provisional ECI, incl. {PROV} scores Epoch hasn&#39;t folded in yet" if PROV else "ECI as published", "__ASOF__": f" (data as of {pd.Timestamp(DATA_DATE):%b %-d, %Y})" if DATA_DATE else "", "__PHI__": str(P_HI), "__RIBN__": str(RIB_N)}.items():
    html = html.replace(k, v)
html = html.replace("<script>__PLOTLY__</script>", "<script>" + PLOTLY.replace("</script", "<\\/script") + "</script>")
open(OUT, "w", encoding="utf-8").write(html)
if PAIR:  # the curves behind the chart, for a post that computes from them
    pd.DataFrame([{"benchmark": b, "display": v["name"], "mid_eci": bc.loc[b, "mid_eci"], "alpha_eci": bc.loc[b, "alpha_eci"],
                   "kappa": bc.loc[b, "kappa"] if "kappa" in bc.columns else np.nan} for b, v in bench.items()]
                 ).to_csv(os.path.splitext(OUT)[0] + "_curves.csv", index=False)
    # and the two models' scores, each with Epoch's published ECI and the ECI the score implies (<output name>_scores.csv)
    sc = w[w.model.isin(PAIR)][["model", "ECI", "benchmark", "performance", "implied_ECI"]].copy()
    sc.insert(3, "display", sc.benchmark.map(lambda b: bench[b]["name"])); sc.insert(4, "short", sc.benchmark.map(lambda b: bench[b]["short"]))
    sc.to_csv(os.path.splitext(OUT)[0] + "_scores.csv", index=False)
print(f"Wrote {OUT}: " + (f"{PAIR[0]} vs {PAIR[1]}" + (" (simple)" if SIMPLE else "") if PAIR else f"{len(fr)} frontier + {len(models) - len(fr)} near-frontier models"))
