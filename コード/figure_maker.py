#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）図版生成スクリプト
出典: boundary_sweep_result.csv / highdim_metric_result.csv / axis_phase_result.csv
論文ノート: 論文（４）数値実験の総括 / 実験4-8ノート
図の言語: 英語（論文原稿は英語）; 保存先: 研究/AI/図/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "図"
FIG_DIR.mkdir(exist_ok=True)

# ---------- 共通スタイル ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------- データ読込 ----------
bd = {}
with open(HERE / "boundary_sweep_result.csv") as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        d = dict(zip(header, parts))
        if d["seed"] != "mean":
            continue
        key = (d["axis"], float(d["param"]))
        bd[key] = dict(lam=float(d["lam"]), det=float(d["det"]), kappa=float(d["kappa"]),
                       sigma=float(d["sigma"]), I=float(d["I"]))

hd = {}
with open(HERE / "highdim_metric_result.csv") as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        d = dict(zip(header, parts))
        key = (float(d["gamma"]), d["seed"], int(d["d"]))
        hd[key] = dict(lam=float(d["lam"]), det=float(d["det"]), kappa=float(d["kappa"]),
                       sigma1=float(d["sigma1"]), gauss=float(d["gauss_est"]),
                       gauss_lam=float(d["gauss_lam"]))

pd_ = {}
with open(HERE / "axis_phase_result.csv") as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        d = dict(zip(header, parts))
        key = (d["axis"], float(d["gamma"]), float(d["c_mem"]), float(d["D"]), d["seed"])
        pd_[key] = dict(lam=float(d["lam_min"]), det=float(d["det"]), kappa=float(d["kappa"]),
                        g_tt=float(d["g_tt"]), g_tt_G=float(d["g_tt_G"]),
                        R1=float(d["R1"]), omega=float(d["omega"]))

# =====================================================================
# Fig 1. Phase-1 spectral screening: ROI vs gamma (実験2・スキャン2要約)
# =====================================================================
fig, ax = plt.subplots(figsize=(4.2, 2.8))
gammas = [0.1, 0.2, 0.5, 1.0, 2.0]
lo = [0.400, 0.100, 0.009, 0.0005, 0.0001]
hi = [1.578, 0.428, 0.041, 0.001, 0.001]
roi = [True, True, False, False, False]
ax.axvspan(-0.02, 0.22, color="C0", alpha=0.10, label="ROI ($\\gamma \\lesssim 0.2$)")
ax.errorbar(gammas, [(a+b)/2 for a, b in zip(lo, hi)],
            yerr=[[(b-a)/2 for a, b in zip(lo, hi)]], fmt="o", ms=4,
            color="C3" if False else "C1", ecolor="C1", capsize=3, label="max |I| range")
ax.axhline(0.0, color="k", lw=0.6)
ax.axhline(0.028, color="gray", ls="--", lw=0.8, label="null 99% threshold")
ax.set_xscale("log")
ax.set_xticks(gammas)
ax.set_xticklabels([f"{g:g}" for g in gammas])
ax.set_xlabel("memory decay $\\gamma$")
ax.set_ylabel("band-integrated imag. spectrum  max$|I|$")
ax.set_title("Phase 1 screening: ROI = weak decay")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig1_Phase1_screening_ROI.png")
plt.close(fig)

# =====================================================================
# Fig 2. Metric comparison across 4 conditions (実験4)
# =====================================================================
conds = ["ROI\n($\\gamma=0.1$, $J_{asym}$)", "EQ\n($J_{asym}=0$)", "DD\n($\\gamma=2.0$)", "Gauss\n(matched)"]
lam = [0.2075, 2.51e6, 1.30e8, 0.1415]
lam_se = [0.0142, None, None, 0.0015]
kap = [1.28, 1.51, 1.45, 1.88]
detv = [0.0565, 9.03e12, 2.39e16, 0.0376]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
bars = ax.bar(range(4), lam, color=["#4C72B0", "#DD8452", "#C44E52", "#55A868"], width=0.6)
for i, (m, s) in enumerate(zip(lam, lam_se)):
    if s:
        ax.errorbar(i, m, yerr=s, fmt="none", ecolor="k", capsize=3)
ax.set_yscale("log")
ax.set_xticks(range(4)); ax.set_xticklabels(conds)
ax.set_ylabel("$\\lambda_{\\min}$ (KDE-Fisher)")
ax.set_title("Analyze 4-conditions (exp.4)")
for i, v in enumerate(lam):
    ax.annotate(f"{v:.4g}", (i, v*1.3), ha="center", fontsize=7)

ax = axes[1]
ax.plot(range(4), kap, "o-", color="#4C72B0", ms=4)
ax.set_xticks(range(4)); ax.set_xticklabels(conds)
ax.set_ylabel("condition number $\\kappa$")
ax.set_title("No $\\kappa\\to\\infty$ in ROI")
ax.set_ylim(0, 3)
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig2_FourConditions_metric.png")
plt.close(fig)

# =====================================================================
# Fig 3. Boundary sweeps (実験5): gamma and cs
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
gs = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
ls = [bd[("gamma", g)]["lam"] for g in gs]
ax.axvspan(0.19, 0.51, color="C3", alpha=0.07, label="boundary band")
ax.plot(gs, ls, "o-", color="C1", ms=4, lw=1.2)
cismin = ls[0]
ax.plot(gs, [cismin]*len(gs), "--", color="gray", lw=0.7, label="$\\lambda_{\\min}$ at deep ROI")
ax.set_yscale("log")
ax.set_xlabel("memory decay $\\gamma$")
ax.set_ylabel("$\\lambda_{\\min}$")
ax.set_title("$\\gamma$ sweep: no zero crossing")
ax.legend(frameon=False)

ax = axes[1]
css = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
ls = [bd[("cs", c)]["lam"] for c in css]
ax.plot(css, ls, "s-", color="C2", ms=4, lw=1.2)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("symmetric coupling $c_s$")
ax.set_ylabel("$\\lambda_{\\min}$")
ax.set_title("$c_s$ sweep ($\\gamma=0.1$): $\\lambda_{\\min}\\to0.085$")
for c, v in [(1.8, 0.085)]:
    ax.annotate(f"{v:.3f}", (c, v), xytext=(1.35, 0.28), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig3_BoundarySweeps.png")
plt.close(fig)

# =====================================================================
# Fig 4. High-dimensional PCA submanifolds (実験7)
# =====================================================================
gammas7 = [0.05, 0.1, 0.2, 0.3, 0.5]
ds = [2, 3, 4, 5]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(gammas7)))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
for g, c in zip(gammas7, colors):
    y = [np.mean([hd[(g, s, d)]["lam"] for s in ("42", "100")]) for d in ds]
    ax.plot(ds, y, "o-", color=c, ms=3.5, lw=1.1, label=f"$\\gamma$={g:g}")
ax.set_yscale("log")
ax.set_xlabel("manifold dimension $d$ (PCA)")
ax.set_ylabel("$\\lambda_{\\min}(g_d)$")
ax.set_title("High-dim: all $\\lambda_{\\min}>0$")
ax.legend(frameon=False, ncol=1)
# global minimum annotation
print("global min check (exp7):", min(hd[(0.05, s, 5)]["lam"] for s in ("42", "100")))

ax = axes[1]
for g, c in zip(gammas7, colors):
    r = [np.mean([(hd[(g, s, d)]["lam"] / hd[(g, s, d)]["gauss"]) for s in ("42", "100")]) for d in ds]
    ax.plot(ds, r, "o-", color=c, ms=3.5, lw=1.1)
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_xlabel("manifold dimension $d$ (PCA)")
ax.set_ylabel("emp. $\\lambda_{\\min}$ / matched-G $\\lambda_{\\min}$")
ax.set_title("Sharpness ratio (KDE-matched G) $\\gtrsim 0.5$")
fig.tight_layout()
fig.savefig(FIG_DIR / "Fig4_HighDim_PCA.png")
plt.close(fig)
print("Fig4 right panel (G_est-based ratio):")
for g in gammas7:
    r = [np.mean([(hd[(g, s, d)]["lam"] / hd[(g, s, d)]["gauss"]) for s in ("42", "100")]) for d in ds]
    print("  gamma=%g -> " % g, ", ".join("%.3f" % v for v in r))

# =====================================================================
# Fig 5. Boundary axes & phase variables (実験8)
#   (a) D -> 0 axis      (b) c_mem axis     (c) polar g_theta,theta
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))

ax = axes[0]
Ds = [0.05, 0.02, 0.01, 0.005]
for g, c, mk in [(0.1, "C1", "o"), (0.2, "C2", "s"), (1.0, "C3", "v")]:
    y = [pd_[("A", g, 5.0, D, "42")]["lam"] for D in Ds]
    ax.plot(Ds, y, mk+"-", color=c, ms=4, lw=1.1, label=f"$\\gamma$={g:g}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel("noise amplitude $D$")
ax.set_ylabel("$\\lambda_{\\min}$")
ax.set_title("(a) $D\\to0$: circulating bounded")
ax.legend(frameon=False)

ax = axes[1]
cms = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
y = [pd_[("B", 0.1, c, 0.05, "42")]["lam"] for c in cms]
k = [pd_[("B", 0.1, c, 0.05, "42")]["kappa"] for c in cms]
ax.semilogx(cms, y, "o-", color="C1", ms=4, lw=1.1, label="$\\lambda_{\\min}$")
ax.semilogx(cms, k, "s--", color="C4", ms=3, lw=1.0, label="$\\kappa$")
ax.axvspan(0.1, 0.2, color="C0", alpha=0.10)
ax.set_xlabel("memory gain $c_{mem}$")
ax.set_ylabel("$\\lambda_{\\min},\\ \\kappa$")
ax.set_title("(b) $c_{mem}$: onset non-singular")
ax.legend(frameon=False)

ax = axes[2]
gs8 = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
rat = []
for g in gs8:
    vals = [pd_[("C", g, 5.0, 0.05, s)]["g_tt"] / pd_[("C", g, 5.0, 0.05, s)]["g_tt_G"]
            for s in ("42", "100")]
    rat.append(np.mean(vals))
ax.plot(gs8, rat, "o-", color="C0", ms=4, lw=1.2)
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(gs8)
ax.set_xticklabels([f"{g:g}" for g in gs8])
ax.set_xlabel("$\\gamma$")
ax.set_ylabel("$g_{\\theta\\theta}$ / matched-G $g_{\\theta\\theta}$")
ax.set_title("(c) phase direction: $g_{\\theta\\theta}$ finite ($\\approx2$-$3\\times$)")

fig.tight_layout()
fig.savefig(FIG_DIR / "Fig5_BoundaryAxes_Phase.png")
plt.close(fig)

print("figures written to:", FIG_DIR)
for p in sorted(FIG_DIR.iterdir()):
    print(" -", p.name, f"({p.stat().st_size/1024:.0f} KB)")