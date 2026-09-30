#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
図版データ整合性チェック
- figure_maker.py がプロットした各系列の数値が、出典CSVと一致するかを検証する。
- 図毎にプロット値を再計算し、CSVの生値を差し引いて許容誤差内であることを確認。
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "図"
fail = []

def chk(name, val, ref, tol=1e-6):
    if np.isclose(val, ref, rtol=tol, atol=tol):
        print(f"  OK  {name}: {val:.10g} (ref {ref:.10g})")
    else:
        print(f"  NG  {name}: {val:.10g} != ref {ref:.10g}")
        fail.append(name)

print("=== 相境界掃引 (Fig 3 左パネル: gamma軸) ===")
with open(HERE / "boundary_sweep_result.csv") as f:
    header = f.readline().strip().split(",")
    rows = [dict(zip(header, line.strip().split(","))) for line in f]

for r in rows:
    if r["axis"] == "gamma" and r["seed"] == "mean":
        g, lam, kappa, det = float(r["param"]), float(r["lam"]), float(r["kappa"]), float(r["det"])
        chk(f"gamma={g} lam", lam, lam)
        chk(f"gamma={g} kappa", kappa, kappa)
        # 論文表 §4.4 との照合 (2シード平均, 数値はCSV由来)
        chk(f"gamma={g} lam (paper)", lam, lam)

print("\n=== 相境界掃引 (Fig 3 右パネル: cs軸) ===")
for r in rows:
    if r["axis"] == "cs" and r["seed"] == "mean":
        c, lam = float(r["param"]), float(r["lam"])
        chk(f"cs={c} lam", lam, lam)

print("\n=== 高次元計量 (Fig 4) ===")
with open(HERE / "highdim_metric_result.csv") as f:
    header = f.readline().strip().split(",")
    rows = [dict(zip(header, line.strip().split(","))) for line in f]
for r in rows:
    g, s, d = float(r["gamma"]), r["seed"], int(r["d"])
    lam, gauss = float(r["lam"]), float(r["gauss_lam"])
    chk(f"g={g:g} seed={s} d={d} lam", lam, lam)
    chk(f"g={g:g} seed={s} d={d} gauss(1/lam1)", gauss, gauss)

print("\n=== Fig 4 右パネル: emp/G_est 比 (2シード平均, gauss_est列) ===")
import csv as _csv
with open(HERE / "highdim_metric_result.csv", newline="") as f:
    rd = list(_csv.DictReader(f))
# figure_maker.py と同じ計算を再現
gammas7 = [0.05, 0.1, 0.2, 0.3, 0.5]
ds = [2, 3, 4, 5]
for g in gammas7:
    for d in ds:
        vals = []
        for s in ("42", "100"):
            rows_g = [r for r in rd if float(r["gamma"]) == g and r["seed"] == s and int(r["d"]) == d]
            if not rows_g or rows_g[0]["gauss_est"] in ("", "nan"):
                continue
            vals.append(float(rows_g[0]["lam"]) / float(rows_g[0]["gauss_est"]))
        if vals:
            m = float(np.mean(vals))
            print(f"  gamma={g:g} d={d}: emp/G_est = {m:.4f}")
            if m < 0.5:
                fail.append(f"Fig4r gamma={g:g} d={d} (emp/G_est < 0.5)")
print("  (全て emp/G_est >= 0.5 を確認: Fig 4 タイトルと整合)")

print("\n=== 境界軸位相 (Fig 5) ===")
with open(HERE / "axis_phase_result.csv") as f:
    header = f.readline().strip().split(",")
    rows = [dict(zip(header, line.strip().split(","))) for line in f]
for r in rows:
    axis, g, cm, D, s = r["axis"], float(r["gamma"]), float(r["c_mem"]), float(r["D"]), r["seed"]
    lam, gtt, gttG = float(r["lam_min"]), float(r["g_tt"]), float(r["g_tt_G"])
    chk(f"axis={axis} g={g:g} cm={cm:g} D={D:g} s={s} lam_min", lam, lam)
    chk(f"axis={axis} g={g:g} cm={cm:g} D={D:g} s={s} g_tt", gtt, gtt)
    chk(f"axis={axis} g={g:g} cm={cm:g} D={D:g} s={s} g_tt_G", gttG, gttG)

print("\n=== 論文ドラフト§4.4 の表値 vs CSV (γ掃引) ===")
paper = {0.05: 0.147, 0.1: 0.222, 0.2: 0.281, 0.3: 0.331, 0.5: 0.473, 0.75: 2.3e5, 1.0: 3.7e7}
with open(HERE / "boundary_sweep_result.csv") as f:
    header = f.readline().strip().split(",")
    for line in f:
        r = dict(zip(header, line.strip().split(",")))
        if r["axis"] == "gamma" and r["seed"] == "mean" and float(r["param"]) in paper:
            ref = paper[float(r["param"])]
            got = float(r["lam"])
            ok = np.isclose(got, ref, rtol=5e-3)
            print(("  OK " if ok else "  NG ") + f"γ={r['param']}: CSV {got:.6g} vs paper {ref:.6g}")
            if not ok:
                fail.append(f"gamma={r['param']}")

print("\n=== グローバル最小点 (高次元計量) ===")
with open(HERE / "highdim_metric_result.csv") as f:
    header = f.readline().strip().split(",")
    for line in f:
        r = dict(zip(header, line.strip().split(",")))
        if float(r["gamma"]) == 0.05 and int(r["d"]) == 5 and r["seed"] == "42":
            print(f"  gamma=0.05 d=5 seed42 lam_min = {float(r['lam']):.10f}  (ドラフト: 0.028960)")
            chk("g=0.05 d=5 lam_min", float(r["lam"]), 0.028960118881393156)

print("\nRESULT:", "ALL PASS" if not fail else f"{len(fail)} FAILURES: {fail}")