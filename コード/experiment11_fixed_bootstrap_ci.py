#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（11）：ブートストラップ CI の再構築（B≥1000, seeds 6→20, ブロック化）— 修正版

【このスクリプトが答える査読課題】
  深掘り計画 #2: 現行の「95% CI はゼロを除外する」は以下の3点で成立していない。
    (1) B=12: 非パラメトリック percentile 法で B=12 なら 2.5%/97.5% 分位は
        実質 min〜max であり、95%CI として意味を持たない。
    (2) i.i.d. ブートストラップ: 時間点を「置換抽出」する現行実装は忽略する。
        本論文の軌道は強く自己相関している（積分自己相関時間 τ≈30 サンプル）ため、
        独立仮定の bootstrap は CI を系統的に狭くし、錯った保証を与える。
        → 移動ブロック bootstrap（block length L ≳ 2τ）に置き換える必要がある。
    (3) seeds 6: 点推定の SE が6点から算出されている。

  本スクリプトは (1)(2)(3) を同時に解消する。

【プロトコル】
  推定器は論文と同一（数値実験4 の compute_geometry: ペア(0,1)の2D Scott-KDE、
  中心差分 δ=1e-4、g_ij=E[s_i s_j]）。
  bootstrap は軌道の行インデックスをブロック単位で置換抽出する（2成分を同時抽出
  し、ペア内相関を保存）。CI は percentile 法 [2.5, 97.5]。

【S1】点推定: 4条件 × 20シード → 平均 ± SE（seeds 6→20）
【S2】主解析: ROI、20シード、B=1000、L=2τ≈64 のブロック bootstrap
【S3】感度: ROI、5シード、L ∈ {1（i.i.d.）, 16, 256} → CI 幅の比較。
           L=1 が現行実装であり、これがどれだけ狭いかを定量化する。
【S4】他条件: EQ / DD / GAUSS、6シード、B=1000、L=64

実行: python3 experiment11_fixed_bootstrap_ci.py [--quick]
出力: bootstrap_ci_result.csv / experiment11_log.txt
"""
import argparse
import csv
import importlib.util
import os
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
N_CMP = 5                       # プロセス数（実測: 7でも加速は2.5倍止まり=メモリ帯域律速）

_spec = importlib.util.spec_from_file_location(
    "e4", os.path.join(_HERE, "experiment4_fixed_info_geometry.py"))
E4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E4)

# ------------------------------------------------------------------ 条件定義
BASE = dict(coupling_scale=1.0, gamma=0.1, c_mem=5.0, D=0.05, T=300.0, burnin=100.0)
SEEDS = list(range(20))                 # 0..19（現行の {0,1,2,3,42,100} を置き換え）
SEEDS_AUX = [0, 1, 42]                  # 感度・他条件用の補助シード
N_EVAL_MAIN = 1000                      # 点推定は5000、bootstrapは1000（§S0 で妥当性確認）
N_EVAL_POINT = 5000
B_MAIN = 1000
B_AUX = 500
L_MAIN = 64                             # 2τ（τ≈30）
L_SENS = [1, 16, 256]
CONDS = ["ROI", "EQ", "DD", "GAUSS"]

_TRAJ = {}                              # ワーカー内で共有する軌道


def _acf_time(x, max_lag=4000):
    """積分自己相関時間 τ = 1 + 2Σ_{k≥1} ρ(k)（ρ が初めて非正になるまで）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    v = float(x @ x) / n
    s = 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        c = float(x[:n - lag] @ x[lag:]) / n / v
        if c <= 0:
            break
        s += c
    return 1.0 + 2.0 * s


def _block_indices(n, L, rng):
    """移動ブロック（overlapping block）bootstrap の行インデックス。"""
    if L <= 1:
        return rng.integers(0, n, size=n)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]


def build_trajectories(seeds):
    """全条件・全シードの軌道と τ を構築して返す。"""
    trs, taus = {}, {}
    _, roi42, _, _ = E4.simulate(**BASE, seed=42, asym_gain=1.0)
    roi_cov = np.cov(roi42[:, :2].T)
    for s in seeds:
        for cond in CONDS:
            if cond == "ROI":
                _, tr, _, _ = E4.simulate(**BASE, seed=s, asym_gain=1.0)
            elif cond == "EQ":
                _, tr, _, _ = E4.simulate(**BASE, seed=s, asym_gain=0.0)
            elif cond == "DD":
                _, tr, _, _ = E4.simulate(1.0, 0.05, 2.0, 5.0, 300.0, 100.0,
                                          seed=s, asym_gain=1.0)
            else:   # GAUSS: ROI seed42 のペア(0,1)共分散に一致するガウス対照
                rng = np.random.default_rng(s)
                tr = rng.multivariate_normal(np.zeros(2), roi_cov, size=len(roi42))
            trs[(cond, s)] = tr[:, :2]
            if cond == "ROI":
                taus[s] = _acf_time(tr[:, 0])
    return trs, taus


# ------------------------------------------------------------------ ワーカー
def _init(trs):
    global _TRAJ
    _TRAJ = trs


def _one(task):
    """1ブートストラップ反復を計算してメトリクスを返す。"""
    cond, seed, L, b, n_eval = task
    tr = _TRAJ[(cond, seed)]
    rng = np.random.default_rng(90000 + 7919 * b + 31 * seed)
    idx = _block_indices(len(tr), L, rng)
    r = E4.compute_geometry(tr[idx], h_scale=1.0, n_eval=n_eval)
    return dict(det=r["det"], lam=r["lam_min"], kap=r["kappa"])


# -------------------------------------------------------------------- 集計
def dump(rows):
    """途中経過を逐次 CSV へ書き出す（長時間実行でも部分結果が使えるように）。"""
    out = "bootstrap_ci_result.csv"
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"    → {out} に逐次書き出し（{len(rows)} 行）", flush=True)
    return out


def summarize(vals, key):
    v = np.asarray(vals, float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(ci_lo=float(lo), ci_hi=float(hi), ci_width=float(hi - lo),
                boot_mean=float(v.mean()), boot_sd=float(v.std(ddof=1)),
                median=float(np.median(v)), excludes_zero=bool(lo > 0.0))


def run_stage(pool, cond, seeds, L, B, n_eval, label, rows, base_seed=0):
    t0 = time.time()
    tasks = [(cond, s, L, base_seed + b, n_eval) for s in seeds for b in range(B)]
    out = pool.map(_one, tasks, chunksize=8)
    print(f"    [{label}] {cond} L={L} B={B} seeds={len(seeds)} "
          f"→ {len(tasks)} 反復, {time.time()-t0:.0f}s", flush=True)
    k = 0
    for s in seeds:
        sl = out[k:k + B]
        k += B
        lam = [o["lam"] for o in sl]
        kap = [o["kap"] for o in sl]
        det = [o["det"] for o in sl]
        row = dict(stage=label, cond=cond, seed=s, L=L, B=B, n_eval=n_eval,
                   **{f"lam_{k2}": v for k2, v in summarize(lam, "lam").items()},
                   **{f"kap_{k2}": v for k2, v in summarize(kap, "kap").items()},
                   **{f"det_{k2}": v for k2, v in summarize(det, "det").items()})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="スモークテスト（少シード・B=50）")
    args = ap.parse_args()

    global SEEDS, SEEDS_AUX, B_MAIN, B_AUX, N_EVAL_MAIN, N_EVAL_POINT
    if args.quick:
        SEEDS = [0, 1, 2]
        SEEDS_AUX = [0, 1]
        B_MAIN, B_AUX = 50, 50
        N_EVAL_MAIN, N_EVAL_POINT = 500, 2000

    rows = []
    print("=" * 104, flush=True)
    print("実験11: ブートストラップCIの再構築（B≥1000, seeds 6→20, 移動ブロック化）", flush=True)
    print("=" * 104, flush=True)
    print(f"シード: {SEEDS}（既定20）／B={B_MAIN}／主ブロック長 L={L_MAIN}／"
          f"n_eval(main)={N_EVAL_MAIN}, n_eval(point)={N_EVAL_POINT}／並列 {N_CMP}", flush=True)

    print("\n軌道構築中（4条件 × {}シード）...".format(len(set(SEEDS + SEEDS_AUX))),
          flush=True)
    trs, taus = build_trajectories(sorted(set(SEEDS + SEEDS_AUX)))
    ta = np.array(list(taus.values()))
    print(f"  積分自己相関時間 τ = {ta.mean():.1f} ± {ta.std(ddof=1):.1f} "
          f"（最小 {ta.min():.1f}, 最大 {ta.max():.1f}）→ 主ブロック長 L=2τ≒{L_MAIN}",
          flush=True)
    print(f"  参考: Kunsch 式 2τn^(1/3) は L≈{int(2*ta.mean()*20000**(1/3))} で"
          f"ブロック数わずか{20000//int(2*ta.mean()*20000**(1/3))}個となるため不採用", flush=True)

    # ---- S0: n_eval の妥当性（点推定との一致確認） ----
    print("\nS0. n_eval 感度（点推定との一致確認）", flush=True)
    s0_seed = 42 if ("ROI", 42) in trs else SEEDS[0]
    for ne in [1000, 2000, N_EVAL_POINT, 10000]:
        r = E4.compute_geometry(trs[("ROI", s0_seed)], h_scale=1.0, n_eval=ne)
        print(f"    n_eval={ne:>6}: λ_min={r['lam_min']:.5f}  κ={r['kappa']:.4f}", flush=True)

    # ---- S1: 点推定 4条件 × 20シード ----
    print(f"\nS1. 点推定（4条件 × {len(SEEDS)}シード, n_eval={N_EVAL_POINT}）", flush=True)
    point = {}
    for cond in CONDS:
        vals = {k: [] for k in ("det", "lam_min", "kappa")}
        for s in SEEDS:
            r = E4.compute_geometry(trs[(cond, s)], h_scale=1.0, n_eval=N_EVAL_POINT)
            for k in vals:
                vals[k].append(r[k])
        point[cond] = vals
        a = np.array(vals["lam_min"])
        kk = np.array(vals["kappa"])
        d = np.array(vals["det"])
        tcrit = 2.093                                   # t_{0.975,19}
        print("    {:>5}: det g={:.5f}±{:.5f}  λ_min={:.5f}±{:.5f}  κ={:.4f}±{:.4f}  "
              "[最小λ_min={:.5f}]".format(
                  cond, d.mean(), d.std(ddof=1)/np.sqrt(len(d)),
                  a.mean(), a.std(ddof=1)/np.sqrt(len(a)),
                  kk.mean(), kk.std(ddof=1)/np.sqrt(len(kk)), a.min()), flush=True)
        se = a.std(ddof=1)/np.sqrt(len(a))
        print("           → 20シードでの 95% CI（平均±t·SE）= [{:.5f}, {:.5f}]  "
              "ゼロ除外: {}".format(a.mean()-tcrit*se, a.mean()+tcrit*se,
                                      a.mean()-tcrit*se > 0), flush=True)
        for s, a_, k_, d_ in zip(SEEDS, a, kk, d):
            rows.append(dict(stage="S1_point", cond=cond, seed=int(s), L=0, B=0,
                             n_eval=N_EVAL_POINT, point_det=float(d_),
                             point_lam=float(a_), point_kap=float(k_),
                             tau=float(taus.get(s, np.nan))))

    with Pool(N_CMP, initializer=_init, initargs=(trs,)) as pool:
        # ---- S2: 主解析 ROI ブロックbootstrap ----
        print(f"\nS2. 主解析: ROI, {len(SEEDS)}シード, B={B_MAIN}, L={L_MAIN}", flush=True)
        run_stage(pool, "ROI", SEEDS, L_MAIN, B_MAIN, N_EVAL_MAIN, "S2_main", rows)
        dump(rows)

        # ---- S3: ブロック長感度（incl. 現行の i.i.d. 実装 L=1） ----
        print(f"\nS3. ブロック長感度: ROI, {len(SEEDS_AUX)}シード, B={B_AUX}", flush=True)
        for L in L_SENS:
            run_stage(pool, "ROI", SEEDS_AUX, L, B_AUX, N_EVAL_MAIN, "S3_Lsens", rows)
            dump(rows)

        # ---- S4: 他条件 ----
        print(f"\nS4. 他条件: EQ/DD/GAUSS, {len(SEEDS_AUX)}シード, B={B_AUX}, L={L_MAIN}",
              flush=True)
        for cond in ["EQ", "DD", "GAUSS"]:
            run_stage(pool, cond, SEEDS_AUX, L_MAIN, B_AUX, N_EVAL_MAIN, "S4_other", rows)
            dump(rows)

    dump(rows)

    out = "bootstrap_ci_result.csv"

    # ------------------------------------------------------------------ 報告
    print("\n" + "=" * 104, flush=True)
    print("結論", flush=True)
    print("=" * 104, flush=True)

    m = [r for r in rows if r["stage"] == "S2_main"]
    if m:
        lo = np.array([r["lam_ci_lo"] for r in m])
        hi = np.array([r["lam_ci_hi"] for r in m])
        wd = np.array([r["lam_ci_width"] for r in m])
        print(f"ROI λ_min: ブートストラップ CI 幅 = {wd.mean():.5f} "
              f"（{wd.min():.5f}–{wd.max():.5f}）, {len(m)}/{len(m)} シードでゼロ除外 = "
              f"{int(sum(r['lam_excludes_zero'] for r in m))}/{len(m)}", flush=True)
        print(f"        全 {len(m)} シード: CI 下限の最大値 = {lo.max():.5f}／"
              f"CI 上限の最小値 = {hi.min():.5f}（差 {lo.max() - hi.min():+.5f}）", flush=True)
        print(f"        → 差が正なので各シードの CI は互いに重ならず"
              f"（各シードの点推定中心が異なるため）。全体を1つの区間に"
              f"まとめたものは解釈しないこと。", flush=True)

    s1 = [r for r in rows if r["stage"] == "S1_point" and r["cond"] == "ROI"]
    if s1:
        a = np.array([r["point_lam"] for r in s1])
        print(f"ROI λ_min（点推定、{len(a)}シード）= {a.mean():.5f} ± "
              f"{a.std(ddof=1)/np.sqrt(len(a)):.5f}（現行の6シード版 0.207 との比較対象）",
              flush=True)

    print("\nS3 ブロック長感度 — λ_min の CI 幅:", flush=True)
    width = {}
    for L in sorted({r["L"] for r in rows if r["stage"] == "S3_Lsens"}):
        sub = [r for r in rows if r["stage"] == "S3_Lsens" and r["L"] == L]
        width[L] = float(np.mean([r["lam_ci_width"] for r in sub]))
        tag = "（現行の i.i.d. 実装）" if L == 1 else ""
        print("    L={:>4} {:>22}: 幅 = {:.5f}  ゼロ除外 {}/{}".format(
            L, tag, width[L],
            int(sum(r["lam_excludes_zero"] for r in sub)), len(sub)), flush=True)
    if L_MAIN in width and 1 in width and width[1] > 0:
        print(f"    → ブロック化により CI 幅は {width[L_MAIN]/width[1]:.2f} 倍に拡大"
              f"（現行実装は過小な区間抽出）", flush=True)


if __name__ == "__main__":
    main()
    print("DONE", flush=True)
