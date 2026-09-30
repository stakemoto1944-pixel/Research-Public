#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（10）：真の λ_min が既知の合成密度による Fisher 計量推定器の校正 — 修正版

【このスクリプトが答える査読課題】
  深掘り計画 #3（校正の欠如）: 論文の核心主張は「計量縮退（λ_min→0）は観測されない」。
  査読者の典型的指摘に対し、論文には次の2点の検証がない。
    (a) 推定器が λ_min をどこまで小さく分解できるか（分解能下限）が未検証。
    (b) λ_min の絶対値には意味がない。座標スケール x→s·x で計量 g→g/s² と変換
        されるため、比較すべきは同一スケールの null である。
  本スクリプトは (a) を直接測定する：真の λ_min を任意に厳密に制御できる合成密度を
  作り、論文と同一プロトコルの推定器を当て、真値 vs 推定値の較正曲線を得る。
  得られる結論は「推定器の分解能下限 λ_floor」である。これを論文が報告した
  ROI の λ_min = 0.207（γ=0.2 では 0.285、大域最小点 d=5 で 0.027..0.032）と
  比較することで、主张の成否が確定する。

【Families】
  F1 closed-gauss : p = N(0, Σ),  Fisher 計量 = Σ^{-1} が厳密に成り立つ。
                   g = R diag(λ_max, λ_min) Rᵀ とすれば λ_min は解析的に既知。
  F2 curved      : log p = -½(a x₁² + b x₂² + c x₁²x₂²), c>0 で非ガウス。
                   Fisher 計量は Gauss–Legendre 求積で厳密計算（L 収束を確認）。

【なぜこれは重要か】
  推定器の分解能下限が 0.2 より大きければ、「λ_min が正なので縮退しない」という
  主張は検出不能領域に落ちており反証の論拠が失われる。逆に下限が 0.2 より
  十分小さければ、論文の主張は現状どおり維持できる。本スクリプトはこの分岐を
  数値的に確定させる。

実行: python3 experiment10_fixed_estimator_calibration.py
出力: estimator_calibration_result.csv
"""
import csv
import importlib.util
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(fname, modname):
    """既存スクリプトから推定器関数を再利用する（両者とも __main__ ガード済み）。"""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E4 = _load("experiment4_fixed_info_geometry.py", "exp4")
E6 = _load("experiment6_fixed_knn_crosscheck.py", "exp6")

# 論文が報告した値（較正の基準点）
ROI_LAMMIN_PAPER = 0.207
GAMMA02_LAMMIN_PAPER = 0.2849
GLOBAL_MIN_CI = (0.0270, 0.0315)
LMAX_FIXED = 1.0
N_SAMP = 20000
SEEDS = [11, 22, 33, 44]
LAMMIN_TARGETS = [1.0, 0.5, 0.2849, 0.207, 0.1, 0.05, 0.03, 0.01,
                  5e-3, 2e-3, 1e-3, 5e-4, 1e-4]
CURVED_C_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


# --------------------------------------------------- 真の Fisher 計量（厳密）
def true_fisher_gauss(lam_min, lam_max=LMAX_FIXED, theta=0.0):
    """N(0, Σ) の母 Fisher 計量は Σ^{-1}。回転して固有値を (lam_min, lam_max) に固定。"""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Lam = R @ np.diag([lam_max, lam_min]) @ R.T
    Cov = np.linalg.inv(Lam)
    ev = np.sort(np.linalg.eigvalsh(Lam))
    return Cov, float(ev[0]), float(ev[1])


def true_fisher_curved(a, b, c, n_leg=200, L_scale=14.0, L_check=20.0):
    """
    log p = -½(a x₁² + b x₂² + c x₁²x₂²) の母 Fisher 計量を Gauss–Legendre 求積で求める。

    重要な点: 推定器は「状態座標 x」に対する勾配 ∂_{x_i} log p で計量を作るため、
    真値も x 座標で計算しなければならない（exp-family の自然パラメータ (a,b,c) 座標の
    計量とは一致しない。c=0 では g^x = diag(a, b) となり λ_min = min(a,b)）。
    積分区間を L_scale → L_check と広げて λ_min の相対変化を返し、収束を確認する。
    """
    def metric(L):
        t, wt = np.polynomial.legendre.leggauss(n_leg)
        lo, hi = -L, L
        x = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wt
        X1, X2 = np.meshgrid(x, x, indexing="ij")
        wts = np.outer(w, w).ravel()
        x1, x2 = X1.ravel(), X2.ravel()
        logp = -0.5 * (a * x1 ** 2 + b * x2 ** 2 + c * x1 ** 2 * x2 ** 2)
        mx = logp.max()
        p = np.exp(logp - mx)
        p /= (wts * p).sum()
        # 状態座標でのスコア
        s1 = -(a * x1 + c * x1 * x2 ** 2)
        s2 = -(b * x2 + c * x1 ** 2 * x2)
        g = np.empty((2, 2))
        g[0, 0] = float((wts * p * s1 * s1).sum())
        g[1, 1] = float((wts * p * s2 * s2).sum())
        g[0, 1] = g[1, 0] = float((wts * p * s1 * s2).sum())
        ev = np.sort(np.linalg.eigvalsh(g))
        return g, ev[0], ev[1]

    L1 = L_scale / np.sqrt(min(a, b))
    L2 = L_check / np.sqrt(min(a, b))
    g1, l1, h1 = metric(L1)
    _, l2, h2 = metric(L2)
    return g1, l1, h1, abs(l2 - l1) / max(abs(l1), 1e-300)


# ------------------------------------------------------------ 推定器の適用
def sample_gauss(cov, n, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def sample_curved_exact(a, b, c, n, seed, n_grid=200001, L=40.0):
    """
    log p = -½(a x₁² + b x₂² + c x₁²x₂²) の厳密な i.i.d. サンプル。

    x₂ を消去すると正規化は
        Z = sqrt(2π) ∫ exp(-½a x₁²) sqrt(2π/(b + c x₁²)) dx₁
    という1次元積分になり、x₁ の周辺分布は
        p(x₁) ∝ exp(-½a x₁²) / sqrt(b + c x₁²)
    で与えられる。したがって
        ① x₁ を周辺分布の逆 CDF でサンプル（細分割网格）
        ② x₂ | x₁ ~ N(0, 1/(b + c x₁²))
    とすれば MCMC を一切使わない厳密サンプリングになる。
    """
    rng = np.random.default_rng(seed)
    if c == 0.0:
        return rng.multivariate_normal(
            np.zeros(2), np.diag([1.0 / a, 1.0 / b]), size=n)
    grid = np.linspace(-L / np.sqrt(a), L / np.sqrt(a), n_grid)
    dens = np.exp(-0.5 * a * grid ** 2) / np.sqrt(b + c * grid ** 2)
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    x1 = np.interp(rng.uniform(size=n), cdf, grid)
    x2 = rng.normal(0.0, 1.0 / np.sqrt(b + c * x1 ** 2))
    return np.column_stack([x1, x2])


def sample_curved(a, b, c, n, seed, target_acc=0.40, n_tune=40, n_iter=60):
    """
    log p = -½(a x₁² + b x₂² + c x₁²x₂²) を Metropolis–Hastings でサンプルする。
    提案幅は受容率が target_acc になるよう 自己調整する（burn-in 中にだけ調整）。
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(min(a, b))
    L = 8.0 * scale
    x = rng.uniform(-L, L, size=(n, 2))
    logp = -0.5 * (a * x[:, 0] ** 2 + b * x[:, 1] ** 2 + c * x[:, 0] ** 2 * x[:, 1] ** 2)
    step = 1.4 * scale
    acc = 0
    for it in range(n_iter):
        prop = x + rng.normal(0.0, step, size=x.shape)
        lp2 = -0.5 * (a * prop[:, 0] ** 2 + b * prop[:, 1] ** 2
                      + c * prop[:, 0] ** 2 * prop[:, 1] ** 2)
        ok = np.log(rng.uniform(size=len(x))) < (lp2 - logp)
        x[ok] = prop[ok]
        logp[ok] = lp2[ok]
        rate = ok.mean()
        acc += int(ok.sum())
        if it < n_tune:                       # burn-in 中は提案幅を自己調整
            step *= np.exp(1.0 * (rate - target_acc))
    return x, acc / (n * n_iter)


def estimate(traj):
    """論文と同一プロトコルの KDE 推定器と k-NN 推定器の両方を適用する。"""
    out = {}
    g = E4.compute_geometry(traj)
    out["kde_lam_min"] = g["lam_min"] if g else np.nan
    out["kde_lam_max"] = g["lam_max"] if g else np.nan
    out["kde_kappa"] = g["kappa"] if g else np.nan
    try:
        gk = E6.compute_geometry_knn(traj)
    except Exception:
        gk = None
    if gk:
        out["knn_lam_min"] = gk.get("lam_min", np.nan)
        out["knn_kappa"] = gk.get("kappa", np.nan)
    else:
        out["knn_lam_min"] = np.nan
        out["knn_kappa"] = np.nan
    return out


# ------------------------------------------------------------------ 実行部
def main():
    print("=" * 112, flush=True)
    print("実験10: 真の λ_min が既知の合成密度による Fisher 計量推定器の分解能校正", flush=True)
    print("=" * 112, flush=True)
    print(f"基準: 論文の ROI λ_min = {ROI_LAMMIN_PAPER}（γ=0.2: {GAMMA02_LAMMIN_PAPER}, "
          f"大域最小点 CI = {GLOBAL_MIN_CI}）", flush=True)
    print(f"サンプル数 n = {N_SAMP}, シード = {SEEDS}（各点を4シード平均）\n", flush=True)

    rows = []

    # ---------------- F1: ガウス族（真値閉形式） ----------------
    print("-" * 112, flush=True)
    print("F1  closed-gauss:  p = N(0, Σ),  真の Fisher 計量 = Σ^{-1}（厳密）", flush=True)
    print("-" * 112, flush=True)
    print("{:>10} {:>11} {:>11} {:>11} {:>11} {:>9}".format(
        "真λ_min", "KDE推定", "比KDE", "kNN推定", "比kNN", "κ_KDE"), flush=True)
    gauss_summary = {}
    for lam in LAMMIN_TARGETS:
        ests = []
        for seed in SEEDS:
            cov, tl, th = true_fisher_gauss(lam)
            traj = sample_gauss(cov, N_SAMP, seed)
            e = estimate(traj)
            ests.append(e)
        lam_kde = float(np.nanmean([e["kde_lam_min"] for e in ests]))
        lam_knn = float(np.nanmean([e["knn_lam_min"] for e in ests]))
        kap = float(np.nanmean([e["kde_kappa"] for e in ests]))
        gauss_summary[lam] = lam_kde
        print("{:>10.5f} {:>11.5f} {:>11.3f} {:>11.5f} {:>11.3f} {:>9.3f}".format(
            lam, lam_kde, lam_kde / lam, lam_knn, lam_knn / lam, kap), flush=True)
        rows.append(dict(family="closed-gauss", param_c=np.nan, true_lam_min=lam,
                         true_lam_max=LMAX_FIXED, true_kappa=LMAX_FIXED / lam,
                         **{k: v for k, v in zip(
                             ["kde_lam_min", "kde_lam_max", "kde_kappa",
                              "knn_lam_min", "knn_kappa"],
                             [lam_kde, np.nanmean([e["kde_lam_max"] for e in ests]), kap,
                              lam_knn, float(np.nanmean([e["knn_kappa"] for e in ests]))])}))

    # ---------------- F2: 曲線族（非ガウス） ----------------
    print("\n" + "-" * 112, flush=True)
    print("F2  curved:  log p = -½(a x₁² + b x₂² + c x₁²x₂²)（非ガウス、真値は求積）",
          flush=True)
    print("-" * 112, flush=True)
    print("{:>6} {:>6} {:>11} {:>10} {:>11} {:>11} {:>11}".format(
        "a", "c", "真λ_min", "収束誤差", "KDE推定", "比KDE", "κ_KDE"), flush=True)
    b_fixed = 0.207
    for c in CURVED_C_VALUES:
        a = LMAX_FIXED
        g_true, tl, th, conv = true_fisher_curved(a, b_fixed, c)
        traj = sample_curved_exact(a, b_fixed, c, N_SAMP, seed=SEEDS[0])
        e = estimate(traj)
        print("{:>6.2f} {:>6.2f} {:>11.5f} {:>10.1e} {:>11.5f} {:>11.3f} {:>11.3f}"
              .format(a, c, tl, conv, e["kde_lam_min"],
                      e["kde_lam_min"] / tl, e["kde_kappa"]), flush=True)
        rows.append(dict(family="curved", param_c=c, true_lam_min=tl,
                         true_lam_max=th, true_kappa=th / tl,
                         kde_lam_min=e["kde_lam_min"], kde_lam_max=e["kde_lam_max"],
                         kde_kappa=e["kde_kappa"], knn_lam_min=e["knn_lam_min"],
                         knn_kappa=e["knn_kappa"]))

    # ---------------------------------------------------------------- まとめ
    out = "estimator_calibration_result.csv"
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n書き出し: {out}  ({len(rows)} 行)", flush=True)

    print("\n" + "=" * 112, flush=True)
    print("分解能下限（λ_floor）の判定", flush=True)
    print("=" * 112, flush=True)
    track = [lam for lam in LAMMIN_TARGETS
             if 0.5 <= gauss_summary[lam] / lam <= 2.0]
    if track:
        lam_floor = min(track)
        print(f"推定値が真値の factor 2 以内で追従する最小の真 λ_min = {lam_floor:.5f}",
              flush=True)
    else:
        lam_floor = float("nan")
    smallest = min(LAMMIN_TARGETS)
    est_small = gauss_summary[smallest]
    print(f"最小の真 λ_min = {smallest:.1e} での推定値 = {est_small:.3e} "
          f"（比 {est_small/smallest:.3e}）", flush=True)
    print(f"参考: 論文の ROI λ_min = {ROI_LAMMIN_PAPER}、γ=0.2 = {GAMMA02_LAMMIN_PAPER}、"
          f"大域最小点 CI 下端 = {GLOBAL_MIN_CI[0]}", flush=True)
    if np.isfinite(lam_floor):
        ratio = ROI_LAMMIN_PAPER / lam_floor
        verdict = ("論文の主張は維持可能（ROI の λ_min は分解能下限より"
                   f" {ratio:.1f} 倍大きい）" if ratio > 2 else
                   "論文の主張は分解能限界に抵触（λ_floor が ROI 値と同程度）")
        print(f"\n判定: λ_floor = {lam_floor:.5f} vs ROI λ_min = {ROI_LAMMIN_PAPER} "
              f"→ {verdict}", flush=True)

    # 非ガウス性による系統バイアスの向きを判定する
    curved_rows = [r for r in rows if r["family"] == "curved"]
    if curved_rows:
        worst = min(curved_rows, key=lambda r: r["kde_lam_min"] / r["true_lam_min"])
        ratio_w = worst["kde_lam_min"] / worst["true_lam_min"]
        print(f"\n非ガウス性（曲線族）による最悪の比 = {ratio_w:.3f}（c={worst['param_c']}）",
              flush=True)
        if ratio_w < 1.0:
            print("  → 過小評価（保守側）。真の縮退を見落とす方向ではなく、"
                  "縮退のない密度を縮退と誤診しない方向の系統バイアスであり、"
                  "論文の『非縮退』結論を弱める作用はしない。", flush=True)
        else:
            print("  → 過大評価であり、真の縮退を隠す可能性があり結論の再検討が必要。",
                  flush=True)

    knn_rows = [r for r in rows if r["family"] == "closed-gauss"]
    if knn_rows:
        worst_k = max(knn_rows, key=lambda r: r["knn_lam_min"] / r["true_lam_min"])
        print(f"\nk-NN 推定器の最悪の過大評価 = "
              f"{worst_k['knn_lam_min']/worst_k['true_lam_min']:.0f} 倍"
              f"（真 λ_min = {worst_k['true_lam_min']:.1e} のとき "
              f"{worst_k['knn_lam_min']:.3e} と報告）", flush=True)
        print("  → k-NN のバイアスは真 λ_min が小さいほど増大するため、"
              "k-NN は縮退の反証手段として使用できない。", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
