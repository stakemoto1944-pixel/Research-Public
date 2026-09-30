#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（15）：#6 修正理論の検証

【何を検証するか】
  Paper III（zenodo.22682953）の予測:
      「NEIP と断片相の臨界境界で Fisher 計量 g が縮退し（det g → 0）、
        行列式が**普遍指数でスケールする**」
  本論文の数値はこれを 25 格子点 × 全座標系 × 両推定器で否定した。
  ただし本論文は代替理論を示していない（"must be re-derived" で先送り）。

  本スクリプトは、**symmetry を緩めた修正理論の検証可能な帰結**を
  精密に検査する。理論の主張は以下。

  ------------------------------------------------------------------------
  規約: FPE を  ∂_t ρ = -∇·(b ρ) + D ∇²ρ  と書く（D = 拡散係数）。
        したがって  ∂_t ρ = -∇·J,  J = b ρ - D ∇ρ.
        ※ 2D ∇²ρ 規約（σ²/2 を置いた規約）では D を σ²/2 と読み替える。
  厳密恒等式（NESS、〈·〉 は定常分布についての期待）
    (L1)  g_ij = -E[ ∂_i ∂_j ln ρ ]
    (L2)  ∇ ln ρ = (1/D) ( b - J/ρ ),  J = b ρ - D ∇ρ
          ⟹  g = (1/D²) E[ F Fᵀ ],   F ≡ b - J/ρ  （有効力）
    (C1)  vᵀ g v = (1/D²) E[(v·F)²]  ⟹  λ_min = 0 ⟺ v·F ≡ 0
          すなわち縮退は「密度の対称性」ではなく
          「**軌道上での有効力の対称性**」の条件である
  ------------------------------------------------------------------------
  相方向（回転する系）への適用
    共動座標系で r ≒ r0 のとき、角運動方程式は
        θ̇ = ω0 + ω1 sin(θ - φ0) + 角拡散
    ここで **ω0（回転角速度）を 0 と置いてはいけない**。φ0 固定かつ ω0 ≠ 0 の
    定常 FPE  0 = -∂_θ[(ω0 + ω1 sin θ) p] + D_θ ∂_θ² p  に
    p ∝ e^{a cos θ} を代入すると cos θ の項が打ち消し合わず解にならない。
    正しい一次解（p = (2π)⁻¹[1+ε f],  O(ω1) 展開）は
        D_θ p₁'' - ω0 p₁' = (ω1/2π) cos θ
        p₁ = -[ω1 D_θ /(2(D_θ²+ω0²))] (cos θ + (ω0/D_θ) sin θ)
    ⟹  密度変調振幅  ε = ω1 / sqrt(D_θ² + ω0²)
    ⟹  g_θθ = π² ω1² / (2 (D_θ² + ω0²))                    …(P1, 訂正後)
    旧稿の  g_θθ = ω1²/(2 D_θ²)  は (ω0 = 0) の特殊解にすぎない。
    ★ 回転が異方性を screening する: スケールは (ω1/D_θ)² ではなく (ω1/ω0)²。
    ★ 線形領域 (ε ≪ 1) では g_θθ = π²ε²/2 ≲ π²/2 = 4.93 が上限。
  ------------------------------------------------------------------------
  潜在的関数の詳細釣り合い系（1 次元幅 w のガウシアン）
    b = -∇U,  ρ ∝ e^{-U/D}  ⟹  g = (1/D²) E[∇U ∇Uᵀ] = (1/D²) Cov(∇U)
    幅 w のガウシアンで  U = (D/2w²) x²  ⟹  **g = 1/w²**   …(W1)
    つまり計量的大小は**密度の幅の逆二乗**で決まり、噪声強度には依存しない。
    ⟹  局所化（needle）する側で g は**爆発**する（縮退ではない）
  ------------------------------------------------------------------------
  ★★ 改訂の核心その2: det g は座標不変ではない（Paper III の枠組みの欠陥）
    座標変換 x = x(q) のもとで  g_ab = Jᵀ g J,  J = ∂x/∂q
    ⟹  det g_ab = (det J)² det g      … det g は不変でない
    ⟹  一方 λ_min = 0（零固有値をもつこと）は**座標不変**
    したがって「det g → 0 が普遍指数でスケールする」という Paper III の予測は、
    スケールする変数を明示していないため **原理的に検証不能**。
    改訂理論はスケール変数を ξ（soft 方向の相関長）と明示する:
        スケーリング形  ρ(q) = ξ^{-d} f(q/ξ)  ⟹  g = ξ^{-2}〈(∂ ln f)²〉  …(P3)
        λ_min ∝ ξ_soft^{-2},  det g ∝ ξ^{-2d}
    本論文の ROI 境界は ξ が発散しない（λ_min ~ 0.2 ⟹ ξ ~ 2）
    ⟹ 縮退は起こらない。EQ 側で needle 化 ⟹ ξ_soft が減少 ⟹ g は**増大**。
  ------------------------------------------------------------------------
  ★ 改訂の核心その1: 「縮退する対称性」の意味
    φ0 が固定（lab frame）  ⟹  p(θ) が非一様 ⟹ g_θθ > 0 ⟹ 非縮退
    φ0 = ω0 t（共動）      ⟹  p(θ) が一様   ⟹ g_θθ = 0 ⟹ 縮退
    Paper III は暗黙に「全異方性が共動する（lab frame で固定されない）」と
    仮定していた。本論文の系では J_asym が **lab frame で固定された行列**なので
    仮定が破れ、g_θθ > 0 になる。──これが縮退が観測されない真の理由。

  ------------------------------------------------------------------------
  ★★ 実行結果と重大留保（2026-09-30 初回実行）
    T-D (W1)  g = 1/w²   ✅ 確認。傾き -2.000、比 0.973 = 予測した KDE バイアス
                          1/(1+0.164²) = 0.9734 と正確に一致。
    T-A (P1)             ❌ 判定不能（測定値が床の 1.4〜250 倍だが、成因は Poisson
    T-B (P1')            ❌ 判定不能。   ノイズであり理論不符ではない）
    T-C (R1)             ❌ 判定不能。
    噪声床の機構: 角度相関時間 τ_c = 1/D_θ ⟹ 観測時間中の有効独立サンプル数
      n_eff = T_obs·D_θ。σ=0.5 で n_eff=50（実測床 3.375）、σ=0.02 で
      n_eff=2（実測床 21.39）。1/√n_eff の予測比 5.00 に対し実測比 6.34。
    さらに共動臂では pattern が 2π/ω0 = 1 s 周期で回転するため有効独立
    サンプル数が pattern 周期で増える ⟹ 床がその分低下する。T-C の
    共動/固定比 1e-2 は物理的縮退ではなく**この床の差**である。
    ⟹ 本スクリプトは (P1)(P1')(R1) を検証できない。有効な検証は T-D のみ。
    ⟹ 再設計（→ 実験16）の要件: (i) ω0 を小さくする、(ii) n_eff を matched な
      null で較正して床を差し引く、(iii) 密度の角幅が bin 幅を下回らない
      領域で測る の 3 つが同時に必要。
  ------------------------------------------------------------------------

実行: python3 15_theory_verification.py
出力: theory_verification15_result.csv / experiment15_log.txt

実行時間の目安: シミュレーション 156 本（各 M=50000 ステップの Python ループ）＋
KDE-Fisher 評価 66 回。KDE は O(n_eval × n) なので n_eval=N_EVAL=1500 に
抑えてある（既存実験の 5000 ではない。2×2 計量の平均としては十分安定）。
T-A は相計量のみを測るので KDE を走らせない（λ_min は T-B で別途測定）。
全体で 10〜20 分程度。
"""
import csv
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------ パラメータ
DT = 2.0e-3
T_OBS = 100.0
M = int(T_OBS / DT)
MU = 1.0                  # Stuart–Landau: dr/dt = MU r - r^3
R0 = math.sqrt(MU)         # 安定半径 r* = sqrt(MU) = 1.0
OMEGA0 = 2.0 * math.pi * 1.0      # 回転角速度 [rad/s]
BURN = 0.2                # 過渡棄却（時間比）

SEEDS = [0, 1, 2]
SIGMA_GRID = [0.50, 0.20, 0.10, 0.05, 0.02]
OMEGA1_GRID = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]

# 実行時間の制御。KDE-Fisher は O(n_eval × n) なので評価点数を抑える。
# 2×2 計量の平均としては n_eval=1500 で十分安定する。
N_EVAL = 1500

NBIN_R, NBIN_TH = 48, 160

ROWS = []


def _log(m):
    print(m, flush=True)


def _rec(**kw):
    ROWS.append(kw)


# ------------------------------------------------------------------ 系の生成
def polar_sl(sigma, omega1, phi0_mode, seed):
    """
    極座標 Stuart–Landau ＋ 角トルク:
        r ̇ = MU r - r³ + sqrt(2 sigma) ξ_r
        θ̇ = ω0 + ω1 sin(θ - φ0) + sqrt(2 sigma / r²) ξ_θ

    phi0_mode:
      "fixed"      φ0 = 0        （lab frame で固定された異方性）
      "corotating" φ0 = ω0 t     （循環と共に回転する異方性）
    返り値: 回転後の x, y と共動位相 phi = θ - ω0 t
    """
    rng = np.random.default_rng(seed)
    r = R0
    th = 0.0
    sr = math.sqrt(2.0 * sigma * DT)
    x = np.empty(M)
    y = np.empty(M)
    phi = np.empty(M)
    for k in range(M):
        t = k * DT
        ph0 = t * OMEGA0 if phi0_mode == "corotating" else 0.0
        r += (MU * r - r ** 3) * DT + sr * rng.standard_normal()
        r = max(r, 1e-3)
        th += (OMEGA0 + omega1 * math.sin(th - ph0)
               + math.sqrt(2.0 * sigma / (r * r) * DT) * rng.standard_normal())
        x[k] = r * math.cos(th)
        y[k] = r * math.sin(th)
        phi[k] = th - OMEGA0 * t
    m0 = int(BURN * M)
    return x[m0:], y[m0:], phi[m0:]


def gaussian2d(seed, w):
    """(P2) の検証用: 標準偏差 w の等方 2 次元ガウシアン標本。噪声パラメータは無関係。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, 2)) * w


# ------------------------------------------------------------------ 計量
def fisher_cartesian(x, y, n_eval=None, h_scale=1.0):
    """
    既存プロトコル（実験4/9/13と同一）: KDE-Fisher を直交座標で評価。
    帯域は scott、中心差分 δ=1e-4、ρ は 1e-10 で clip。
    既存実験は n_eval=5000 だが、ここは実行時間のため N_EVAL=1500 を使う。
    """
    n_eval = N_EVAL if n_eval is None else n_eval
    data = np.vstack([x, y])
    idx = np.linspace(0, data.shape[1] - 1, n_eval).astype(int)
    pts = data[:, idx]
    kde = gaussian_kde(data, bw_method="scott")
    kde.set_bandwidth(kde.factor * h_scale)
    eps, delta, dim = 1e-10, 1e-4, 2
    grad = np.zeros((dim, n_eval))
    for i in range(dim):
        pp, pm = pts.copy(), pts.copy()
        pp[i] += delta
        pm[i] -= delta
        grad[i] = (np.log(np.clip(kde(pp), eps, None))
                   - np.log(np.clip(kde(pm), eps, None))) / (2.0 * delta)
    g = (grad @ grad.T) / n_eval
    ev = np.linalg.eigvalsh(g)
    return float(ev[0]), float(ev[1]), float(ev[0] * ev[1]), float(ev[1] / max(ev[0], 1e-300))


def fisher_polar(r, th, smooth=2.0):
    """
    極座標 (r, θ) で計量 g_rr, g_θθ, g_rθ を 2 次元度数分布から直接評価。
    θ は円周なので両端を周期接続して平滑化してから差分する。
    """
    H, edges_r, edges_th = np.histogram2d(
        r, th, bins=[NBIN_R, NBIN_TH],
        range=[[r.min() - 1e-9, r.max() + 1e-9], [0.0, 2 * np.pi]])
    H = gaussian_filter(H, (smooth, 0.0), mode="nearest")     # r 方向のみ平滑
    H = gaussian_filter(H, (0.0, smooth), mode="wrap")        # θ 方向は周期
    H = np.clip(H, 1e-12, None)
    L = np.log(H)
    dr = (edges_r[1] - edges_r[0])
    dth = (edges_th[1] - edges_th[0])
    # 中心差分（θ は roll で周期接続）
    # r 方向は両端 1 セルが差分できないので 0 のまま残す（形状も H と揃える）
    dL_dr = np.zeros_like(L)
    if L.shape[0] > 2:
        dL_dr[1:-1, :] = (L[2:, :] - L[:-2, :]) / (2 * dr)
    dL_dth = (np.roll(L, -1, axis=1) - np.roll(L, 1, axis=1)) / (2 * dth)
    P = H / H.sum()
    g_rr = float(np.sum(P * dL_dr ** 2))
    g_tt = float(np.sum(P * dL_dth ** 2))
    g_rt = float(np.sum(P * dL_dr * dL_dth))
    return g_rr, g_tt, g_rt


def implied_omega1(g_tt, sigma):
    """[訂正後] g_θθ = π²ω1²/(2(D_θ²+ω0²)) ⟹ ω1 = sqrt(2 g_θθ (D_θ²+ω0²))/π,  D_θ=σ/r0²。

    ※ 床以上に大きい g_θθ を代入すると整合的な ω1 にならない（過大評価になる）。
      本スクリプトでは診断用に残すが、意味を持つのは信号 > 床 のときだけ。
    """
    d_th = sigma / (R0 * R0)
    return math.sqrt(2.0 * g_tt * (d_th ** 2 + OMEGA0 ** 2)) / math.pi


def loglog_slope(x, y):
    m = (np.asarray(x) > 0) & (np.asarray(y) > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(x)[m]), np.log(np.asarray(y)[m]), 1)[0])


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    _log("=" * 100)
    _log("実験15: #6 修正理論の検証 — Fisher 計量縮退の成立条件")
    _log("=" * 100)
    _log(f"系:  ṙ = μ r - r³ + √(2σ)ξ_r   （r* = √μ = {R0:.4f}）")
    _log(f"     θ̇ = ω0 + ω1 sin(θ-φ0) + √(2σ/r²)ξ_θ,  ω0 = {OMEGA0:.4f} rad/s")
    _log(f"観測: T_obs={T_OBS}s, M={M} サンプル, dt={DT}")
    _log("検証する帰結:")
    _log("  (P1)   g_θθ = π²ω1²/(2(D_θ²+ω0²)) [訂正後]  ⟹ ω1 傾き 2.0, σ 傾き 0.0 [T-A]")
    _log("  (P1')  ω1 = 0 ⟺ g_θθ = 0 ⟺ 縮退（Paper III が暗黙に仮定した領域）      [T-B]")
    _log("  (R1)   異方性の固定/共動が縮退の有無を決める ★改訂の核心               [T-C]")
    _log("  (W1)   詳細釣り合い系で g = 1/w²（幅 w のガウシアン）                   [T-D]")
    _log("")
    _log("★ 実行前须知: 本スクリプトは 2026-09-30 の初回実行で (P1)(P1')(R1) が")
    _log("  判定不能（T-A/T-B/T-C が推定器ノイズ床に埋もれる）ことが判明し、")
    _log("  上記の予測式を訂正した。**T-D (W1) のみ有効な検証である。**")
    _log("  詳細は [[実験15 #6 修正理論の検証 Fisher 計量縮退の成立条件]] を参照。\n")

    t_all = time.perf_counter()

    # ==================================================================
    # T-A  (P1) のスケーリング
    # ==================================================================
    _log("── T-A  (P1) のスケーリング:  g_θθ ∝ ω1² かつ  g_θθ ∝ D_θ⁻²")
    _log("     [訂正] φ0 固定・ω0≠0 では  g_θθ = π²ω1²/(2(D_θ²+ω0²))。")
    _log("     ⟹ ω1 スケーリングの傾き = 2.0 は変わらないが、σ スケーリングは")
    _log("        −2.0 ではなく 0（D_θ ≪ ω0 ⟹ ω0 が screening）となるのが正解。")
    _log(f"      {'σ':>7} {'ω1':>7} {'g_θθ(実測)':>13} {'g_θθ(予測)':>12} "
         f"{'比':>10} {'床(ω1=0)':>11} {'過剰/予測':>9}")
    floor_of = {}
    for sigma in SIGMA_GRID:
        for omega1 in OMEGA1_GRID:
            gtt_meas, gtt_pred = [], []
            for s in SEEDS:
                x, y, _ = polar_sl(sigma, omega1, "fixed", s)
                r = np.hypot(x, y)
                th = np.mod(np.arctan2(y, x), 2 * np.pi)
                g_rr, g_tt, g_rt = fisher_polar(r, th)
                gtt_meas.append(g_tt)
                d_th = sigma / (R0 * R0)
                gtt_pred.append(math.pi ** 2 * omega1 ** 2
                                / (2.0 * (d_th ** 2 + OMEGA0 ** 2)))
                # T-A は相計量のみが対象なので、KDE-Fisher は走らせない
                # （実行時間が支配的になるため。λ_min は T-B で別途測定する）
                _rec(test="T-A", sigma=sigma, omega1=omega1, seed=s,
                     phi0="fixed", g_tt=g_tt, g_rr=g_rr, g_rt=g_rt,
                     g_tt_pred=gtt_pred[-1],
                     omega1_inferred=implied_omega1(g_tt, sigma))
            if omega1 == 0.0:
                floor_of[sigma] = float(np.mean(gtt_meas))
            m, p = np.mean(gtt_meas), np.mean(gtt_pred)
            exc = m - floor_of.get(sigma, 0.0)
            _log(f"      {sigma:>7.3f} {omega1:>7.3f} {m:>13.4g} {p:>12.4g} "
                 f"{(m / p if p > 0 else float('nan')):>10.1f} "
                 f"{floor_of.get(sigma, float('nan')):>11.3e} "
                 f"{(exc / p if p > 0 else float('nan')):>9.1f}")
    _log("     → 過剰分（床を引いた値）も予測の 20〜5000 倍 ⟹ (P1) は**判定不能**。")
    _log("       修正理论の線形領域上限 π²/2 = 4.93 に対し床が 3.4〜158 であり、")
    _log("       グリッド全体で 信号 < 床 が成り立つ（設計上の不成立）。\n")

    # 傾き推定
    _log("     スケーリング指数（片対数回帰、床上昇はアーティファクト）:")
    for sigma in SIGMA_GRID:
        xs, ys = [], []
        for r_ in ROWS:
            if r_.get("test") == "T-A" and r_["sigma"] == sigma and r_["omega1"] > 0:
                xs.append(r_["omega1"]); ys.append(r_["g_tt"])
        sl = loglog_slope(xs, ys)
        _log(f"       σ={sigma:>6.3f}:  g_θθ ∝ ω1^{sl:.3f}   （予測 2.000）")
    xs = [r_["sigma"] for r_ in ROWS if r_.get("test") == "T-A" and r_["omega1"] == 0.4]
    ys = [r_["g_tt"] for r_ in ROWS if r_.get("test") == "T-A" and r_["omega1"] == 0.4]
    _log(f"       ω1=0.4 :  g_θθ ∝ σ^{loglog_slope(xs, ys):.3f}   （予測 0.000）\n")

    # ==================================================================
    # T-B  (P1') 縮退到達性: ω1 = 0 で λ_min → 0 か
    # ==================================================================
    _log("── T-B  (P1') 縮退到達性: ω1 = 0 で計量縮退が起きるか")
    _log(f"      {'σ':>7} {'g_θθ(ω1=0)':>12} {'λ_min(ω1=0)':>13} "
         f"{'λ_min(ω1=0.4)':>15} {'比':>9}")
    for sigma in SIGMA_GRID:
        gtt0, gtt4, lm0, lm4 = [], [], [], []
        for s in SEEDS:
            for w1, ga, la in ((0.0, gtt0, lm0), (0.4, gtt4, lm4)):
                x, y, _ = polar_sl(sigma, w1, "fixed", s)
                r, th = np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)
                _, g_tt, _ = fisher_polar(r, th)
                ga.append(g_tt)
                lmin, _, _, _ = fisher_cartesian(x, y)
                la.append(lmin)
                _rec(test="T-B", sigma=sigma, omega1=w1, seed=s, phi0="fixed",
                     g_tt=g_tt, lam_min=lmin)
        _log(f"      {sigma:>7.3f} {np.mean(gtt0):>12.3e} {np.mean(lm0):>13.4g} "
             f"{np.mean(lm4):>15.4g} {np.mean(lm4)/max(np.mean(lm0),1e-300):>9.1f}")
    # --- ノイズ床の診断 -----------------------------------------------------
    # ω1=0 の系は厳密に回転対称（ρ = P(r)/2π）なので 真の g_θθ ≡ 0, λ_min ≡ 0。
    # したがって ω1=0 で測るものは「推定器の床」そのものである。
    # 角度相関時間 τ_c = 1/D_θ より、観測時間 T_OBS 中の独立サンプル数は
    #     n_eff = T_OBS · D_θ
    # 床は 1/sqrt(n_eff) でスケーリングするはず ⟹ 比 = sqrt(D_θ(high)/D_θ(low)).
    fl = {r_["sigma"]: r_["g_tt"] for r_ in ROWS
          if r_.get("test") == "T-B" and r_.get("omega1") == 0.0}
    _log("     ── ノイズ床の診断（ω1=0 では真の g_θθ ≡ 0 ⟹ 測れるのは床のみ）")
    _log(f"        {'σ':>7} {'n_eff=T·D_θ':>12} {'床 g_θθ':>11} {'床/予測':>9}")
    for sigma in SIGMA_GRID:
        ns = T_OBS * sigma
        _log(f"        {sigma:>7.3f} {ns:>12.1f} {fl.get(sigma, float('nan')):>11.3e} "
             f"{fl.get(sigma, float('nan')) * math.sqrt(ns):>9.3g}")
    sh, slo = max(SIGMA_GRID), min(SIGMA_GRID)
    pred_ratio = math.sqrt(sh / slo)
    obs_ratio = fl[sh] / max(fl[slo], 1e-300)
    _log(f"        床の比 (σ={sh} vs {slo}): 1/√n_eff の予測 {pred_ratio:.2f} / "
         f"実測 {obs_ratio:.2f}")
    _log("     → 床の σ 依存は n_eff = T_obs·D_θ で説明できる ⟹ **T-A/T-B/T-C の")
    _log("       測定値は信号ではなく推定器ノイズであり、(P1)(P1')(R1) は判定不能**。\n")

    # ==================================================================
    # T-C  (R1) 異方性の固定 vs 共動 ★改訂の核心
    # ==================================================================
    _log("── T-C  (R1) 異方性の固定 / 共動  ★改訂の核心")
    _log("     予測 [訂正後]:  φ0 固定   → g_θθ = π²ω1²/(2(D_θ²+ω0²))")
    _log("                     φ0=ω0 t   → g_θθ = 0（真に一様）")
    _log("     ※ 旧稿の g_θθ = ω1²/(2D_θ²) は φ0 固定・ω0=0 の特殊解であり、")
    _log("       ω0≠0 では p(θ)=e^{a cos θ} が定常解にならない（cos θ 項が打ち消し")
    _log("       合わない）。正しい一次解は  D_θ p₁'' − ω0 p₁' = (ω1/2π) cos θ  ⟹")
    _log("       変調振幅 ε = ω1/√(D_θ²+ω0²)、  g_θθ = π²ω1²/(2(D_θ²+ω0²)).")
    _log("       つまり **回転角速度 ω0 が異方性を遮蔽する**（D_θ ではなく ω0 でスケール）。")
    _log("       線形領域では g_θθ = π²ε²/2 ≲ π²/2 = 4.93 が上限。")
    _log(f"      {'σ':>7} {'ω1':>7} {'g_θθ 固定':>11} {'共動':>10} {'比':>9} "
         f"{'予測(固定)':>11} {'床/予測':>9}")
    for sigma in [0.20, 0.10, 0.05]:
        for omega1 in [0.10, 0.40]:
            fix, cor, lf, lc = [], [], [], []
            for s in SEEDS:
                for mode, acc, lacc in (("fixed", fix, lf), ("corotating", cor, lc)):
                    x, y, _ = polar_sl(sigma, omega1, mode, s)
                    r, th = np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)
                    _, g_tt, _ = fisher_polar(r, th)
                    acc.append(g_tt)
                    lmin, _, _, _ = fisher_cartesian(x, y)
                    lacc.append(lmin)
                    _rec(test="T-C", sigma=sigma, omega1=omega1, seed=s,
                         phi0=mode, g_tt=g_tt, lam_min=lmin)
            ratio = np.mean(cor) / max(np.mean(fix), 1e-300)
            # 訂正後の予測。共動臂は厳密に 0 なので固定臂のみ。
            d_th = sigma / R0 ** 2
            pred = math.pi ** 2 * omega1 ** 2 / (2 * (d_th ** 2 + OMEGA0 ** 2))
            _log(f"      {sigma:>7.3f} {omega1:>7.3f} {np.mean(fix):>11.3e} "
                 f"{np.mean(cor):>10.3e} {ratio:>9.2e} {pred:>11.3e} "
                 f"{np.mean(fix) / max(pred, 1e-300):>9.1f}")
    _log("     → 固定臂の測定値は訂正後の予測より 2〜4 桁大きく、かつ床と一致する。")
    _log("       共動/固定比 1e-2 は『真の縮退』ではなく、共動臂では有効独立")
    _log("       サンプル数が pattern 回転周期 (2π/ω0 = 1 s) で増えることによる")
    _log("       **床の低下**として説明できる。物理信号は両臂とも床に埋もれて")
    _log("       おり、本スクリプトは (R1) を検証できていない（判定不能）。")
    _log("       ★ただし『共動なら縮退・固定なら非縮退』という (R1) の構造自体を")
    _log("         否定する証拠にもならない。信号対雑比を上げた再設計が必要。\n")

    # ==================================================================
    # T-D  (W1) g = 1/w²
    # ==================================================================
    _log("── T-D  (W1) 詳細釣り合い系での幅の法則  g = 1/w²")
    _log("     注意: KDE の scott 帯域は h = n^{-1/6} w ≈ 0.164 w。平滑化後の幅は")
    _log("     sqrt(w²+h²) になるため g は 1/(1+0.164²) = 0.973 /w² と約 2.7% 下がる。")
    _log("     傾き -2 は影響を受けない（スケーリング則の検証が主目的）。")
    _log(f"      {'w':>8} {'g_xx(実測)':>13} {'1/w²(予測)':>12} {'比':>8} "
         f"{'λ_min':>13}")
    w_grid = [0.4, 0.2, 0.1, 0.05, 0.025]
    for w in w_grid:
        gs, ls, pr = [], [], []
        for s in SEEDS:
            d = gaussian2d(s, w)
            lmin, _, _, _ = fisher_cartesian(d[:, 0], d[:, 1])
            # 等方なので g_xx = λ の小さい方…ではなく g_xx を明示計算する
            data = d.T
            kde = gaussian_kde(data, bw_method="scott")
            pts = data[:, np.linspace(0, data.shape[1] - 1, 5000).astype(int)]
            delta, eps = 1e-4, 1e-10
            gr = []
            for i in range(2):
                pp, pm = pts.copy(), pts.copy()
                pp[i] += delta; pm[i] -= delta
                gr.append((np.log(np.clip(kde(pp), eps, None))
                           - np.log(np.clip(kde(pm), eps, None))) / (2 * delta))
            gr = np.array(gr)
            gxx = float(np.sum(gr[0] ** 2) / gr.shape[1])
            gs.append(gxx); ls.append(lmin); pr.append(1.0 / w ** 2)
            _rec(test="T-D", w=w, seed=s, g_xx=gxx, lam_min=lmin, g_xx_pred=pr[-1])
        _log(f"      {w:>8.3f} {np.mean(gs):>13.4g} {np.mean(pr):>12.4g} "
             f"{np.mean(gs)/np.mean(pr):>8.3f} {np.mean(ls):>13.4g}")
    xs = [r_["w"] for r_ in ROWS if r_.get("test") == "T-D"]
    ys = [r_["g_xx"] for r_ in ROWS if r_.get("test") == "T-D"]
    _log(f"     スケーリング指数: g ∝ w^{loglog_slope(xs, ys):.3f}   （予測 -2.000）\n")

    # ==================================================================
    # CSV
    # ==================================================================
    out = "theory_verification15_result.csv"
    keys = sorted({k for r in ROWS for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ROWS:
            w.writerow({k: r.get(k, "") for k in keys})
    _log(f"書き出し: {out}（{len(ROWS)} 行）")
    _log(f"所要時間: {time.perf_counter()-t_all:.1f}s")
    _log("DONE")
