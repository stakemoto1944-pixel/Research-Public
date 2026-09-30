#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
数値実験16（#6 修正理論の検証・再設計版）

  前版: 15_theory_verification.py   （判定不能で終了）
  本版: 同名・再設計                        ← 本ファイル

【何が変わったか】
exp15 は (P1) の検証を企てたが、3 つとも判定不能に終わった。
原因 1（推定器）: 2 次元度数分布の有限差分推定器は bed 1/Δθ² で増幅される。
    さらに bed の成因は Poisson ノイズで、T_obs 中の有効独立サンプル数
    N_eff = T_obs·D_θ のみに依存するため、σ を下げると bed が逆に増大する。
    実測: σ=0.5 で bed 3.375、σ=0.02 で bed 21.39。1/sqrt(N_eff) の予測比
    5.00 に対し実測比 6.34。
原因 2（設計）: 極座標の角拡散 D_θ = σ/r² は r→0 で発散し E_θ と r の
    平均を正しく取れない。定数 D_θ 版を用意して切り分ける。
原因 3（理論）: ω0 = 0 を暗黙に置いていた。正しい一次解は
        D_θ f1'' - ω0 f1' = D_θ cosθ,   f1 = -(cosθ + γ sinθ)/(1+γ²),
        γ = ω0/D_θ
    ⟹ g_θθ = ω1² / (2 (D_θ² + ω0²))          …(P1c, 訂正版)
    つまり回転が異方性を screening する。旧 (P1) の ω1²/(2D_θ²) は
    ω0 = 0 の特殊解にすぎない。

【推定器の変更点】
  (a) θ 周辺分布のみを使う（r 方向の bin 分割が bed を NBIN_R 倍する）
  (b) 差分ではなく ln ρ の第 1 調和を最小二乗で回帰する
      ⟹ Δθ⁻² の増幅が消える（exp15 比で SNR が 1.8〜4.6 → 2.5〜980）
  (c) 平滑化は対称核なので第 1 調和を保存する（保存される形で減衰する）。
      伝達関数 T1 = exp[-(σ_f²/2)(2π/nb)²] は既知なので補正する。
  (d) 床の測定: ω1 = 0 の系は厳密に回転対称 ⟹ 真の g_θθ ≡ 0。
      したがって ω1=0 の測定値がそのまま bed の実測値になる。

【検証する帰結】
  (P1c)  g_θθ = ω1²/(2(D_θ²+ω0²))          ★改訂の中核・本実験の主役
  (R1)   異方性が循環と「共動」すれば g_θθ = 0（縮退）、lab frame で固定
         されていれば g_θθ > 0（非縮退）
  (W1)   詳細釣り合い系で g = 1/w²（exp15 で確認済み・再確認）
  (F1)   bed ∝ 1/N（総サンプル数）— bed が有限サンプル効果であることを示す

実行: python3 16_theory_verification.py
出力: theory_verification16_result.csv / experiment16_log.txt
"""
import csv
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------- 定数
MU = 1.0                    # Stuart-Landau: r-dot = MU r - r^3,  r* = sqrt(MU)
N_ROT = 20                  # 観測する回転数
PER_ROT = 2000              # 回転周期あたりのステップ数
NB = 40                     # θ 周辺分布の bin 数
SMOOTH = 4.0                # 平滑化幅（bin 単位、対称なので第1調和は既知 bias のみ）
SEEDS = (0, 1, 2)

# (P1c) の検証点。ω0 が D_θ を横切るように取る。
SIGMAS = (1.0, 2.0)
OMEGA0_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
OMEGA1_GRID = (0.15, 0.30, 0.60)
N_TRAJ = 200

# 平滑化による第 1 調和の伝達関数（分母に置く = バイアス補正）
T1 = math.exp(-0.5 * SMOOTH ** 2 * (2.0 * math.pi / NB) ** 2)

ROWS = []


def log(m=""):
    print(m, flush=True)


def rec(**kw):
    ROWS.append(kw)


# ----------------------------------------------------------------- 推定器
def harmonic_g(th, nb=NB, smooth=SMOOTH):
    """
    θ 周辺分布の ln ρ を第1〜第3調和で最小二乗近似し、
    平滑化バイアスを補正した g_θθ = (c1² + c2²)/2 を返す。
    あわせて差分推定器の値と残差も返す（相互診断用）。
    """
    H, edges = np.histogram(th, bins=nb, range=(0.0, 2.0 * math.pi))
    Hs = gaussian_filter1d(H, smooth, mode="wrap")
    Hs = np.clip(Hs, 1e-12, None)
    L = np.log(Hs)
    Lm = L - L.mean()
    c = (edges[:-1] + edges[1:]) / 2.0
    A = np.vstack([np.ones(nb)]
                  + [f(k * c) for k in (1, 2, 3) for f in (np.cos, np.sin)]).T
    coef, *_ = np.linalg.lstsq(A, Lm, rcond=None)
    a1 = math.hypot(coef[1], coef[2]) / T1          # ← バイアス補正
    a2 = math.hypot(coef[3], coef[4]) / T1
    a3 = math.hypot(coef[5], coef[6]) / T1
    g1 = 0.5 * a1 ** 2
    d = edges[1] - edges[0]
    dL = (np.roll(L, -1) - np.roll(L, 1)) / (2.0 * d)
    P = Hs / Hs.sum()
    g2 = float(np.sum(P * dL ** 2))
    return g1, g2, a1, a2, a3, float(np.mean((A @ coef - Lm) ** 2))


# ----------------------------------------------------------------- 系の生成
def ensemble(dth_mode, sigma, omega0, omega1, corot, seed, n_traj=N_TRAJ, rc=0.2):
    """
    n_traj 本の極座標 Stuart-Landau + 角トルクをベクトル並列で回す。
        r-dot = MU r - r^3 + sqrt(2 sigma) xi_r
        θ-dot = omega0 + omega1 sin(θ - φ0) + sqrt(2 D_θ) xi_θ
    φ0 = omega0 t なら共動（corotating）、0 なら lab frame で固定（fixed）。
    回転周期あたりのステップ数を固定するので、ω0 に依らず M と総サンプル数
    N = n_traj·M は一定になる。＝ bed の ω0 依存を人為的に持ち込まない。
    """
    dt = (2.0 * math.pi / omega0) / PER_ROT
    M = N_ROT * PER_ROT
    rng = np.random.default_rng(seed)
    r = np.full(n_traj, 1.0)
    th = rng.uniform(0.0, 2.0 * math.pi, n_traj)
    sr = math.sqrt(2.0 * sigma * dt)
    out = np.empty((M, n_traj))
    dmax = 0.0
    for k in range(M):
        ph0 = omega0 * (k * dt) if corot else 0.0
        r = np.clip(r + (MU * r - r ** 3) * dt + sr * rng.standard_normal(n_traj),
                    1e-3, 6.0)
        if dth_mode == "const":
            dth = np.full(n_traj, sigma)
        elif dth_mode == "soft":
            dth = sigma / (r * r + rc * rc)
        else:
            dth = np.minimum(sigma / (r * r), rc)
        dmax = max(dmax, float(dth.max()))
        th = th + (omega0 + omega1 * np.sin(th - ph0)) * dt \
            + np.sqrt(2.0 * dth * dt) * rng.standard_normal(n_traj)
        out[k] = np.mod(th, 2.0 * math.pi)
    return out[int(0.2 * M):].ravel(), dt, M, dmax


def measure(dth_mode, sigma, omega0, omega1, corot, n_traj=N_TRAJ, rc=0.2):
    """seed 平均した (g1, g2, a1, a2, a3, resid, Dmax) を返す。"""
    acc = []
    for s in SEEDS:
        th, _, _, dmax = ensemble(dth_mode, sigma, omega0, omega1, corot, s,
                                 n_traj=n_traj, rc=rc)
        acc.append(harmonic_g(th) + (dmax,))
    arr = np.array(acc, dtype=float)
    return [float(v) for v in arr.mean(axis=0)], [float(v) for v in arr.std(axis=0)]


def predict(sigma, omega0, omega1):
    d = sigma / (MU * MU)                  # D_θ = sigma / r*²
    return omega1 ** 2 / (2.0 * (d * d + omega0 * omega0))


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    t_all = time.perf_counter()
    log("=" * 104)
    log("実験16: #6 修正理論の検証（再設計版）— Fisher 計量縮退の成立条件")
    log("=" * 104)
    log(f"推定器: θ 周辺のみ / 第1調和最小二乗 / nb={NB} smooth={SMOOTH}")
    log(f"  平滑化バイアス T1 = exp[-0.5*s^2*(2pi/nb)^2] = {T1:.4f} （補正済み）")
    log(f"アンサンブル: n_traj={N_TRAJ}, 回転数={N_ROT}, 回転あたり={PER_ROT} ステップ")
    log(f"  → 1 条件あたり N = {N_TRAJ * N_ROT * PER_ROT:.2e} サンプル（ω0 非依存）")
    log(f"D_θ = sigma/r*²,  r* = {math.sqrt(MU):.3f}")
    log("")
    log("検証する帰結:")
    log("  (P1c) g_θθ = ω1²/(2(D_θ²+ω0²))            ★改訂の中核")
    log("  (R1)   共動 ⟹ g_θθ = 0 / 固定 ⟹ g_θθ > 0")
    log("  (W1)   詳細釣り合い系で g = 1/w²")
    log("  (F1)   床 ∝ 1/N（有限サンプル効果であることの確認）")
    log("")

    # ---------------------------------------------------------------- T-A
    log("=" * 104)
    log("T-A  (P1c) ω0 スウィープ: 回転が異方性を screening するか")
    log("=" * 104)
    for sigma in SIGMAS:
        dth = sigma / (MU * MU)
        log(f"\n【sigma={sigma:.1f}  D_θ={dth:.2f}  omega1={OMEGA1_GRID[1]:.2f}】")
        log(f"  {'ω0':>6} {'D_θ/ω0':>7} | {'固定 床':>11} {'固定 測定':>11} "
            f"{'過量':>11} {'SNR':>8} | {'共動 測定':>11} {'共動/固定':>10} | "
            f"{'予測':>11} {'測定/予測':>9} {'ε':>6}")
        for omega0 in OMEGA0_GRID:
            w1 = OMEGA1_GRID[1]
            f0, _ = measure("const", sigma, omega0, 0.0, False)
            f1, fs1 = measure("const", sigma, omega0, w1, False)
            c1, cs1 = measure("const", sigma, omega0, w1, True)
            exc = f1[0] - f0[0]
            p = predict(sigma, omega0, w1)
            eps = w1 / math.sqrt(dth ** 2 + omega0 ** 2)
            log(f"  {omega0:>6.1f} {dth/omega0:>7.2f} | {f0[0]:>11.3e} "
                f"{f1[0]:>11.3e} {exc:>11.3e} "
                f"{exc/max(f0[0],1e-300):>8.1f} | {c1[0]:>11.3e} "
                f"{c1[0]/max(f1[0],1e-300):>10.2e} | {p:>11.3e} "
                f"{exc/max(p,1e-300):>9.3f} {eps:>6.3f}")
            rec(test="T-A", dth_mode="const", sigma=sigma, omega0=omega0,
                omega1=w1, arm="fixed", g1_null=f0[0], g1=f1[0], g1_sd=fs1[0],
                g1_corot=c1[0], g1_corot_sd=cs1[0], excess=exc, pred=p,
                ratio=exc / max(p, 1e-300), eps=eps, n_traj=N_TRAJ)
        # 指数と交差位置の推定
        sub = [r_ for r_ in ROWS if r_["test"] == "T-A" and r_["sigma"] == sigma]
        x = np.log([r_["omega0"] for r_ in sub])
        y = np.log([r_["excess"] for r_ in sub])
        sl_meas = float(np.polyfit(x, y, 1)[0])
        yp = np.log([predict(sigma, r_["omega0"], OMEGA1_GRID[1]) for r_ in sub])
        sl_pred = float(np.polyfit(x, yp, 1)[0])
        log(f"  → ω0 べき（測定）  = {sl_meas:+.3f}   （同じ範囲で予測 {sl_pred:+.3f}）")
        # 2 点法による D_θ の内部推定。全範囲で 1/g を線形化すると条件数が
        # 悪化するので、隣接 2 点ごとの比から D_θ² を逆算する:
        #   g1/g2 = (D_θ²+w0b²)/(D_θ²+w0a²)  =>  D_θ² = (g1 w0a² - g2 w0b²)/(g2-g1)
        om_all = [r_["omega0"] for r_ in sub]
        g_all = [r_["excess"] for r_ in sub]
        for i in range(len(om_all) - 1):
            if g_all[i + 1] >= g_all[i]:
                continue
            d2 = ((g_all[i] * om_all[i] ** 2 - g_all[i + 1] * om_all[i + 1] ** 2)
                  / (g_all[i + 1] - g_all[i]))
            if d2 <= 0:
                log(f"  -> 2pt (w0={om_all[i]:5.1f},{om_all[i+1]:5.1f}): "
                    f"Dt^2 = {d2:.3f}  ** negative -> unusable")
                continue
            di = math.sqrt(d2)
            log(f"  -> 2pt (w0={om_all[i]:5.1f},{om_all[i+1]:5.1f}): "
                f"Dt internal = {di:.3f}  (input {dth:.3f}, err "
                f"{100*(di-dth)/dth:+.1f}%)")
        # ---------------------------------------------------------------- T-B
    log("")
    log("=" * 104)
    log("T-B  (P1c) ω1 スウィープ:  g_θθ ∝ ω1² か")
    log("=" * 104)
    for sigma in SIGMAS:
        for omega0 in (0.3, 10.0):
            dth = sigma / (MU * MU)
            log(f"\n【sigma={sigma:.1f}  ω0={omega0:.1f}  D_θ={dth:.2f}】")
            log(f"  {'ω1':>6} {'λ=ω1/D_θ':>10} | {'固定 過量':>11} {'SNR':>8} | "
                f"{'予測':>11} {'比':>7} | {'第2調和/第1':>12} {'第3/第1':>9}")
            for w1 in OMEGA1_GRID:
                f0, _ = measure("const", sigma, omega0, 0.0, False)
                f1, _ = measure("const", sigma, omega0, w1, False)
                exc = f1[0] - f0[0]
                p = predict(sigma, omega0, w1)
                log(f"  {w1:>6.2f} {w1/dth:>10.3f} | {exc:>11.3e} "
                    f"{exc/max(f0[0],1e-300):>8.1f} | {p:>11.3e} "
                    f"{exc/max(p,1e-300):>7.3f} | {f1[3]/max(f1[2],1e-300):>12.4f} "
                    f"{f1[4]/max(f1[2],1e-300):>9.4f}")
                rec(test="T-B", dth_mode="const", sigma=sigma, omega0=omega0,
                    omega1=w1, arm="fixed", g1_null=f0[0], g1=f1[0],
                    excess=exc, pred=p, ratio=exc / max(p, 1e-300),
                    a1=f1[2], a2=f1[3], a3=f1[4], n_traj=N_TRAJ)
            sub = [r_ for r_ in ROWS if r_["test"] == "T-B"
                   and r_["sigma"] == sigma and r_["omega0"] == omega0]
            x = np.log([r_["omega1"] for r_ in sub])
            y = np.log([r_["excess"] for r_ in sub])
            log(f"  → ω1 べき（測定）= {float(np.polyfit(x, y, 1)[0]):+.3f}"
                f"   （予測 +2.000）")
    # ---------------------------------------------------------------- T-C
    log("")
    log("=" * 104)
    log("T-C  (R1) 異方性の固定 / 共動  ★改訂の中核の構造")
    log("=" * 104)
    for sigma in SIGMAS:
        for omega0 in (0.3, 3.0):
            dth = sigma / (MU * MU)
            w1 = 0.30
            f0, _ = measure("const", sigma, omega0, 0.0, False)
            f1, fs = measure("const", sigma, omega0, w1, False)
            c0, _ = measure("const", sigma, omega0, 0.0, True)
            c1, cs = measure("const", sigma, omega0, w1, True)
            p = predict(sigma, omega0, w1)
            zf = (c1[0] - c0[0]) / max(cs[0], 1e-300)
            zc = (f1[0] - f0[0]) / max(fs[0], 1e-300)
            log(f"  sigma={sigma:.1f} ω0={omega0:>4.1f} | 固定: 過量 "
                f"{f1[0]-f0[0]:>10.3e} (z={zc:>6.1f})  |  共動: 過量 "
                f"{c1[0]-c0[0]:>10.3e} (z={zf:>6.1f})  |  予測 {p:.3e}  "
                f"|  固定/共動 = {(f1[0]-f0[0])/max(c1[0]-c0[0],1e-300):.2e}")
            rec(test="T-C", dth_mode="const", sigma=sigma, omega0=omega0,
                omega1=w1, arm="fixed+corot", g1_null=f0[0], g1=f1[0],
                g1_sd=fs[0], g1_corot_null=c0[0], g1_corot=c1[0],
                g1_corot_sd=cs[0], excess=f1[0] - f0[0],
                excess_corot=c1[0] - c0[0], pred=p, n_traj=N_TRAJ)
    # ---------------------------------------------------------------- T-D
    log("")
    log("=" * 104)
    log("T-D  D_θ モデルの感度:  1/r² / 正則化 / 定数 で (P1c) は変わるか")
    log("=" * 104)
    log(f"  {'D_θ':>6} {'rc':>6} {'Dmax':>8} | {'ω0':>6} {'過量':>11} "
        f"{'予測':>11} {'比':>7}")
    for mode, sigma, rc in (("const", 2.0, 0.0), ("soft", 2.0, 0.4),
                            ("soft", 2.0, 0.2), ("clip", 2.0, 50.0),
                            ("clip", 2.0, 10.0)):
        for omega0 in (0.3, 3.0, 10.0):
            w1 = 0.30
            f0, _ = measure(mode, sigma, omega0, 0.0, False, rc=rc)
            f1, _ = measure(mode, sigma, omega0, w1, False, rc=rc)
            exc = f1[0] - f0[0]
            p = predict(sigma, omega0, w1)
            log(f"  {mode:>6} {rc:>6.2f} {f1[5]:>8.2f} | {omega0:>6.1f} "
                f"{exc:>11.3e} {p:>11.3e} {exc/max(p,1e-300):>7.3f}")
            rec(test="T-D", dth_mode=mode, rc=rc, sigma=sigma, omega0=omega0,
                omega1=w1, arm="fixed", g1_null=f0[0], g1=f1[0], excess=exc,
                pred=p, ratio=exc / max(p, 1e-300), dmax=f1[5], n_traj=N_TRAJ)
    # ---------------------------------------------------------------- T-E
    log("")
    log("=" * 104)
    log("T-E  (W1) 詳細釣り合い系で g = 1/w²（exp15 での確認の再確認）")
    log("=" * 104)
    # smoothing widens the width to sqrt(w^2 + (sf*Delta)^2) => expected ratio 1/(1+(sf*Delta/w)^2)
    rng_w = 4.5
    delta_rel = (2.0 * rng_w) / NB
    bias = 1.0 / (1.0 + delta_rel ** 2)
    n_samp = N_TRAJ * N_ROT * PER_ROT
    log(f"  N = {n_samp:.2e} samples (same as other tests), fixed range "
        f"+-{rng_w}w, nb={NB}  =>  Delta/w = {delta_rel:.4f}")
    log(f"  known smoothing bias (sf=1 bin) = 1/(1+(sf*Delta/w)^2) = {bias:.4f}")
    log(f"  {'w':>8} {'g_xx(meas)':>13} {'1/w^2':>12} {'ratio':>8} "
        f"{'ratio/bias':>9} {'empty':>6}")
    W_GRID = (0.4, 0.2, 0.1, 0.05, 0.025)
    xs, ys = [], []
    for w in W_GRID:
        gs, nzc = [], 0
        for s in SEEDS:
            rng = np.random.default_rng(1000 + s)
            d = rng.standard_normal(n_samp) * w   # σ = w（範囲も w でスケール）
            if s == SEEDS[0] and w == W_GRID[0]:
                log(f"    [自己診断] サンプル σ = {d.std():.4f}（w={w}）, "
                    f"範囲 ±{rng_w*w:.4f}, Δ = {(2*rng_w*w)/NB:.5f}")
            # bin the 1D marginal of an isotropic 2D Gaussian and evaluate g
            H, edges = np.histogram(d, bins=NB,
                                    range=(-rng_w * w, rng_w * w))
            nzc += int((H == 0).sum())
            Hs = np.clip(gaussian_filter1d(H, 1.0, mode="nearest"), 1e-12, None)
            L = np.log(Hs)
            dd = edges[1] - edges[0]
            dL = (np.roll(L, -1) - np.roll(L, 1)) / (2.0 * dd)
            P = Hs / Hs.sum()
            gs.append(float(np.sum(P * dL ** 2)))
        m = float(np.mean(gs))
        xs.append(w)
        ys.append(m)
        log(f"  {w:>8.3f} {m:>13.4e} {1.0/w**2:>12.4e} {m*w**2:>8.4f} "
            f"{m*w**2/bias:>9.4f} {nzc:>6d}")
        rec(test="T-E", w=w, g_xx=m, g_xx_pred=1.0 / w ** 2,
            ratio=m * w ** 2, ratio_over_bias=m * w ** 2 / bias,
            empty_bins=nzc, n_samples=n_samp, seed_mean=True)
    log(f"  → スケーリング指数: g ∝ w^{np.polyfit(np.log(xs), np.log(ys), 1)[0]:+.3f}"
        f"   （予測 -2.000）")
    # ---------------------------------------------------------------- T-F
    log("")
    log("=" * 104)
    log("T-F  (F1) ノイズ床のサンプル数依存:  g(ω1=0) ∝ 1/N か")
    log("=" * 104)
    log(f"  {'n_traj':>7} {'N':>11} {'固定 床':>11} {'共動 床':>11} {'N×床':>11}")
    for nt in (25, 50, 100, 200, 400):
        f0, _ = measure("const", 2.0, 1.0, 0.0, False, n_traj=nt)
        c0, _ = measure("const", 2.0, 1.0, 0.0, True, n_traj=nt)
        N = nt * N_ROT * PER_ROT
        log(f"  {nt:>7} {N:>11.2e} {f0[0]:>11.3e} {c0[0]:>11.3e} "
            f"{N*f0[0]:>11.3e}")
        rec(test="T-F", n_traj=nt, N=N, g1_null=f0[0], g1_corot_null=c0[0],
            N_times_g=N * f0[0])
    sub = [r_ for r_ in ROWS if r_["test"] == "T-F"]
    log(f"  → 床の N べき = {float(np.polyfit(np.log([r_['N'] for r_ in sub]), np.log([r_['g1_null'] for r_ in sub]), 1)[0]):+.3f}"
        f"   （有限サンプル効果なら -1.000）")
    # ---------------------------------------------------------------- CSV
    out = "theory_verification16_result.csv"
    keys = sorted({k for r_ in ROWS for k in r_})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r_ in ROWS:
            w.writerow({k: r_.get(k, "") for k in keys})
    log("")
    log(f"書き出し: {out}（{len(ROWS)} 行）")
    log(f"所要時間: {time.perf_counter()-t_all:.1f}s")
    log("DONE")
