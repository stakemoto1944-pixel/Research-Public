#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（８）：別境界軸（D→0, c_mem）と位相変数（r,θ）再パラメータ化での計量縮退検証

背景: 実験4–7でγ/cs軸・高次元PCA部分多様体のいずれでも「相境界で det g→0 ∧ λ_min→0 ∧ κ→∞」
の予言は棄却された。残る検証は:
  ① D→0 極限（雑音消滅で密度が決定的アトラクタへ集中）
  ② 記憶強度 c_mem 方向（別の境界軸・循環の発現境界）
  ③ 位相変数 (r, θ) への再パラメータ化（論文の自然座標 = 振幅・位相）

特に③は重要: 循環相（NEIP）では位相が一様に回転し ρ(r,θ)=ρ(r) となるため、
位相方向のFisher計量 g_θθ = E[(∂_θ lnρ)²] が理論的にゼロになる。
これは「真の"位相ゼロモード"」であり、境界特異でなく座標由来（ゲージ方向）であるか、
それとも境界で特異に現れるかを定量する。

方法:
  - (x,y) = ペア(0,1)。カルテシアンKDEで ∂_x lnρ, ∂_y lnρ を計算し、
    連鎖律で極座標スコアを構成（r=0 の特異性を回避）:
      ∂_r = (x ∂_x + y ∂_y),   ∂_θ = x ∂_y − y ∂_x
    → g_rr, g_rθ, g_θθ, det, κ を評価。
  - 位相の一様性: 巻き数レート ω̄ = ⟨dθ/dt⟩（unwrap）、circular concentration R1 = |⟨e^{iθ}⟩|。
  - マッチドガウス null: 同一ペア共分散の等方性ガウス標本で同じパイプライン → g_θθ の推定器 null 値。
  - カルテシアン計量（λ_min, det, κ）も併記。ROIマーカー |I| も測定。

実行: python3 experiment8_fixed_axes_phase.py
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import csd

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
N_EVAL = 2000


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42, asym_gain=1.0):
    np.random.seed(seed)
    steps = int(T / DT)
    J_raw = np.random.randn(N, N) / np.sqrt(N)
    J_sym = coupling_scale * 0.5 * (J_raw + J_raw.T)
    np.fill_diagonal(J_sym, 0.0)
    J_raw = np.random.randn(N, N) / np.sqrt(N)
    J_asym = coupling_scale * 0.5 * (J_raw - J_raw.T) * asym_gain
    psi = np.random.randn(N) * 0.01
    Q = np.zeros(N)
    subsample = int(round(SAVE_DT / DT))
    saved_steps = steps // subsample
    history = np.zeros((saved_steps, N))
    saved_time = np.linspace(0, T, saved_steps)
    save_idx = 0
    diverged = False
    clip_count = 0
    for t in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        clip_count += int(np.any(np.abs(psi) >= CLIP_BOUND))
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], diverged, clip_count


def quick_I(x, y):
    """ペア(0,1)のCSD虚部の帯域積分（|I| ROIマーカー、0.05–1.5 Hz帯）。"""
    fs = 1.0 / SAVE_DT
    nperseg = min(256, len(x) // 4)
    if nperseg < 8:
        return 0.0
    f, Pxy = csd(x, y, fs=fs, nperseg=nperseg)
    m = (f >= 0.05) & (f <= 1.5)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(np.imag(Pxy[m]), f[m]))


def cartesian_scores(xy, n_eval=N_EVAL):
    """(n,2) 実軌道 → カルテシアンKDEのスコア ∂_x lnρ, ∂_y lnρ（評価点はデータ間引き）。"""
    data = xy.T  # (2, n)
    n = data.shape[1]
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    pts = data[:, idx]
    ok = False
    try:
        kde = gaussian_kde(data, bw_method='scott')
        eps = 1e-10
        delta = 1e-4
        gx = np.zeros(n_eval); gy = np.zeros(n_eval)
        pp = pts.copy(); pp[0, :] += delta
        pm = pts.copy(); pm[0, :] -= delta
        gx = (np.log(np.clip(kde(pp), eps, None)) - np.log(np.clip(kde(pm), eps, None))) / (2 * delta)
        pp = pts.copy(); pp[1, :] += delta
        pm = pts.copy(); pm[1, :] -= delta
        gy = (np.log(np.clip(kde(pp), eps, None)) - np.log(np.clip(kde(pm), eps, None))) / (2 * delta)
        ok = True
    except np.linalg.LinAlgError:
        return None
    return pts, gx, gy, idx if ok else None


def full_metrics(xy, seed_g=7, gauss_null=True):
    """ペア軌道(n,2) → カルテシアン計量 + 極座標計量 + 位相診断。"""
    x = xy[:, 0]; y = xy[:, 1]
    res = {}
    # --- 位相診断 ---
    th = np.arctan2(y, x)
    th_u = np.unwrap(th)
    dth = np.diff(th_u)
    res['omega'] = float(np.mean(dth))  # rad per sample (τ=0.01)
    res['R1'] = float(np.abs(np.mean(np.exp(1j * th))))
    res['phase_std'] = float(np.std(np.arctan2(np.sin(th), np.cos(th))))
    res['clip'] = 0
    # --- カルテシアンKDEスコア ---
    out = cartesian_scores(xy)
    if out is None:
        return None
    pts, gx, gy, idx = out
    rp = np.sqrt(pts[0] ** 2 + pts[1] ** 2)
    # 極座標スコア（連鎖律）: ∂_r = e_r·∇, ∂_θ = r²·e_θ·∇ の単位量の代わりに r のままで:
    #   正規のパラメータ(r,θ)に対するスコアは
    #   ∂_θ lnρ = −y ∂_x + x ∂_y  （微分作用素）
    #   ∂_r lnρ = (x ∂_x + y ∂_y)/r  （単位ベクトル成分 → r=0 でも発散しない）
    sr = (pts[0] * gx + pts[1] * gy) / np.maximum(rp, 1e-8)
    st = (-pts[1] * gx + pts[0] * gy)
    # --- 極座標計量 ---
    g_rr = float(np.mean(sr * sr)); g_rt = float(np.mean(sr * st)); g_tt = float(np.mean(st * st))
    Gp = np.array([[g_rr, g_rt], [g_rt, g_tt]])
    evp = np.linalg.eigvalsh(Gp)
    res['g_rr'] = g_rr; res['g_rt'] = g_rt; res['g_tt'] = g_tt
    res['det_polar'] = float(evp[0] * evp[1])
    res['lam_min_polar'] = float(evp[0])
    res['kappa_polar'] = float(evp[-1] / max(evp[0], 1e-300))
    # --- カルテシアン計量 ---
    G = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    ev = np.linalg.eigvalsh(G)
    res['det'] = float(np.prod(ev))
    res['lam_min'] = float(ev[0])
    res['kappa'] = float(ev[-1] / max(ev[0], 1e-300))
    res['sigma_pair'] = float(np.sqrt(max(0.0, np.trace(np.cov(xy.T)) / 2.0)))
    res['I'] = quick_I(x, y)
    # --- マッチドガウス null（極座標 g_θθ の推定器 baseline）---
    if gauss_null:
        cov = np.cov(xy.T)
        rng = np.random.default_rng(seed_g)
        gs = rng.multivariate_normal(np.zeros(2), cov, size=max(len(x), 20000))
        outg = cartesian_scores(gs)
        if outg is not None:
            pg, ggx, ggy, _ = outg
            rg2 = np.sqrt(pg[0] ** 2 + pg[1] ** 2)
            sg = (-pg[1] * ggx + pg[0] * ggy)
            res['g_tt_G'] = float(np.mean(sg * sg))
            res['R1_G'] = float(np.abs(np.mean(np.exp(1j * np.arctan2(pg[1], pg[0])))))
        else:
            res['g_tt_G'] = float('nan'); res['R1_G'] = float('nan')
    return res


def print_row(label, r):
    if r is None:
        print("  %-28s | 発散" % label, flush=True)
        return
    f = "  %-28s | λm=%8.5f det=%9.4g κ=%6.2f | g_θθ=%10.5f g_rr=%9.3f det_p=%9.4g κ_p=%8.2f | ω̄=%+.4f R1=%.4f |g_tt/G=%.2f |I|=%.3f σ=%5.2f"
    gt_rel = r['g_tt'] / max(r.get('g_tt_G', float('nan')), 1e-12)
    print(f % (label, r['lam_min'], r['det'], r['kappa'], r['g_tt'], r['g_rr'],
               r['det_polar'], r['kappa_polar'], r['omega'], r['R1'],
               gt_rel if np.isfinite(gt_rel) else float('nan'), r['I'], r['sigma_pair']), flush=True)


def main():
    print("=== 実験8: 別境界軸（D→0, c_mem）と位相変数（r,θ）の計量縮退検証 ===", flush=True)
    print("基準: c_mem=5.0, D=0.05, T=300, burn-in=100, J_asymあり。ペア(0,1)。", flush=True)
    rows = []

    # ========== Axis A: D → 0 極限 ==========
    print("\n" + "=" * 112, flush=True)
    print("A. D→0 極限（密度の決定的アトラクタへの集中）| γ∈{0.1 (ROI), 0.2 (境界), 1.0 (EQ)}", flush=True)
    print("=" * 112, flush=True)
    for g in (0.1, 0.2, 1.0):
        for D in (0.05, 0.02, 0.01, 0.005):
            tt, tr, div, clip = simulate(1.0, D, g, 5.0, 300.0, 100.0, seed=42)
            if div:
                print_row("γ=%.1f D=%.3f" % (g, D), None)
                continue
            r = full_metrics(tr[:, 0:2])
            if r is not None:
                r.update(gamma=g, D=D, c_mem=5.0, seed=42, axis='A')
                rows.append(r)
            print_row("γ=%.2f D=%.3f (clip=%d)" % (g, D, clip), r)
    # シード100 の確認（ROI 最小D）
    for g, D in ((0.1, 0.005), (0.2, 0.005)):
        tt, tr, div, clip = simulate(1.0, D, g, 5.0, 300.0, 100.0, seed=100)
        r = full_metrics(tr[:, 0:2]) if not div else None
        if r is not None:
            r.update(gamma=g, D=D, c_mem=5.0, seed=100, axis='A')
            rows.append(r)
        print_row("γ=%.2f D=%.3f seed100" % (g, D), r)

    # ========== Axis B: c_mem 方向（記憶強度）==========
    print("\n" + "=" * 112, flush=True)
    print("B. c_mem 掃引（γ=0.1, D=0.05）: 記憶強度の弱化 → 循環の発現/消失境界", flush=True)
    print("=" * 112, flush=True)
    for cm in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        tt, tr, div, clip = simulate(1.0, 0.05, 0.1, cm, 300.0, 100.0, seed=42)
        r = full_metrics(tr[:, 0:2]) if not div else None
        if r is not None:
            r.update(gamma=0.1, D=0.05, c_mem=cm, seed=42, axis='B')
            rows.append(r)
        print_row("c_mem=%5.2f (clip=%d)" % (cm, clip), r)

    # ========== Axis C: 位相変数 (r,θ) 再パラメータ化の γ 依存 ==========
    print("\n" + "=" * 112, flush=True)
    print("C. 位相変数再パラメータ化: γ掃引（D=0.05, c_mem=5, 2シード）", flush=True)
    print("=" * 112, flush=True)
    for g in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        for s in (42, 100):
            tt, tr, div, clip = simulate(1.0, 0.05, g, 5.0, 300.0, 100.0, seed=s)
            r = full_metrics(tr[:, 0:2]) if not div else None
            if r is not None:
                r.update(gamma=g, D=0.05, c_mem=5.0, seed=s, axis='C')
                rows.append(r)
            print_row("γ=%.2f seed=%d (clip=%d)" % (g, s, clip), r)

    # ========== 判定 ==========
    print("\n" + "=" * 112, flush=True)
    print("D. 判定", flush=True)
    print("=" * 112, flush=True)
    from collections import defaultdict
    by_g = defaultdict(list)
    for r in rows:
        if r.get('axis') == 'C' and abs(r['D'] - 0.05) < 1e-9:
            by_g[r['gamma']].append(r)
    for g in sorted(by_g):
        rs = by_g[g]
        gtt = np.array([r['g_tt'] for r in rs]); gttG = np.array([r['g_tt_G'] for r in rs])
        ratio = np.mean(gtt) / max(np.mean(gttG), 1e-300)
        print("  γ=%.2f: g_θθ=%.5f (G-null=%.5f, 比=%.2f), R1=%.4f (G=%.4f), ω̄=%.4f" % (
            g, np.mean(gtt), np.mean(gttG), ratio, np.mean([r['R1'] for r in rs]),
            np.mean([r['R1_G'] for r in rs]), np.mean([r['omega'] for r in rs])), flush=True)

    # CSV
    import csv, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_phase_result.csv")
    keys = ['axis', 'gamma', 'c_mem', 'D', 'seed', 'lam_min', 'det', 'kappa',
            'g_rr', 'g_rt', 'g_tt', 'det_polar', 'lam_min_polar', 'kappa_polar',
            'omega', 'R1', 'phase_std', 'g_tt_G', 'R1_G', 'I', 'sigma_pair']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nCSV保存: %s" % out, flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()