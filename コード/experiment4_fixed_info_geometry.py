#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（４）：情報幾何学的特異点の検証（KDE & ブートストラップ）— 修正版

旧記録は `dummy_traj = randn(1000,30)*0.5`（ガウス雑音）に対して det g=9.95, λ_min=2.93 を
算出したもので、SPDE 実軌道での検証になっていなかった（無効）。

本スクリプトは修正版SPDE（実験2/3と同一: gamma=0.1, c_mem=5.0, cs=1.0, D=0.05,
T=300, burn-in=100）の実軌道に対して、同一のKDEフィッシャー情報幾何プロトコルで:

  C.  4条件比較（6シード平均±SE）: 
       ① ROI（非平衡駆動あり・実験3で因果確定）
       ② EQ（ J_asym=0 平衡対照）
       ③ DD（過減衰 γ=2.0・実験2でROI=Falseの実軌道）
       ④ GAUSS（①のペア(0,1)共分散に一致するガウス対照＝旧プロトコルの正規化版）
  D.  λ_min の条件間比較検定（対応なしt検定）
  E.  ブートストラップ CI（B=12, seed42, h=1.0）
  F.  バンド幅 h 掃引ロバストネスカーブ（h_scale ∈ {0.5,0.75,1.0,1.5,2.0}）
  G.  サブサンプリング収束（N ∈ {3000,6000,10000,20000}）
  Z.  旧ダミープロトコルの再現（整合性サニティチェック）

幾何学: ペア(0,1)の2次元定常密度を Scott 則 KDE で再構成し、中心差分（δ=1e-4）で
score を評価、g_ij = E[s_i s_j]（Fisher情報）。評価点は計算コスト管理のため
最大5000点の時間一様サブサンプルを使用（本番の意味的プロトコルは同一）。

実行: python3 experiment4_fixed_info_geometry.py
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
SEEDS = [0, 1, 2, 3, 42, 100]
BASE = dict(coupling_scale=1.0, gamma=0.1, c_mem=5.0, D=0.05, T=300.0, burnin=100.0)


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42, asym_gain=1.0):
    """修正版SPDE（実験2/3と同一実装）。J_asym に asym_gain を乗算。"""
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
    clip = 0
    for t in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        clip += int(np.sum(np.abs(psi) > CLIP_BOUND))
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], clip, diverged


def compute_geometry(traj, h_scale=1.0, n_eval=None):
    """
    KDEによるFisher情報計量（元ノートと同一プロトコル）:
    g_ij = mean_s[(∂_i log ρ)(∂_j log ρ)]，中心差分δ=1e-4，ρはε=1e-10でclip。
    評価点は時間一様サブサンプル（既定 max 5000点）＝計算コスト管理。
    """
    data = traj[:, :2].T  # (2, n)
    n = data.shape[1]
    if n_eval is None:
        n_eval = min(n, 5000)
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    pts = data[:, idx]
    try:
        kde = gaussian_kde(data, bw_method='scott')
        kde.set_bandwidth(kde.factor * h_scale)
        eps = 1e-10
        rho = np.clip(kde(pts), eps, None)
        delta = 1e-4
        dim = 2
        grad = np.zeros((dim, n_eval))
        for i in range(dim):
            pp = pts.copy(); pp[i, :] += delta
            pm = pts.copy(); pm[i, :] -= delta
            rp = np.clip(kde(pp), eps, None)
            rm = np.clip(kde(pm), eps, None)
            grad[i, :] = (np.log(rp) - np.log(rm)) / (2.0 * delta)
    except np.linalg.LinAlgError:
        return None
    g = np.zeros((dim, dim))
    for s in range(n_eval):
        g += np.outer(grad[:, s], grad[:, s])
    g /= n_eval
    ev = np.linalg.eigvalsh(g)
    lam_min = float(ev[0])
    lam_max = float(ev[1])
    det = float(lam_min * lam_max)
    kappa = float(lam_max / max(lam_min, 1e-300))
    return dict(g=g, det=det, lam_min=lam_min, lam_max=lam_max, kappa=kappa)


def gen_traj_roi(seed):
    t, tr, clip, div = simulate(BASE['coupling_scale'], BASE['D'], BASE['gamma'],
                                BASE['c_mem'], BASE['T'], BASE['burnin'], seed=seed, asym_gain=1.0)
    return tr, clip, div


def gen_traj_eq(seed):
    t, tr, clip, div = simulate(BASE['coupling_scale'], BASE['D'], BASE['gamma'],
                                BASE['c_mem'], BASE['T'], BASE['burnin'], seed=seed, asym_gain=0.0)
    return tr, clip, div


def gen_traj_dd(seed):
    t, tr, clip, div = simulate(1.0, 0.05, 2.0, 5.0, 300.0, 100.0, seed=seed, asym_gain=1.0)
    return tr, clip, div


def gen_traj_gauss(cov, seed, n=20000):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def summ(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')


if __name__ == "__main__":
    print("=== 実験4: 情報幾何学的特異点検証（修正版・実軌道・統計付き） ===", flush=True)
    print("基準: γ=0.1 c_mem=5.0 cs=1.0 D=0.05 T=300 burn-in=100, 6シード, 幾何=ペア(0,1)の2D KDE-Fisher", flush=True)

    # ---------- Z. 旧ダミープロトコルの再現（実装整合性チェック） ----------
    np.random.seed(42)
    dummy = np.random.randn(1000, 30) * 0.5
    z = compute_geometry(dummy[:, :2], h_scale=1.0, n_eval=1000)
    print("\nZ. 旧ダミープロトコル再現（randn(1000,30)*0.5, seed42）: det g=%.4f, λ_min=%.4f（旧記録9.9534/2.9292）"
          % (z['det'], z['lam_min']), flush=True)

    # ---------- C. 4条件・6シード（h=1.0） ----------
    conds = ['ROI', 'EQ', 'DD', 'GAUSS']
    cols = {c: dict(det=[], lam=[], kap=[], sig=[], rho=[]) for c in conds}
    roi_cov_tr42 = None
    for s in SEEDS:
        tr, clip, div = gen_traj_roi(s)
        assert not div
        if s == 42:
            roi_cov_tr42 = np.cov(tr[:, :2].T)
    for c in conds:
        for s in SEEDS:
            if c == 'ROI':
                tr, clip, div = gen_traj_roi(s)
            elif c == 'EQ':
                tr, clip, div = gen_traj_eq(s)
            elif c == 'DD':
                tr, clip, div = gen_traj_dd(s)
            if c == 'GAUSS':
                trg = gen_traj_gauss(roi_cov_tr42, s)
                r = compute_geometry(trg)
                cols[c]['sig'].append(float(np.std(trg[:, 0])))
                cols[c]['rho'].append(float(np.corrcoef(trg[:, 0], trg[:, 1])[0, 1]))
            else:
                assert not div
                r = compute_geometry(tr)
                cols[c]['sig'].append(float(np.std(tr[:, 0])))
                cols[c]['rho'].append(float(np.corrcoef(tr[:, 0], tr[:, 1])[0, 1]))
            cols[c]['det'].append(r['det']); cols[c]['lam'].append(r['lam_min'])
            cols[c]['kap'].append(r['kappa'])

    print("\n" + "=" * 104, flush=True)
    print("C. 4条件比較（6シード, h=1.0）", flush=True)
    print("=" * 104, flush=True)
    print("{:>6} | {:>11} {:>9} | {:>11} {:>9} | {:>9} {:>9} | {:>7} {:>7}".format(
        "条件", "det g", "±SE", "λ_min", "±SE", "κ", "±SE", "σ₁", "ρ"), flush=True)
    print("-" * 104, flush=True)
    for c in conds:
        d_m, d_se = summ(cols[c]['det']); l_m, l_se = summ(cols[c]['lam']); k_m, k_se = summ(cols[c]['kap'])
        sig_m, _ = summ(cols[c]['sig']); rho_m, _ = summ(cols[c]['rho'])
        print("{:>6} | {:>11.5f} {:>9.5f} | {:>11.5f} {:>9.5f} | {:>9.3f} {:>9.3f} | {:>7.3f} {:>7.3f}".format(
            c, d_m, d_se, l_m, l_se, k_m, k_se, sig_m, rho_m), flush=True)

    # λ_min 比較検定（ROI を基準に）
    print("\nD. λ_min 比較（対応なしt検定, ROI 基準）", flush=True)
    for c in ['EQ', 'DD', 'GAUSS']:
        tst = stats.ttest_ind(cols['ROI']['lam'], cols[c]['lam'])
        print("   ROI vs {:>5}: λ_min {:.4f} vs {:.4f}, t={:.2f}, p={:.4g}".format(
            c, np.mean(cols['ROI']['lam']), np.mean(cols[c]['lam']), tst.statistic, tst.pvalue), flush=True)

    # ---------- E. ブートストラップ（seed42, B=12, h=1.0） ----------
    n_eval = 5000
    print("\n" + "=" * 104, flush=True)
    print("E. ブートストラップCI（seed42, B=12, h=1.0, 評価点%d）" % n_eval, flush=True)
    print("=" * 104, flush=True)
    tr_roi42, _, _ = gen_traj_roi(42)
    tr_eq42, _, _ = gen_traj_eq(42)
    tr_dd42, _, _ = gen_traj_dd(42)
    tr_g42 = gen_traj_gauss(roi_cov_tr42, 42)
    for c, tr in [('ROI', tr_roi42), ('EQ', tr_eq42), ('DD', tr_dd42), ('GAUSS', tr_g42)]:
        rng = np.random.default_rng(7)
        dets, lams, kaps = [], [], []
        for _ in range(12):
            idxb = rng.integers(0, len(tr), size=len(tr))
            r = compute_geometry(tr[idxb], h_scale=1.0, n_eval=n_eval)
            dets.append(r['det']); lams.append(r['lam_min']); kaps.append(r['kappa'])
        dets, lams, kaps = map(np.array, (dets, lams, kaps))
        print("   {:>5}: det g [{:.4f}, {:.4f}]  λ_min [{:.4f}, {:.4f}]  κ [{:.2f}, {:.2f}]  (95% percentile)".format(
            c, *np.percentile(dets, [2.5, 97.5]), *np.percentile(lams, [2.5, 97.5]),
            *np.percentile(kaps, [2.5, 97.5])), flush=True)

    # ---------- F. バンド幅掃引ロバストネス（seed42） ----------
    h_list = [0.5, 0.75, 1.0, 1.5, 2.0]
    print("\n" + "=" * 104, flush=True)
    print("F. バンド幅 h_scale 掃引（seed42）", flush=True)
    print("=" * 104, flush=True)
    print("{:>6} | {:>11} {:>11} | {:>11} {:>11}".format("h", "det g (ROI)", "λ_min (ROI)", "det g (GAUSS)", "λ_min (GAUSS)"), flush=True)
    print("-" * 104, flush=True)
    for h in h_list:
        r1 = compute_geometry(tr_roi42, h_scale=h)
        r2 = compute_geometry(tr_g42, h_scale=h)
        print("{:>6} | {:>11.5f} {:>11.5f} | {:>11.5f} {:>11.5f}".format(
            h, r1['det'], r1['lam_min'], r2['det'], r2['lam_min']), flush=True)

    # ---------- G. サブサンプリング収束（ROI seed42, h=1.0） ----------
    N_list = [3000, 6000, 10000, 20000]
    print("\n" + "=" * 104, flush=True)
    print("G. サブサンプリング収束（ROI seed42, h=1.0）", flush=True)
    print("=" * 104, flush=True)
    print("{:>7} | {:>11} {:>11} {:>9}".format("N", "det g", "λ_min", "κ"), flush=True)
    print("-" * 104, flush=True)
    for Nt in N_list:
        r = compute_geometry(tr_roi42[:Nt], h_scale=1.0)
        print("{:>7} | {:>11.5f} {:>11.5f} {:>9.3f}".format(Nt, r['det'], r['lam_min'], r['kappa']), flush=True)

    print("\nDONE", flush=True)