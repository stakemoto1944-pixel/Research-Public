#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（２）：修正版ダイナミクスでの Phase-1 スクリーニング（パラメータ探索込み）

修正版SPDE（数値実験(1)で修正）で：
  1. スキャン1: coupling_scale × D（γ=2.0, c_mem=1.0 固定）
  2. スキャン2: 記憶パラメータ γ × c_mem（coupling_scale=1.0, D=0.05 固定）
  3. 最良ROI点の複数シード頑健性

評価指標:
  - 全435ペアのクロススペクトル虚部の帯域積分 I_ij = ∫_{[0.5,5]Hz} Im[S_ij(f)] df
  - 同振幅 iid null（σ一致・白色雑音）の99%分位を閾値とし、超えるペア数を「有意ペア」と判定
  - ROI: 有意ペアが1以上（且つ符号一貫性 > 0.85）

実行: python3 experiment2_fixed_param_search.py
"""
import numpy as np
from scipy.signal import csd
from scipy.integrate import trapezoid

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
FS = 100.0
F1, F2 = 0.5, 5.0
NPERSEG = 256


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42):
    """修正版SPDEでシミュレーションし burn-in を破棄"""
    np.random.seed(seed)
    steps = int(T / DT)
    J_raw_sym = np.random.randn(N, N) / np.sqrt(N)
    J_sym = coupling_scale * 0.5 * (J_raw_sym + J_raw_sym.T)
    np.fill_diagonal(J_sym, 0.0)
    J_raw_asym = np.random.randn(N, N) / np.sqrt(N)
    J_asym = coupling_scale * 0.5 * (J_raw_asym - J_raw_asym.T)
    psi = np.random.randn(N) * 0.01
    Q = np.zeros(N)
    subsample = int(round(SAVE_DT / DT))
    saved_steps = steps // subsample
    history = np.zeros((saved_steps, N))
    saved_time = np.linspace(0, T, saved_steps)
    save_idx = 0
    diverged = False
    clip_hits = 0
    for t_idx in range(steps):
        xi = psi**3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        clip_hits += int(np.sum(np.abs(psi) > CLIP_BOUND))
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t_idx % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], clip_hits, diverged


def phase1_integrals(traj):
    """全ペアの帯域積分（符号付き）と符号一貫性行列"""
    n_dim = traj.shape[1]
    integrals = np.zeros((n_dim, n_dim))
    consistency = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            x, y = traj[:, i], traj[:, j]
            f, P = csd(x, y, fs=FS, nperseg=NPERSEG)
            m = (f >= F1) & (f <= F2)
            im = np.imag(P[m])
            integrals[i, j] = trapezoid(im, f[m])
            integrals[j, i] = -integrals[i, j]
            if len(im) > 0:
                consistency[i, j] = max(np.mean(im > 0), np.mean(im < 0))
    return integrals, consistency


def null_q99(sigma, n_t, n_reps=5):
    """同振幅iid白色雑音（σ一致）に対する |I| の99%分位（有意性判定の閾値）"""
    q99s = []
    for _ in range(n_reps):
        tr = np.random.default_rng(0).standard_normal((n_t, N)) * sigma
        integrals, _ = phase1_integrals(tr)
        vals = np.abs(integrals[np.triu_indices(N, 1)])
        q99s.append(np.percentile(vals, 99))
    return float(np.mean(q99s))


def evaluate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42):
    """1パラメータ点のROI判定"""
    t, traj, clip_hits, diverged = simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed)
    if diverged:
        return dict(diverged=True)
    integrals, consistency = phase1_integrals(traj)
    vals = np.abs(integrals[np.triu_indices(N, 1)])
    q99 = null_q99(traj.std(), len(traj))
    up = np.triu_indices(N, 1)
    sig = np.sum((vals > q99) & (consistency[up] > 0.85))
    return dict(
        diverged=False,
        maxI=float(vals.max()), meanI=float(vals.mean()),
        sigma=float(traj.std()), clip=clip_hits,
        n_sig=int(sig), sig_frac=float(sig / len(vals)),
        q99=q99, roi=bool(sig >= 1),
    )


if __name__ == "__main__":
    # ---- スキャン1: 結合強度 × ノイズ（γ=2.0, c_mem=1.0, T=150, burn-in=50） ----
    print("=" * 78)
    print("スキャン1: coupling_scale × D（γ=2.0, c_mem=1.0, T=150, burn-in=50）")
    print("=" * 78)
    print(f"{'scale':>5} {'D':>6} | {'max|I|':>10} {'mean|I|':>10} {'σ(ψ)':>7} {'clip':>5} | {'ROI':>5}")
    print("-" * 72)
    for cs in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for D in [0.005, 0.01, 0.02, 0.05, 0.1]:
            r = evaluate(cs, D, 2.0, 1.0, 150.0, 50.0)
            print(f"{cs:>5} {D:>6} | {r['maxI']:>10.5f} {r['meanI']:>10.5f} "
                  f"{r['sigma']:>7.3f} {r['clip']:>5} | {str(r['roi']):>5}")

    # ---- スキャン2: 記憶パラメータ γ × c_mem（cs=1.0, D=0.05, T=300, burn-in=100） ----
    print()
    print("=" * 78)
    print("スキャン2: γ × c_mem（coupling_scale=1.0, D=0.05, T=300, burn-in=100）")
    print("=" * 78)
    print(f"{'γ':>5} {'c_mem':>5} | {'max|I|':>10} {'mean|I|':>10} {'σ(ψ)':>7} {'clip':>5} | {'ROI':>5} 有意ペア率")
    print("-" * 72)
    for gamma in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for cm in [0.5, 1.0, 2.0, 5.0, 10.0]:
            r = evaluate(1.0, 0.05, gamma, cm, 300.0, 100.0)
            print(f"{gamma:>5} {cm:>5} | {r['maxI']:>10.5f} {r['meanI']:>10.5f} "
                  f"{r['sigma']:>7.3f} {r['clip']:>5} | {str(r['roi']):>5} {r['sig_frac']:>10.3f}")

    # ---- 頑健性: 最良ROI点の複数シード ----
    print()
    print("=" * 78)
    print("頑健性: (γ=0.1, c_mem=5.0, cs=1.0, D=0.05, T=300, burn-in=100) の複数シード")
    print("=" * 78)
    print(f"{'seed':>5} | {'max|I|':>10} {'mean|I|':>10} {'σ(ψ)':>7} | {'ROI':>5} 有意ペア率")
    print("-" * 72)
    for seed in [0, 1, 2, 3, 42, 100]:
        r = evaluate(1.0, 0.05, 0.1, 5.0, 300.0, 100.0, seed=seed)
        print(f"{seed:>5} | {r['maxI']:>10.5f} {r['meanI']:>10.5f} "
              f"{r['sigma']:>7.3f} | {str(r['roi']):>5} {r['sig_frac']:>10.3f}")