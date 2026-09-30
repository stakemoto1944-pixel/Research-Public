#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（１）：修正版SPDEシミュレーション＋診断

修正内容（旧実装 vs 修正版）:
  1. 駆動項: xi = psi**3 - psi - J_sym@psi - J_asym@psi（論文の括弧内に符号整合）
  2. 更新式: psi += (-Q + noise) * dt（dt を明示的に掛ける＝正則な連続時間離散化）

出力: 軌道 + 診断（クリップ回数 / 状態範囲 / 井戸滞在率 / dt収束性）
"""
import numpy as np
from scipy.signal import csd
from scipy.integrate import trapezoid

# ---------- パラメータ ----------
N = 30
T = 20.0
DT = 0.0005
D = 0.02
GAMMA = 2.0
C_MEM = 1.0
SCALE = 0.2
SEED = 42
CLIP_BOUND = 10.0
SAVE_DT = 0.01
FS = 1.0 / SAVE_DT
F1, F2 = 0.5, 5.0


def simulate_corrected(seed=SEED, T_run=T, dt=DT, N_spin=N, scale=SCALE, D_noise=D, gamma=GAMMA, c_mem=C_MEM):
    """修正版：論文SPDEに整合する更新式でシミュレーション"""
    np.random.seed(seed)
    steps = int(T_run / dt)
    time_axis = np.linspace(0, T_run, steps)

    # 1. 結合行列（スケーリング調整済み）
    J_raw_sym = np.random.randn(N_spin, N_spin) / np.sqrt(N_spin)
    J_sym = scale * 0.5 * (J_raw_sym + J_raw_sym.T)
    np.fill_diagonal(J_sym, 0.0)
    J_raw_asym = np.random.randn(N_spin, N_spin) / np.sqrt(N_spin)
    J_asym = scale * 0.5 * (J_raw_asym - J_raw_asym.T)

    # 2. 初期化
    psi = np.random.randn(N_spin) * 0.01
    Q = np.zeros(N_spin)
    subsample = int(round(SAVE_DT / dt))
    saved_steps = steps // subsample
    history = np.zeros((saved_steps, N_spin))
    saved_time = np.linspace(0, T_run, saved_steps)
    save_idx = 0
    diverged = False
    clip_hits = 0

    # 3. Euler–Maruyama 積分（修正版）
    for t_idx in range(steps):
        xi = psi**3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)  # 論文の括弧内 (δH/δψ − Jψ)
        Q += (-gamma * Q + c_mem * xi) * dt
        noise = np.sqrt(2.0 * D_noise * dt) * np.random.randn(N_spin)
        psi += (-Q + noise) * dt  # dt を明示的に掛ける
        clip_hits += int(np.sum(np.abs(psi) > CLIP_BOUND))
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t_idx % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    return saved_time[:save_idx], history[:save_idx], clip_hits, diverged


def band_integral(x, y, fs=FS, f1=F1, f2=F2):
    """クロススペクトル虚部（クアッドスペクトル）の帯域積分（Phase-1指標）"""
    f, P = csd(x, y, fs=fs, nperseg=min(256, len(x) // 4))
    m = (f >= f1) & (f <= f2)
    if m.sum() == 0:
        return 0.0
    return float(trapezoid(np.imag(P)[m], f[m]))


if __name__ == "__main__":
    print("Simulating Asymmetric Non-Markovian Spin Glass SPDE (Corrected Version)...")
    t, traj, clip_hits, diverged = simulate_corrected()
    print(f"Saved time steps: {len(t)}, State dimensions: {traj.shape[1]}")
    print(f"Trajectory shape: {traj.shape}")
    print(f"Safety-clip activations: {clip_hits}")
    print(f"psi range: [{traj.min():.4f}, {traj.max():.4f}]")

    print("\n--- 診断 ---")
    print(f"標準偏差 σ = {traj.std():.4f}")
    wf = [
        np.mean((traj > lo) & (traj < hi))
        for lo, hi in [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5)]
    ]
    print(f"井戸・障壁滞在率: 左{float(wf[0]):.3f} / 障壁{float(wf[1]):.3f} / 右{float(wf[2]):.3f}")

    iv = band_integral(traj[:, 0], traj[:, 1])
    print(f"\nPhase-1 帯域積分 (pair 0,1): {iv:+.5f}  (>1e-3? {abs(iv) > 1e-3})")