#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PILOT: g_{θθ} 推定器のノイズ床の実測（実験16の設計判断用）

ω1 = 0 の系は厳密に回転対称（ρ = P(r)/2π）なので真の g_θθ ≡ 0。
したがって ω1=0 で測った値は推定器ノイズ床そのものである。
ω1 ≠ 0 との差が信号対雑比の可否を決める。
"""
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

MU = 1.0
N_TRAJ = 40
T_OBS = 20.0
SEEDS = (0, 1)
PER_ROT = 2000          # 回転周期あたりのサンプル数 ≈ 有効独立サンプル数


def g_tt_marginal(th, nb=40, smooth=2.0, eps=1e-12):
    """θ 周辺分布のみから g_θθ を評価する。"""
    H, edges = np.histogram(th, bins=nb, range=(0.0, 2 * math.pi))
    Hs = gaussian_filter1d(H, smooth, mode="wrap")
    Hs = np.clip(Hs, eps, None)
    L = np.log(Hs)
    d = edges[1] - edges[0]
    dL = (np.roll(L, -1) - np.roll(L, 1)) / (2 * d)
    P = Hs / Hs.sum()
    return float(np.sum(P * dL ** 2))


def ensemble(sigma, omega0, omega1, corot, seed):
    """n_traj 本の極座標 Stuart-Landau をベクトル並列で回す。"""
    rng = np.random.default_rng(seed)
    dt = (2 * math.pi / omega0) / PER_ROT
    M = int(round(T_OBS / dt))
    r = np.full(N_TRAJ, 1.0)
    th = rng.uniform(0, 2 * math.pi, N_TRAJ)
    sr = math.sqrt(2.0 * sigma * dt)
    out = np.empty((M, N_TRAJ))
    for k in range(M):
        ph0 = omega0 * (k * dt) if corot else 0.0
        r = np.clip(r + (MU * r - r ** 3) * dt + sr * rng.standard_normal(N_TRAJ),
                    1e-3, 6.0)
        th = th + (omega0 + omega1 * np.sin(th - ph0)) * dt \
            + np.sqrt(2.0 * sigma / (r * r) * dt) * rng.standard_normal(N_TRAJ)
        out[k] = np.mod(th, 2 * math.pi)
    return out[int(0.2 * M):].ravel()


def measure(sigma, omega0, omega1, corot, nb=40):
    acc = [g_tt_marginal(ensemble(sigma, omega0, omega1, corot, s), nb=nb)
           for s in SEEDS]
    return float(np.mean(acc)), float(np.std(acc) / max(np.sqrt(len(acc)), 1))


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("g_θθ 推定器のパイロット（θ 周辺のみ、nb=40、平滑 σ_f=2）")
    print(f"n_traj={N_TRAJ} T_obs={T_OBS}s  →  1本あたり {T_OBS*50000:.0f} サンプル、"
          f"N_eff≈{PER_ROT}")
    print()
    hdr = (f"{'σ':>5} {'ω0':>6} | {'固定 w1=0':>11} {'固定 w1=1':>11} {'差':>10} "
           f"{'SNR':>7} | {'共動 w1=0':>11} {'共動 w1=1':>11} {'差':>10} | "
           f"{'予測(固定)':>11} {'予測/床':>9}")
    print(hdr)
    print("-" * len(hdr))
    for sigma in (2.0,):
        for omega0 in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
            w1 = 1.0
            f0, _ = measure(sigma, omega0, 0.0, False)
            f1, fe = measure(sigma, omega0, w1, False)
            c0, _ = measure(sigma, omega0, 0.0, True)
            c1, ce = measure(sigma, omega0, w1, True)
            dth = sigma / MU
            pred = math.pi ** 2 * w1 ** 2 / (2 * (dth ** 2 + omega0 ** 2))
            eps_mod = w1 / math.sqrt(dth ** 2 + omega0 ** 2)
            print(f"{sigma:>5.1f} {omega0:>6.1f} | {f0:>11.4e} {f1:>11.4e} "
                  f"{f1-f0:>10.3e} {(f1-f0)/max(f0,1e-300):>7.1f} | "
                  f"{c0:>11.4e} {c1:>11.4e} {c1-c0:>10.3e} | "
                  f"{pred:>11.4e} {pred/max(f0,1e-300):>9.2e}  ε={eps_mod:.3f}")
    print(f"\nelapsed {time.perf_counter()-t0:.1f}s")
