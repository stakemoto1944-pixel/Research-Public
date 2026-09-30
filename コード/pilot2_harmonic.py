#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PILOT2: 第1調和成分の最小二乗推定器（差分推定器の Δθ^-2 増幅を回避）"""
import math
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

MU = 1.0
N_TRAJ = 200
N_ROT = 20
PER_ROT = 2000
NB = 40
SMOOTH = 4.0


def harmonic_fit(th, nb=NB, smooth=SMOOTH, eps=1e-12):
    """ln ρ(θ) の第1〜第3調和を最小二乗で求め g_θθ = (c1²+c2²)/2 を返す。"""
    H, edges = np.histogram(th, bins=nb, range=(0.0, 2 * math.pi))
    H = gaussian_filter1d(H, smooth, mode="wrap")
    H = np.clip(H, eps, None)
    L = np.log(H)
    L = L - L.mean()
    th_c = (edges[:-1] + edges[1:]) / 2.0
    cols, amps = [np.ones(nb)], []
    for k in (1, 2, 3):
        cols += [np.cos(k * th_c), np.sin(k * th_c)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    resid = float(np.mean((A @ coef - L) ** 2))
    amps = [float(math.hypot(coef[2 * k - 1], coef[2 * k])) for k in (1, 2, 3)]
    g1 = 0.5 * amps[0] ** 2
    return g1, amps, resid


def ensemble(sigma, omega0, omega1, corot, seed):
    dt = (2 * math.pi / omega0) / PER_ROT
    M = N_ROT * PER_ROT
    T = M * dt
    rng = np.random.default_rng(seed)
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


if __name__ == "__main__":
    t0 = time.perf_counter()
    w1 = 0.3
    print(f"第1調和成分の最小二乗推定器  nb={NB} smooth={SMOOTH}  n_traj={N_TRAJ} "
          f"N={N_TRAJ*N_ROT*PER_ROT:.2e}  N_rot={N_ROT}  ω1={w1}")
    print()
    hdr = (f"{'σ':>4} {'ω0':>6} | {'固定 g(0)':>11} {'固定 g':>11} {'差':>11} "
           f"{'SNR':>8} | {'共動 g(0)':>11} {'共動 g':>11} {'差':>11} | "
           f"{'予測':>11} {'予測/差':>8} {'ε':>6}")
    print(hdr)
    print("-" * len(hdr))
    for sigma in (1.0, 2.0):
        dth = sigma / MU
        for omega0 in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            out = []
            for arm, co in (("f", False), ("c", True)):
                f0 = np.mean([harmonic_fit(ensemble(sigma, omega0, 0.0, co, s))[0]
                              for s in (0, 1, 2)])
                f1 = [harmonic_fit(ensemble(sigma, omega0, w1, co, s))
                      for s in (0, 1, 2)]
                f1m = float(np.mean([v[0] for v in f1]))
                f1s = float(np.std([v[0] for v in f1]) / math.sqrt(3))
                out.append((f0, f1m, f1s))
            pred = w1 ** 2 / (2.0 * (dth ** 2 + omega0 ** 2))
            eps = w1 / math.sqrt(dth ** 2 + omega0 ** 2)
            (a0, a1, ae), (b0, b1, be) = out
            da, db = a1 - a0, b1 - b0
            print(f"{sigma:>4.1f} {omega0:>6.1f} | {a0:>11.3e} {a1:>11.3e} "
                  f"{da:>11.3e} {da/max(a0,1e-300):>8.1f} | {b0:>11.3e} "
                  f"{b1:>11.3e} {db:>11.3e} | {pred:>11.3e} "
                  f"{pred/max(da,1e-300):>8.2f} {eps:>6.3f}")
    print(f"\nelapsed {time.perf_counter()-t0:.1f}s")
