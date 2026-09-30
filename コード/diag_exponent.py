#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIAG: omega0 exponent mismatch diagnosis.

Hypothesis: the polar angular diffusivity D_theta = sigma/r^2 diverges as r->0,
            which makes the Euler step uncontrolled and corrupts g.
Compare: constant D_theta / soft-regularised / clipped 1-over-r-squared.
"""
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

MU = 1.0
N_TRAJ = 200
N_ROT = 20
PER_ROT = 2000
NB = 40
SMOOTH = 4.0


def harmonic_fit(th, nb=NB, smooth=SMOOTH):
    H, edges = np.histogram(th, bins=nb, range=(0.0, 2 * np.pi))
    H = np.clip(gaussian_filter1d(H, smooth, mode="wrap"), 1e-12, None)
    L = np.log(H)
    L = L - L.mean()
    th_c = (edges[:-1] + edges[1:]) / 2.0
    A = np.vstack([np.ones(nb)]
                  + [f(k * th_c) for k in (1, 2, 3) for f in (np.cos, np.sin)]).T
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    return 0.5 * (coef[1] ** 2 + coef[2] ** 2)


def ensemble(mode, sigma, omega0, omega1, rc, seed):
    dt = (2 * np.pi / omega0) / PER_ROT
    M = N_ROT * PER_ROT
    rng = np.random.default_rng(seed)
    r = np.full(N_TRAJ, 1.0)
    th = rng.uniform(0, 2 * np.pi, N_TRAJ)
    sr = np.sqrt(2.0 * sigma * dt)
    out = np.empty((M, N_TRAJ))
    mx = 0.0
    for k in range(M):
        r = np.clip(r + (MU * r - r ** 3) * dt + sr * rng.standard_normal(N_TRAJ),
                    1e-3, 6.0)
        if mode == "const":
            dth = np.full(N_TRAJ, sigma)
        elif mode == "soft":
            dth = sigma / (r * r + rc * rc)
        else:
            dth = np.minimum(sigma / (r * r), rc)
        mx = max(mx, float(dth.max()))
        th = th + (omega0 + omega1 * np.sin(th)) * dt \
            + np.sqrt(2.0 * dth * dt) * rng.standard_normal(N_TRAJ)
        out[k] = np.mod(th, 2 * np.pi)
    return out[int(0.2 * M):].ravel(), mx


def probe(mode, sigma, rc, w1, oms):
    ys, mxs = [], []
    for o in oms:
        e, mx = [], 0.0
        for s in (0, 1, 2):
            t0, _ = ensemble(mode, sigma, o, 0.0, rc, s)
            t1, a = ensemble(mode, sigma, o, w1, rc, s)
            e.append(harmonic_fit(t1) - harmonic_fit(t0))
            mx = max(mx, a)
        ys.append(float(np.mean(e)))
        mxs.append(mx)
    sl = float(np.polyfit(np.log(np.array(oms, float)),
                          np.log(np.abs(np.array(ys))), 1)[0])
    return sl, ys, mxs


if __name__ == "__main__":
    t0 = time.perf_counter()
    oms = [0.3, 1.0, 3.0, 10.0]
    w1 = 0.3
    print("omega0 exponent diagnosis (linear theory: -2)")
    print("%7s %6s %6s %9s %7s  " % ("mode", "sigma", "rc", "Dmax", "expo")
          + "".join("%11s" % ("w0=%g" % o) for o in oms))
    for mode, sigma, rc in [("const", 1.0, 0.0), ("const", 2.0, 0.0),
                            ("soft", 2.0, 0.4), ("soft", 2.0, 0.2),
                            ("clip", 2.0, 50.0), ("clip", 2.0, 10.0)]:
        sl, ys, mxs = probe(mode, sigma, rc, w1, oms)
        print("%7s %6.1f %6.2f %9.2f %7.2f  " % (mode, sigma, rc, max(mxs), sl)
              + "".join("%11.3e" % v for v in ys))
    print("\nelapsed %.1fs" % (time.perf_counter() - t0))
