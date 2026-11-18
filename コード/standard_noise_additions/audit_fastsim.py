# -*- coding: utf-8 -*-
"""差し替え後の積分器 fastsim.integrate を、解析解のある自由拡散で監査する。
c_mem=0 とすると Q=0 となり psi は純粋な拡散になる: Var(psi_T)/(2 D T) = 1 が標準EM。
legacy=True は旧実装（Δt二重）を再現するためだけのフラグで、比は Δt² になる。
実行: python3 audit_fastsim.py   （standard_noise_additions/ から）"""
import sys, numpy as np
sys.path.insert(0, "..")
import fastsim
N, D, DT, T = 30, 0.02, 5e-4, 5.0
steps = int(T / DT); sub = 10; saved = steps // sub
def run(legacy, reps=200):
    v = []
    for r in range(reps):
        psi = np.zeros(N); Q = np.zeros(N); Z = np.zeros((N, N)); hist = np.zeros((saved, N))
        fastsim.integrate(psi, Q, Z, Z, 2.0, 0.0, D, DT, steps, sub, saved, 1e9, legacy, 1000 + r, hist)
        v.append(psi.copy())
    return np.var(np.concatenate(v), ddof=1) / (2 * D * T)
for name, lg in [("standard (legacy=False)", False), ("legacy   (legacy=True)", True)]:
    r = run(lg); print("%-26s Var/(2DT) = %.4e  -> %s" % (name, r, "合格" if abs(r - 1) < 0.05 else "不合格 (Δt²=%.2e)" % DT**2))
