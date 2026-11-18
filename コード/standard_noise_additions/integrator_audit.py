#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
積分器監査ツール（Euler–Maruyama のノイズ尺度の検査）

目的: 論文の SDE  dψ = f(ψ) dt + sqrt(2D) dW  を離散化した更新式が、ノイズを正しい強さで入れているかを、
      解析解のある2つの系で検査する。旧論文（I〜III）のコードは、更新式を step() の形で差し込めば同じ検査にかけられる。

検査:
  T1 自由拡散   f=0        : Var[ψ_t] = 2 D t                  （比 = 実測/理論 が 1 でなければならない）
  T2 OU過程     f=-θψ      : 定常分散 = D/θ                    （比が 1 に近い。離散化誤差は O(θΔt)）
  T3 Δt二重検出 T1 の比が Δt² に比例して縮むなら、ノイズに Δt が余分に掛かっている（比 ≈ Δt²）。
                √Δt が欠けていれば比は 1/Δt に比例して増える。

差し込み方:  step(psi, drift, dt, D, z) -> psi_next   （z は標準正規乱数配列）
"""
import numpy as np

# ---- 検査対象の更新式（旧: Δt二重, 標準: 正しい EM, 逆: ノイズ過大の例） ----
def step_standard(psi, drift, dt, D, z):
    return psi + drift * dt + np.sqrt(2 * D * dt) * z           # 正しい Euler–Maruyama

def step_legacy(psi, drift, dt, D, z):
    return psi + (drift + np.sqrt(2 * D * dt) * z) * dt         # ノイズに Δt が二重（旧実装）

def step_wrong_scale(psi, drift, dt, D, z):
    return psi + drift * dt + np.sqrt(2 * D) * z                # √Δt の欠落（ノイズ過大）

def free_diffusion(step, D=0.05, dt=5e-4, T=10.0, n=4000, seed=0):
    rng = np.random.default_rng(seed); psi = np.zeros(n)
    for _ in range(int(T / dt)):
        psi = step(psi, 0.0, dt, D, rng.standard_normal(n))
    return psi.var(ddof=1) / (2 * D * T)

def ou_stationary(step, theta=1.0, D=0.05, dt=5e-4, T=40.0, burn=10.0, n=2000, seed=1):
    rng = np.random.default_rng(seed); psi = np.zeros(n); acc = []; steps = int(T / dt); b = int(burn / dt)
    for k in range(steps):
        psi = step(psi, -theta * psi, dt, D, rng.standard_normal(n))
        if k >= b and (k - b) % 200 == 0: acc.append(psi.var(ddof=1))
    return np.mean(acc) / (D / theta)

if __name__ == "__main__":
    print("積分器監査（解析解との比。1.000 が正しい）")
    print("%-14s | %-22s | %-24s | %s" % ("更新式", "T1 自由拡散 Var/(2Dt)", "T2 OU 定常分散 /(D/θ)", "判定"))
    for name, st in [("standard", step_standard), ("legacy(Δt二重)", step_legacy), ("√Δt欠落", step_wrong_scale)]:
        t1 = free_diffusion(st); t2 = ou_stationary(st)
        ok = abs(t1 - 1) < 0.05 and abs(t2 - 1) < 0.05
        print("%-14s | %22.4f | %24.4f | %s" % (name, t1, t2, "合格" if ok else "不合格"))
    print("\nT3 Δt依存（T1 の比）: legacy の比は Δt² に比例して縮み（比/Δt² ≈ 1）、standard は Δt によらず 1")
    for dt in (2e-3, 1e-3, 5e-4):
        s = free_diffusion(step_standard, dt=dt, T=5.0, n=3000); l = free_diffusion(step_legacy, dt=dt, T=5.0, n=3000)
        print("  Δt=%.0e: standard %.4f  legacy %.3e  legacy/Δt² = %.3f" % (dt, s, l, l / dt**2))
