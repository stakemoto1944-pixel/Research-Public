import numpy as np, sys
from numba import njit

@njit(cache=True)
def step(psi, Q, Js, Ja, gamma, c, dt):
    n = psi.shape[0]
    xi = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += (Js[i, j] + Ja[i, j]) * psi[j]
        xi[i] = psi[i] ** 3 - psi[i] - s
    for i in range(n):
        Q[i] += (-gamma * Q[i] + c * xi[i]) * dt
    for i in range(n):
        psi[i] += -Q[i] * dt            # D=0: ノイズなし（従来式・標準式とも同一）

@njit(cache=True)
def lyap(psi, Q, Js, Ja, gamma, c, dt, nsteps, nrenorm, d0, seed, burn):
    np.random.seed(seed)
    n = psi.shape[0]
    psi2 = psi.copy(); Q2 = Q.copy()
    v = np.random.randn(2 * n); v *= d0 / np.sqrt((v ** 2).sum())
    for i in range(n):
        psi2[i] += v[i]; Q2[i] += v[n + i]
    acc = 0.0; cnt = 0; t = 0.0; run = 0.0
    for k in range(nsteps):
        step(psi, Q, Js, Ja, gamma, c, dt)
        step(psi2, Q2, Js, Ja, gamma, c, dt)
        if (k + 1) % nrenorm == 0:
            d = 0.0
            for i in range(n):
                d += (psi2[i] - psi[i]) ** 2 + (Q2[i] - Q[i]) ** 2
            d = np.sqrt(d)
            if k * dt >= burn:
                acc += np.log(d / d0); cnt += 1
            sc = d0 / d
            for i in range(n):
                psi2[i] = psi[i] + (psi2[i] - psi[i]) * sc
                Q2[i] = Q[i] + (Q2[i] - Q[i]) * sc
    return acc / (cnt * nrenorm * dt)

N = 30; DT = 5e-4
def run(seed, gamma, c, asym, T=300.0, burn=100.0, scale=1.0):
    np.random.seed(seed)
    Jr = np.random.randn(N, N) / np.sqrt(N); Js = scale * 0.5 * (Jr + Jr.T); np.fill_diagonal(Js, 0.0)
    Jr = np.random.randn(N, N) / np.sqrt(N); Ja = scale * 0.5 * (Jr - Jr.T) * asym
    psi = np.random.randn(N) * 0.01; Q = np.zeros(N)
    # 過渡期を捨ててからBenettin
    return lyap(psi, Q, Js, Ja, gamma, c, DT, int(T / DT), 20, 1e-8, seed, burn)

if __name__ == "__main__":
    print("D=0 最大リアプノフ指数 λ1 [1/時間]  (T=300, burn-in 100, 摂動1e-8, 20ステップごとに再規格化)")
    for name, g, c, a in [("ROI   γ=0.1 J_asym on ", 0.1, 5.0, 1.0), ("EQ    γ=0.1 J_asym=0", 0.1, 5.0, 0.0),
                          ("DD    γ=2.0 J_asym on ", 2.0, 5.0, 1.0), ("fast  γ=1.0 J_asym on ", 1.0, 5.0, 1.0)]:
        vals = [run(s, g, c, a) for s in (0, 1, 2, 3, 42, 100)]
        print(f"{name}: " + "  ".join(f"{v:+.4f}" for v in vals) + f"   平均 {np.mean(vals):+.4f} ± {np.std(vals, ddof=1)/np.sqrt(len(vals)):.4f}")
