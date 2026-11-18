"""numba版の積分ループ（元スクリプトのループと同一の更新則）。
legacy=True : psi += (-Q + sqrt(2 D dt) * randn) * dt   （従来コード：Δtが二重）
legacy=False: psi += -Q*dt + sqrt(2 D dt) * randn        （標準のEuler-Maruyama）
"""
import numpy as np
try:
    from numba import njit
except ImportError:  # 遅いが同じ結果になる純Python版
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

@njit(cache=True)
def integrate(psi, Q, Js, Ja, gamma, c_mem, D, dt, steps, subsample, saved_steps,
              clip_bound, legacy, seed, history):
    np.random.seed(seed)
    n = psi.shape[0]
    amp = np.sqrt(2.0 * D * dt)
    save_idx = 0
    clip_hits = 0
    diverged = False
    xi = np.empty(n)
    for t in range(steps):
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += (Js[i, j] + Ja[i, j]) * psi[j]
            xi[i] = psi[i] ** 3 - psi[i] - s
        for i in range(n):
            Q[i] += (-gamma * Q[i] + c_mem * xi[i]) * dt
        for i in range(n):
            r = np.random.randn()
            if legacy:
                psi[i] += (-Q[i] + amp * r) * dt
            else:
                psi[i] += -Q[i] * dt + amp * r
        for i in range(n):
            if abs(psi[i]) > clip_bound:
                clip_hits += 1
            if psi[i] > clip_bound:
                psi[i] = clip_bound
            elif psi[i] < -clip_bound:
                psi[i] = -clip_bound
        bad = False
        for i in range(n):
            if np.isnan(psi[i]) or np.isinf(psi[i]):
                bad = True
        if bad:
            diverged = True
            break
        if t % subsample == 0 and save_idx < saved_steps:
            for i in range(n):
                history[save_idx, i] = psi[i]
            save_idx += 1
    return save_idx, clip_hits, diverged
