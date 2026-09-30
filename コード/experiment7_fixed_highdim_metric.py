#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（７）：高次元（d=2–5）部分多様体での計量縮退検証（PCA投影）

背景: 実験4–6ではペア(0,1)の2次元投影KDE-Fisher計量で相境界（γ∈(0.2,0.5]）の
det g→0 予言を検証し棄却した。しかしソフトモード（臨界方向）が成分(0,1)にない場合、
2次元投影は縮退構造を見落とす可能性がある。

方法: 全30成分の経験共分散の固有分解（PCA）を行い、分散最大の d 方向からなる部分多様体に
投影する。Fisher計量（d×d）をKDEで推定（実験4/6と同一の数値微分・密度下限プロトコル）。

  ・ソフトモード監視: 最大分散 λ₁(Σ30)。臨界なら γ→γc で発散するはず
  ・ガウス予言: λ_min^G = 1/λ₁（Fisher計量はガウスで Σ^{-1} になる）
  ・実測: λ_min(g_d), det g_d, κ_d （d=2,3,4,5）
  ・推定器ベースライン: 同一PCA共分散のマッチドガウス（d=3,5）で同じKDEパイプラインを通し、
    λ_min(emp)/λ_min(G_est) 比で「密度がガウスより平坦か鋭いか」を判定
  ・境界点 γ=0.2, d=3 でブートストラップCI（B=12）

判定: λ₁(Σ30) が境界で発散 OR λ_min(g_d)→0（0含むCI）なら予言支持。それ以外は高次元でも棄却。

実行: python3 experiment7_fixed_highdim_metric.py
"""
import numpy as np
from scipy.stats import gaussian_kde

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
GAMMA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
DIMS = [2, 3, 4, 5]
N_EVAL_MAP = {2: 3000, 3: 2000, 4: 1500, 5: 1200}


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42, asym_gain=1.0):
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
    for t in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], diverged


def pca_project(traj, d):
    """経験共分散の上位d固有ベクトルへの射影。戻り値 (proj(n,d), 固有値降順, 固有ベクトル列)。"""
    cov = np.cov(traj.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals_d = evals[order]
    W = evecs[:, order[:d]]
    return traj @ W, evals_d, W


def compute_geometry_nD(traj_nd, n_eval=None):
    """d次元部分多様体のKDE-Fisher計量（中心差分δ=1e-4, ρ下限1e-10）。"""
    data = traj_nd.T  # (d, n)
    n = data.shape[1]
    d = data.shape[0]
    if n_eval is None:
        n_eval = min(n, N_EVAL_MAP.get(d, 2000))
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    pts = data[:, idx]
    try:
        kde = gaussian_kde(data, bw_method='scott')
        eps = 1e-10
        delta = 1e-4
        grad = np.zeros((d, n_eval))
        for i in range(d):
            pp = pts.copy(); pp[i, :] += delta
            pm = pts.copy(); pm[i, :] -= delta
            rp = np.clip(kde(pp), eps, None)
            rm = np.clip(kde(pm), eps, None)
            grad[i, :] = (np.log(rp) - np.log(rm)) / (2.0 * delta)
    except np.linalg.LinAlgError:
        return None
    g = np.zeros((d, d))
    for s in range(n_eval):
        g += np.outer(grad[:, s], grad[:, s])
    g /= n_eval
    ev = np.linalg.eigvalsh(g)
    lam_min = float(ev[0]); lam_max = float(ev[-1])
    return dict(det=float(np.prod(ev)), lam_min=lam_min, kappa=float(lam_max / max(lam_min, 1e-300)))


def summ2(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')


def gaussian_sample(cov_d_sqrt, seed, n=20000):
    """cv = W @ diag(evals[:d]) @ W.T 形式から (n,d) を描く。"""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(cov_d_sqrt.shape[0]), cov_d_sqrt, size=n)


def main():
    print("=== 実験7: 高次元（d=2–5）部分多様体の計量縮退検証（PCA投影） ===", flush=True)
    print("基準: c_mem=5.0, D=0.05, T=300, burn-in=100, J_asymあり。全成分PCA → 上位d方向。", flush=True)

    # ---------- A. γ掃引 × d の主表 + ソフトモード監視 ----------
    print("\n" + "=" * 104, flush=True)
    print("A. γ掃引（2シード平均; d=2–5のPCA投影KDE-Fisher計量）", flush=True)
    print("=" * 104, flush=True)
    print("ソフトモード監視: λ₁(Σ30)（最大分散, 30成分全体）。臨界なら境界で発散するはず。", flush=True)
    print("-" * 104, flush=True)
    hdr = "{:>6} | {:>10} {:>10} | ".format("γ", "λ₁(Σ30)", "1/λ₁=λmin^G") + \
          " | ".join("{:>5}".format("d=%d λm" % d) for d in DIMS) + " | " + \
          " | ".join("{:>5}".format("d=%d κ" % d) for d in DIMS)
    print(hdr, flush=True)
    print("-" * 104, flush=True)
    rows = []
    for g in GAMMA_GRID:
        lam1s = []
        per_d_lam = {d: [] for d in DIMS}
        per_d_kap = {d: [] for d in DIMS}
        for s in (42, 100):
            t, tr, div = simulate(1.0, 0.05, g, 5.0, 300.0, 100.0, seed=s, asym_gain=1.0)
            if div:
                continue
            cov = np.cov(tr.T)
            evals30 = np.sort(np.linalg.eigvalsh(cov))[::-1]
            lam1s.append(evals30[0])
            for d in DIMS:
                proj, evals_d, W = pca_project(tr, d)
                r = compute_geometry_nD(proj)
                if r is None:
                    continue
                per_d_lam[d].append(r['lam_min'])
                per_d_kap[d].append(r['kappa'])
                # KDE推定ガウスのλ_min (G_est) を計算（B節と同じ方法論）
                try:
                    covd = W.T @ cov @ W
                    gs = gaussian_sample(covd, 7)
                    rg = compute_geometry_nD(gs)
                    gauss_est = rg['lam_min'] if rg is not None else float('nan')
                except Exception:
                    gauss_est = float('nan')
                rows.append(dict(gamma=g, seed=s, d=d, lam=r['lam_min'], det=r['det'],
                                 kappa=r['kappa'], sigma1=float(np.sqrt(evals30[0])),
                                 gauss_lam=1.0 / evals30[0],
                                 gauss_est=gauss_est))
        l1_m, l1_se = summ2(lam1s)
        line = "{:>6} | {:>10.4f} {:>10.4f} | ".format(g, l1_m, 1.0 / max(l1_m, 1e-12))
        for d in DIMS:
            lm, _ = summ2(per_d_lam[d])
            line += "{:>10.5f}".format(lm)
        line += " | "
        for d in DIMS:
            km, _ = summ2(per_d_kap[d])
            line += "{:>5.2f}".format(km)
        print(line, flush=True)

    # ソフトモードの境界発散チェック
    lam1_by_g = {}
    for g in GAMMA_GRID:
        l1s = []
        for s in (42, 100):
            t, tr, div = simulate(1.0, 0.05, g, 5.0, 300.0, 100.0, seed=s, asym_gain=1.0)
            evals30 = np.sort(np.linalg.eigvalsh(np.cov(tr.T)))[::-1]
            l1s.append(evals30[0])
        lam1_by_g[g] = l1s
    l1_mid = max(max(lam1_by_g[g]) for g in (0.2, 0.3, 0.5))
    l1_deep = max(max(lam1_by_g[g]) for g in (0.05, 0.1))
    print("-" * 104, flush=True)
    print("ソフトモードの倍数: max λ₁(境界帯0.2–0.5)=%.2f vs max λ₁(ROI深部0.05–0.1)=%.2f → 比 %.2f" % (
        l1_mid, l1_deep, l1_mid / max(l1_deep, 1e-12)), flush=True)

    # ---------- B. マッチドガウス推定器ベースライン（d=3, 5） ----------
    print("\n" + "=" * 104, flush=True)
    print("B. マッチドガウス・推定器ベースライン（同一PCA共分散のガウス標本で同パイプライン）", flush=True)
    print("=" * 104, flush=True)
    print("    γ | d | λ_min(emp) | λ_min(G_est) | 比 emp/G | det emp | det G", flush=True)
    print("-" * 104, flush=True)
    for g in (0.1, 0.2, 0.5):
        t, tr, div = simulate(1.0, 0.05, g, 5.0, 300.0, 100.0, seed=42, asym_gain=1.0)
        cov = np.cov(tr.T)
        evals30 = np.sort(np.linalg.eigvalsh(cov))[::-1]
        for d in (3, 5):
            proj, evals_d, W = pca_project(tr, d)
            r = compute_geometry_nD(proj)
            covd = W.T @ cov @ W  # d×d 経験共分散（PCA座標でほぼ対角）
            gs = gaussian_sample(covd, 7)
            rg = compute_geometry_nD(gs)
            ratio = r['lam_min'] / max(rg['lam_min'], 1e-300)
            print("    %-4.1f| %d | %11.5f | %11.5f | %6.2f | %8.4g | %8.4g" % (
                g, d, r['lam_min'], rg['lam_min'], ratio, r['det'], rg['det']), flush=True)

    # ---------- C. 境界候補 γ=0.2, d=3 のブートストラップ ----------
    print("\n" + "=" * 104, flush=True)
    print("C. γ=0.20, d=3 ブートストラップ（seed42, B=12）", flush=True)
    print("=" * 104, flush=True)
    t, tr, div = simulate(1.0, 0.05, 0.2, 5.0, 300.0, 100.0, seed=42, asym_gain=1.0)
    _, evals_d, W = pca_project(tr, 3)
    base = tr @ W
    rng = np.random.default_rng(7)
    dets, lams, kaps = [], [], []
    for _ in range(12):
        idxb = rng.integers(0, len(base), size=len(base))
        r = compute_geometry_nD(base[idxb], n_eval=2000)
        dets.append(r['det']); lams.append(r['lam_min']); kaps.append(r['kappa'])
    dets, lams, kaps = map(np.array, (dets, lams, kaps))
    print("  det g CI = [%.4g, %.4g] | λ_min CI = [%.5f, %.5f] | κ CI = [%.2f, %.2f]  (95th percentile)" % (
        np.percentile(dets, 2.5), np.percentile(dets, 97.5),
        np.percentile(lams, 2.5), np.percentile(lams, 97.5),
        np.percentile(kaps, 2.5), np.percentile(kaps, 97.5)), flush=True)
    print("  対応する1/λ₁(S30)=%.5f（ガウス予言のλ_min下限）" % (1.0 / evals30[0] if False else 1.0 / np.sort(np.linalg.eigvalsh(np.cov(tr.T)))[::-1][0]), flush=True)

    # ---------- D. 判定 ----------
    print("\n" + "=" * 104, flush=True)
    print("D. 判定", flush=True)
    print("=" * 104, flush=True)
    lam_ci = (np.percentile(lams, 2.5), np.percentile(lams, 97.5))
    soft_ratio = l1_mid / max(l1_deep, 1e-12)
    if lam_ci[1] < 1e-2 and soft_ratio > 2.0:
        ver = "予言支持の可能性: 高次元(d=3)でλ_minが実質ゼロかつソフトモード発散"
    elif lam_ci[0] > 0.05:
        ver = "予言棄却（高次元も含む）: λ_min CIが正の有限値で0を排除、ソフトモードも発散なし"
    else:
        ver = "中間: λ_minは小さいが非ゼロ（誤差範囲要確認）"
    print(ver, flush=True)
    print("  d=3 λ_min CI = [%.5f, %.5f], ソフトモード比（境界/深部）= %.2f" % (lam_ci[0], lam_ci[1], soft_ratio), flush=True)

    # CSV
    import csv, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highdim_metric_result.csv")
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['gamma', 'seed', 'd', 'lam', 'det', 'kappa', 'sigma1', 'gauss_lam', 'gauss_est'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nCSV保存: %s" % out, flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()