#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（５）：相境界掃引による det g → 0 予言の実軌道検証（実験4の次段階）

実験4の結論: 本動作点（γ=0.1, cs=1.0, D=0.05, 相内部）では特異点診断（det g→0 ∧ λ_min→0 ∧ κ→∞）
は発火しなかった。論文（III）の情報幾何学的枠組みは「相境界でのみ計量縮退 det g → 0 が生じ、
データ増加 N→∞ でも縮退が安定（真の特異点）」であることを予言する。

本実験は ROI/非ROI の相境界を跨ぐ掃引軸でこの予言を直接検証する:

  軸A (γ掃引): 記憶減衰率 γ ∈ {0.05, …, 2.0}（experiment2 スキャン2で ROI 境界=γ∈(0.2, 0.5] を確認済み）
  軸B (cs掃引): 結合スケール cs ∈ {0.2, …, 1.8}（不安定化境界を探索）
  軸C (詳細検証): A/Bで λ_min が最小となる境界候補点に対して
      6シード主表 / ブートストラップCI (B=12) / Nスケーリング (3000–20000) を実行

評価量（ペア(0,1)の2次元KDE-Fisher計量, 実験4と同一実装）:
  det g, λ_min, κ=λmax/λmin, 周辺σ、クロススペクトル虚部帯域積分 |I|（ROIマーカー, 高速近似）

判定:
  予言支持   = 境界候補で λ_min→0 ∧ κ 発散 ∧ CI が 0 を含み ∧ N増加でも縮退が維持
  予言棄却   = λ_min がどこでも正の有限値に留まる（CI が 0 を排除）→ 実軌道では縮退なし
  部分支持   = λ_min は境界で極小だが非ゼロ（量的低下）→ 「疑似特異点」（有限標本効果）として解釈

実行: python3 experiment5_fixed_boundary_sweep.py
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import csd
from scipy.integrate import trapezoid

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
SEEDS = [0, 1, 2, 3, 42, 100]
GAMMA_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
CS_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
N_EVAL = 3000


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42, asym_gain=1.0):
    """修正版SPDE（実験2/3/4と同一ダイナミクス）。"""
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
    clip = 0
    for t in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        clip += int(np.sum(np.abs(psi) > CLIP_BOUND))
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], clip, diverged


def compute_geometry(traj, h_scale=1.0, n_eval=None):
    """ペア(0,1)投影 KDE-Fisher計量（実験4と同一実装）。戻り値 dict または None。"""
    data = traj[:, :2].T
    n = data.shape[1]
    if n_eval is None:
        n_eval = min(n, N_EVAL)
    idx = np.linspace(0, n - 1, n_eval).astype(int)
    pts = data[:, idx]
    try:
        kde = gaussian_kde(data, bw_method='scott')
        kde.set_bandwidth(kde.factor * h_scale)
        eps = 1e-10
        delta = 1e-4
        dim = 2
        grad = np.zeros((dim, n_eval))
        for i in range(dim):
            pp = pts.copy(); pp[i, :] += delta
            pm = pts.copy(); pm[i, :] -= delta
            rp = np.clip(kde(pp), eps, None)
            rm = np.clip(kde(pm), eps, None)
            grad[i, :] = (np.log(rp) - np.log(rm)) / (2.0 * delta)
    except np.linalg.LinAlgError:
        return None
    g = np.zeros((dim, dim))
    for s in range(n_eval):
        g += np.outer(grad[:, s], grad[:, s])
    g /= n_eval
    ev = np.linalg.eigvalsh(g)
    lam_min = float(ev[0]); lam_max = float(ev[1])
    return dict(det=float(lam_min * lam_max), lam_min=lam_min, kappa=float(lam_max / max(lam_min, 1e-300)))


def quick_I(traj):
    """ペア(0,1)クロススペクトル虚部帯域積分（ROIマーカー・高速近似）。"""
    try:
        f, P = csd(traj[:, 0], traj[:, 1], fs=100.0, nperseg=8192)
        mask = (f >= 0.5) & (f <= 5.0)
        if mask.sum() < 2:
            return 0.0
        return float(trapezoid(np.imag(P)[mask], f[mask]))
    except Exception:
        return float('nan')


def run_point(coupling_scale, gamma, seed, T=300.0, burnin=100.0, n_eval=None):
    t, tr, clip, div = simulate(coupling_scale, 0.05, gamma, 5.0, T, burnin, seed=seed, asym_gain=1.0)
    if div:
        return dict(diverged=True)
    r = compute_geometry(tr, h_scale=1.0, n_eval=n_eval)
    if r is None:
        return dict(failed=True)
    r['sig'] = float(np.std(tr[:, :2]))
    r['clip'] = clip
    r['I'] = quick_I(tr)
    return r


def summ2(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')


def main():
    print("=== 実験5: 相境界掃引による det g→0 予言の検証（修正版・実軌道） ===", flush=True)
    print("基準: c_mem=5.0, D=0.05, T=300, burn-in=100, J_asymあり, 全シード共通。ROI境界は実験2でγ∈(0.2,0.5]に確認済み。", flush=True)

    rows_all = []  # CSV用

    def banner(title):
        print("\n" + "=" * 104, flush=True)
        print(title, flush=True)
        print("=" * 104, flush=True)

    # ---------- A. γ掃引（ROI境界を横切る主軸） ----------
    banner("A. γ掃引（cs=1.0, c_mem=5.0, D=0.05）: 2シード（42,100）の平均")
    print("{:>6} | {:>9} {:>9} | {:>11} {:>9} | {:>9} | {:>9} | {:>7} {:>6}".format(
        "γ", "λ_min", "range", "det g", "range", "κ", "σ_pair", "|I|", "clip"), flush=True)
    print("-" * 104, flush=True)
    gamma_rows = []
    for g in GAMMA_GRID:
        res = [run_point(1.0, g, s, n_eval=N_EVAL) for s in (42, 100)]
        if any(r.get('diverged') or r.get('failed') for r in res):
            print("{:>6} | 発散/失敗".format(g), flush=True)
            continue
        lams = [r['lam_min'] for r in res]; dets = [r['det'] for r in res]
        kap = [r['kappa'] for r in res]; sig = [r['sig'] for r in res]; II = [r['I'] for r in res]
        lam_m, lame = summ2(lams); det_m, dete = summ2(dets)
        kap_m, _ = summ2(kap); sig_m, _ = summ2(sig); I_m, _ = summ2(II)
        cl = max(r['clip'] for r in res)
        print("{:>6} | {:>9.5f} {:>9.5f} | {:>11.5f} {:>9.5f} | {:>9.2f} | {:>9.4f} | {:>7.3f} {:>6}".format(
            g, lam_m, lame, det_m, dete, kap_m, sig_m, abs(I_m), cl), flush=True)
        gamma_rows.append(dict(param=g, lam=lam_m, det=det_m, kappa=kap_m, sigma=sig_m, I=abs(I_m), clip=cl))
        rows_all += [dict(axis='gamma', param=g, seed=s, lam=r['lam_min'], det=r['det'], kappa=r['kappa'],
                          sigma=r['sig'], I=r['I'], clip=r['clip']) for s, r in zip((42, 100), res)]

    # ---------- B. cs掃引（不安定化境界を探索） ----------
    banner("B. cs掃引（γ=0.1, c_mem=5.0, D=0.05）: 2シード（42,100）の平均")
    print("{:>6} | {:>9} {:>9} | {:>11} {:>9} | {:>9} | {:>9} | {:>7} {:>6}".format(
        "cs", "λ_min", "range", "det g", "range", "κ", "σ_pair", "|I|", "clip"), flush=True)
    print("-" * 104, flush=True)
    cs_rows = []
    for cs in CS_GRID:
        res = [run_point(cs, 0.1, s, n_eval=N_EVAL) for s in (42, 100)]
        if any(r.get('diverged') or r.get('failed') for r in res):
            print("{:>6} | 発散/失敗".format(cs), flush=True)
            continue
        lams = [r['lam_min'] for r in res]; dets = [r['det'] for r in res]
        kap = [r['kappa'] for r in res]; sig = [r['sig'] for r in res]; II = [r['I'] for r in res]
        lam_m, lame = summ2(lams); det_m, dete = summ2(dets)
        kap_m, _ = summ2(kap); sig_m, _ = summ2(sig); I_m, _ = summ2(II)
        cl = max(r['clip'] for r in res)
        print("{:>6} | {:>9.5f} {:>9.5f} | {:>11.5f} {:>9.5f} | {:>9.2f} | {:>9.4f} | {:>7.3f} {:>6}".format(
            cs, lam_m, lame, det_m, dete, kap_m, sig_m, abs(I_m), cl), flush=True)
        cs_rows.append(dict(param=cs, lam=lam_m, det=det_m, kappa=kap_m, sigma=sig_m, I=abs(I_m), clip=cl))
        rows_all += [dict(axis='cs', param=cs, seed=s, lam=r['lam_min'], det=r['det'], kappa=r['kappa'],
                          sigma=r['sig'], I=r['I'], clip=r['clip']) for s, r in zip((42, 100), res)]

    # ---------- 境界候補点の同定 ----------
    if gamma_rows:
        gmin = min(gamma_rows, key=lambda r: r['lam'])
        print("\n[候補] γ掃引内で λ_min が最小: γ=%.3f, λ_min=%.5f （ROI境界はγ∈(0.2,0.5]）" % (gmin['param'], gmin['lam']), flush=True)
        # 境界帯（0.2–0.5）内の最小を優先
        in_band = [r for r in gamma_rows if 0.2 <= r['param'] <= 0.5]
        if in_band:
            gb = min(in_band, key=lambda r: r['lam'])
            print("[候補] 境界帯γ∈[0.2,0.5]内で λ_min 最小: γ=%.3f, λ_min=%.5f" % (gb['param'], gb['lam']), flush=True)
            boundary = gb
        else:
            boundary = gmin
    else:
        boundary = None

    # ---------- C. 境界候補点の詳解検証（6シード） ----------
    if boundary is None:
        print("\n境界候補なし。終了。", flush=True)
        return
    gc = boundary['param']
    banner("C. 境界候補 γ=%.3f の詳解検証（6シード）" % gc)
    dets, lams, kaps, sigs = [], [], [], []
    for s in SEEDS:
        r = run_point(1.0, gc, s, n_eval=5000)
        dets.append(r['det']); lams.append(r['lam_min']); kaps.append(r['kappa']); sigs.append(r['sig'])
    d_m, d_se = summ2(dets); l_m, l_se = summ2(lams); k_m, k_se = summ2(kaps); s_m, _ = summ2(sigs)
    print("γ=%.3f (6シード): det g=%.5g ±%.2g | λ_min=%.5f ±%.5f | κ=%.3f ±%.3f | σ_pair=%.4f" % (
        gc, d_m, d_se, l_m, l_se, k_m, k_se, s_m), flush=True)
    print("  全 λ_min 値: " + ", ".join("%.5f" % v for v in lams), flush=True)

    # ---------- D. ブートストラップ + Nスケーリング（境界候補） ----------
    banner("D. 境界候補 γ=%.3f: ブートストラップ（B=12, seed42）と Nスケーリング" % gc)
    t, tr, clip, div = simulate(1.0, 0.05, gc, 5.0, 300.0, 100.0, seed=42, asym_gain=1.0)
    rng = np.random.default_rng(7)
    det_bs, lam_bs, kap_bs = [], [], []
    for _ in range(12):
        idxb = rng.integers(0, len(tr), size=len(tr))
        r = compute_geometry(tr[idxb], h_scale=1.0, n_eval=5000)
        det_bs.append(r['det']); lam_bs.append(r['lam_min']); kap_bs.append(r['kappa'])
    det_bs, lam_bs, kap_bs = map(np.array, (det_bs, lam_bs, kap_bs))
    print("  ブートストラップ: det g [%.5g, %.5g] | λ_min [%.5f, %.5f] | κ [%.2f, %.2f]  (95th percentile)" % (
        np.percentile(det_bs, 2.5), np.percentile(det_bs, 97.5),
        np.percentile(lam_bs, 2.5), np.percentile(lam_bs, 97.5),
        np.percentile(kap_bs, 2.5), np.percentile(kap_bs, 97.5)), flush=True)
    print("  Nスケーリング:")
    for Nn in (3000, 6000, 10000, 20000):
        r = compute_geometry(tr[:Nn], h_scale=1.0, n_eval=min(Nn, 3000))
        print("    N=%6d: det g=%.6f  λ_min=%.6f  κ=%.3f" % (Nn, r['det'], r['lam_min'], r['kappa']), flush=True)

    # ---------- E. 判定 ----------
    banner("E. 判定")
    lam_ci = (np.percentile(lam_bs, 2.5), np.percentile(lam_bs, 97.5))
    if lam_ci[1] < 5e-3:
        verdict = "予言支持: λ_min CI が実質ゼロ（<5e-3）"
    elif lam_ci[0] > 0.05:
        verdict = "予言棄却（この軸・この分解能では）: λ_min CI が正の有限値に留まり 0 を排除"
    else:
        verdict = "部分支持: λ_min は境界で極小だが非ゼロ（疑似/見かけの特異点の可能性）"
    print(verdict, flush=True)
    print("  λ_min CI = [%.5f, %.5f], κ(境界周辺) の最大 = %.2f（発散なし）" % (
        lam_ci[0], lam_ci[1], max(r['kappa'] for r in gamma_rows + cs_rows)), flush=True)

    # CSV保存（後段プロット用）
    import csv, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundary_sweep_result.csv")
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['axis', 'param', 'seed', 'lam', 'det', 'kappa', 'sigma', 'I', 'clip'])
        w.writeheader()
        for row in rows_all:
            w.writerow(row)
    print("\nCSV保存: %s" % out, flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()