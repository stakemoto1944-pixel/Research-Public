#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（３）：因果同定とロバスト性検証（介入・ノイズ掃引） — 修正版

修正版SPDE（論文に整合: xi = psi**3 - psi - J_sym@psi - J_asym@psi, psi += (-Q+noise)*dt）で、
実験2で確定したROI点（γ=0.1, c_mem=5.0, cs=1.0, D=0.05, T=300, burn-in=100）を基準に:

  A.1 駆動介入 : 非対称結合 J_asym のゲイン λ を掃引 → 循環指標 |I| が λ に追従するかを検定
  A.2 対照介入 : 対称結合 J_sym のゲイン λ を掃引 → 循環は非対称項のみに追従するはず（負の対照）
  A.3 ステップ介入: λ_asym = OFF(0) vs ON(1) を同一シード（同一 J・同一ノイズパス）で対応付き比較
  B.   ノイズ掃引 : D を掃引しプラトー（∂|I|/∂D ≈ 0 の帯域）を検証

全量を 6 シード平均 ± SE で表示（旧記録は無統計・符号非単調でアーティファクトのため無効）。

実行: python3 experiment3_fixed_intervention_noise.py
"""
import numpy as np
from scipy.signal import csd
from scipy.integrate import trapezoid
from scipy import stats

N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
FS = 100.0
F1, F2 = 0.5, 5.0
NPERSEG = 256

BASE = dict(coupling_scale=1.0, gamma=0.1, c_mem=5.0, D=0.05, T=300.0, burnin=100.0)
SEEDS = [0, 1, 2, 3, 42, 100]


def simulate(coupling_scale, D, gamma, c_mem, T, burnin, seed=42, asym_gain=1.0, sym_gain=1.0):
    """修正版SPDE。J_asym に asym_gain、J_sym に sym_gain を乗じて介入を実装。"""
    np.random.seed(seed)
    steps = int(T / DT)
    J_raw = np.random.randn(N, N) / np.sqrt(N)
    J_sym = coupling_scale * 0.5 * (J_raw + J_raw.T) * sym_gain
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
    clip_hits = 0
    for t_idx in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D * DT) * np.random.randn(N)) * DT
        clip_hits += int(np.sum(np.abs(psi) > CLIP_BOUND))
        psi = np.clip(psi, -CLIP_BOUND, CLIP_BOUND)
        if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
            diverged = True
            break
        if t_idx % subsample == 0 and save_idx < saved_steps:
            history[save_idx] = psi
            save_idx += 1
    mask = saved_time >= burnin
    return saved_time[mask], history[mask], clip_hits, diverged


def phase1_integrals(traj):
    n_dim = traj.shape[1]
    integrals = np.zeros((n_dim, n_dim))
    consistency = np.zeros((n_dim, n_dim))
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            x, y = traj[:, i], traj[:, j]
            f, P = csd(x, y, fs=FS, nperseg=NPERSEG)
            m = (f >= F1) & (f <= F2)
            im = np.imag(P[m])
            integrals[i, j] = trapezoid(im, f[m])
            integrals[j, i] = -integrals[i, j]
            consistency[i, j] = max(np.mean(im > 0), np.mean(im < 0))
    return integrals, consistency


def null_q99(sigma, n_t, n_reps=5):
    q99s = []
    for _ in range(n_reps):
        tr = np.random.default_rng(0).standard_normal((n_t, N)) * sigma
        integrals, _ = phase1_integrals(tr)
        vals = np.abs(integrals[np.triu_indices(N, 1)])
        q99s.append(np.percentile(vals, 99))
    return float(np.mean(q99s))


def measure(traj):
    integrals, consistency = phase1_integrals(traj)
    vals = np.abs(integrals[np.triu_indices(N, 1)])
    up = np.triu_indices(N, 1)
    q99 = null_q99(traj.std(), len(traj))
    n_sig = int(np.sum((vals > q99) & (consistency[up] > 0.85)))
    return dict(
        I01=float(integrals[0, 1]),
        maxI=float(vals.max()),
        meanI=float(vals.mean()),
        sigma=float(traj.std()),
        q99=q99,
        n_sig=n_sig,
        roi=bool(n_sig >= 1),
    )


def run_point(asym_gain=1.0, sym_gain=1.0, D=None, seed=42):
    p = dict(BASE)
    if D is not None:
        p['D'] = D
    t, traj, clip, div = simulate(
        p['coupling_scale'], p['D'], p['gamma'], p['c_mem'], p['T'], p['burnin'],
        seed=seed, asym_gain=asym_gain, sym_gain=sym_gain,
    )
    if div:
        return None
    r = measure(traj)
    r['clip'] = clip
    return r


def summarize(rows):
    rows = [r for r in rows if r is not None]
    n = len(rows)
    I01 = np.array([r['I01'] for r in rows])
    aI01 = np.abs(I01)
    maxI = np.array([r['maxI'] for r in rows])
    roi = np.array([r['roi'] for r in rows])
    return dict(
        n=n,
        mean_I01=float(I01.mean()),
        se_I01=float(I01.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
        mean_absI=float(aI01.mean()),
        se_absI=float(aI01.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
        mean_maxI=float(maxI.mean()),
        se_maxI=float(maxI.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan'),
        roi_frac=float(roi.mean()),
    )


if __name__ == "__main__":
    print("=== 実験3: 介入・ノイズ掃引（修正版ダイナミクス・統計付き） ===", flush=True)
    print("ベースライン: γ={} c_mem={} cs={} D={} T={} burn-in={}".format(
        BASE['gamma'], BASE['c_mem'], BASE['coupling_scale'], BASE['D'], BASE['T'], BASE['burnin']), flush=True)
    print("シード: {}".format(SEEDS), flush=True)
    total_clip = 0

    # ============ A.1 駆動介入: J_asym ゲイン λ 掃引 ============
    lam_list = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    rowsA1 = {}
    for lam in lam_list:
        rowsA1[lam] = []
        for s in SEEDS:
            r = run_point(asym_gain=lam, seed=s)
            rowsA1[lam].append(r)
            if r is not None:
                total_clip += r['clip']
    print("\n" + "=" * 96, flush=True)
    print("A.1 駆動介入: J_asym ゲイン λ 掃引（J_sym=1.0 固定）", flush=True)
    print("=" * 96, flush=True)
    print("{:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>9} {:>7} | {:>8} | {:>8}".format(
        "λ", "meanI01", "±SE", "mean|I01|", "±SE", "meanmax|I|", "±SE", "ROI率", "clip計"), flush=True)
    print("-" * 96, flush=True)
    sA1 = {lam: summarize(rowsA1[lam]) for lam in lam_list}
    for lam in lam_list:
        s = sA1[lam]
        clip_sum = sum((r['clip'] for r in rowsA1[lam] if r is not None))
        print("{:>6} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>8.3f} | {:>8}".format(
            lam, s['mean_I01'], s['se_I01'], s['mean_absI'], s['se_absI'],
            s['mean_maxI'], s['se_maxI'], s['roi_frac'], clip_sum), flush=True)
    lr_abs = stats.linregress(lam_list, [sA1[la]['mean_absI'] for la in lam_list])
    lr_max = stats.linregress(lam_list, [sA1[la]['mean_maxI'] for la in lam_list])
    print("\n応答検定（mean|I01| の λ 回帰）: slope={:.4f} ± {:.4f}, p={:.4g}, r={:.3f}".format(
        lr_abs.slope, lr_abs.stderr, lr_abs.pvalue, lr_abs.rvalue), flush=True)
    print("応答検定（mean max|I| の λ 回帰）: slope={:.4f} ± {:.4f}, p={:.4g}, r={:.3f}".format(
        lr_max.slope, lr_max.stderr, lr_max.pvalue, lr_max.rvalue), flush=True)
    # 符号安定性（シード内で λ≥0.25 の全点で I01 の符号が一定か）
    stable = 0
    for k, s in enumerate(SEEDS):
        signs = [np.sign(rowsA1[la][k]['I01']) for la in lam_list[1:]]
        if signs and all(x == signs[0] for x in signs):
            stable += 1
    print("符号安定実現（λ≥0.25 で I01 の符号が一定のシード数）: {}/{}".format(stable, len(SEEDS)), flush=True)

    # ============ A.2 対照介入: J_sym ゲイン λ 掃引 ============
    lam_sym = [0.0, 0.5, 1.0, 1.5, 2.0]
    rowsA2 = {}
    for lam in lam_sym:
        rowsA2[lam] = []
        for s in SEEDS:
            r = run_point(sym_gain=lam, seed=s)
            rowsA2[lam].append(r)
            if r is not None:
                total_clip += r['clip']
    print("\n" + "=" * 96, flush=True)
    print("A.2 対照介入: J_sym ゲイン λ 掃引（J_asym=1.0 固定）→ 応答が出ないこと（負の対照）を確認", flush=True)
    print("=" * 96, flush=True)
    print("{:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>9} {:>7} | {:>8}".format(
        "λ", "meanI01", "±SE", "mean|I01|", "±SE", "meanmax|I|", "±SE", "ROI率"), flush=True)
    print("-" * 96, flush=True)
    sA2 = {lam: summarize(rowsA2[lam]) for lam in lam_sym}
    for lam in lam_sym:
        s = sA2[lam]
        print("{:>6} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>8.3f}".format(
            lam, s['mean_I01'], s['se_I01'], s['mean_absI'], s['se_absI'],
            s['mean_maxI'], s['se_maxI'], s['roi_frac']), flush=True)
    lr_ctl = stats.linregress(lam_sym, [sA2[la]['mean_absI'] for la in lam_sym])
    print("対照応答の λ 回帰: slope={:.4f} ± {:.4f}, p={:.4g}, r={:.3f}".format(
        lr_ctl.slope, lr_ctl.stderr, lr_ctl.pvalue, lr_ctl.rvalue), flush=True)

    # ============ A.3 ステップ介入（同一シード＝同一 J・同一ノイズパス） ============
    dI01, dmaxI, droi = [], [], []
    for s in SEEDS:
        off = run_point(asym_gain=0.0, seed=s)
        on = run_point(asym_gain=1.0, seed=s)
        dI01.append(on['I01'] - off['I01'])
        dmaxI.append(on['maxI'] - off['maxI'])
        droi.append(int(on['roi']) - int(off['roi']))
    dI01 = np.array(dI01); dmaxI = np.array(dmaxI)
    tI, pI = stats.ttest_rel(dI01, np.zeros(len(dI01)))
    tM, pM = stats.ttest_rel(dmaxI, np.zeros(len(dmaxI)))
    print("\n" + "=" * 96, flush=True)
    print("A.3 ステップ介入: J_asym OFF(λ=0) vs ON(λ=1.0) 同一シード対応付き（同一ノイズパス）", flush=True)
    print("=" * 96, flush=True)
    print("ΔI01  (ON−OFF): mean={:+.5f} ± {:.5f}, t={:.2f}, p={:.4g}".format(
        dI01.mean(), dI01.std(ddof=1) / np.sqrt(len(dI01)), tI, pI), flush=True)
    print("Δmax|I| (ON−OFF): mean={:+.5f} ± {:.5f}, t={:.2f}, p={:.4g}".format(
        dmaxI.mean(), dmaxI.std(ddof=1) / np.sqrt(len(dmaxI)), tM, pM), flush=True)
    print("ΔROI フラグ（ON−OFF）: 平均 {:.2f} / 最大増分 {}".format(
        np.mean(droi), max(droi)), flush=True)

    # ============ B ノイズ掃引（J_asym=1.0 固定） ============
    D_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    rowsB = {}
    for Dv in D_list:
        rowsB[Dv] = []
        for s in SEEDS:
            r = run_point(D=Dv, seed=s)
            rowsB[Dv].append(r)
            if r is not None:
                total_clip += r['clip']
    print("\n" + "=" * 96, flush=True)
    print("B ノイズ掃引: D 掃引（J_asym=1.0 固定）→ プラトー検証", flush=True)
    print("=" * 96, flush=True)
    print("{:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>9} {:>7} | {:>8}".format(
        "D", "meanI01", "±SE", "mean|I01|", "±SE", "meanmax|I|", "±SE", "ROI率"), flush=True)
    print("-" * 96, flush=True)
    sB = {d: summarize(rowsB[d]) for d in D_list}
    for d in D_list:
        s = sB[d]
        print("{:>6} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>9.5f} {:>7.5f} | {:>8.3f}".format(
            d, s['mean_I01'], s['se_I01'], s['mean_absI'], s['se_absI'],
            s['mean_maxI'], s['se_maxI'], s['roi_frac']), flush=True)
    vals = np.array([sB[d]['mean_maxI'] for d in D_list])
    for lo, hi in [(0.02, 0.05), (0.05, 0.1)]:
        sl = (vals[D_list.index(hi)] - vals[D_list.index(lo)]) / (hi - lo)
        print("プラトー斜率 ∂mean max|I|/∂D  [{}→{}]: {:.4f}".format(lo, hi, sl), flush=True)

    print("\n総クリップ回数（全評価合計）: {}".format(total_clip), flush=True)
    print("DONE", flush=True)