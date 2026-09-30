#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文（４）数値実験（９）：Phase-1 の多重比較補正（FWER 制御）と null の再設計 — 修正版

【このスクリプトが答える査読課題】
  深掘り計画 #1（統計設計）: 旧基準「435ペア中1ペアでもペア別99%分位を超えたら ROI=True」は
  多重比較を制御していない。純白ノイズ下でも ROI=True になる確率は 1-0.99^435 = 98.7%。
  → max 統計量 max_j|I_j| の帰無分位に基づく FWER 制御で置換し、FWER を实测する。
  併せて、旧実装で発見した null_q99() の重複バグ（default_rng(0) を毎回使用＝
  「5試行平均」が実1試行の重複）を修正する。

  深掘り計画 #5（可同定性の一般性）: OFF/ON 対応比較を γ×c_mem 全格子で実行し、
  「記憶由来の受動遅延が偽の循環を作らない」ことを動作点1つではなく格子全体で示す。

【null の3設計】
  N1 iid白色      : 旧基準の null（時間相関なし）。FWER 未制御 Moth.
  N2 侨存(記憶)    : 同一 γ, c_mem, D, ノイズ経路で J_asym=0（λ_asym=0）。
                     時間相関と記憶を保ったまま循環のみを除いた「物理的に正しい」null。
  N3 陪替相ランダム化: ON 軌道の振幅スペクトルを保ち位相のみランダム化。
                     自己相関を厳密保存しつつ時間反転非対称性のみを破壊する null。

【数値の厳密性】
  I_ij は scipy.signal.csd（Welch, nperseg=256, Hann, detrend='constant'）をそのまま用いる。
  csd は x,y について双線形なので I_ij(σ) = σ^2 I_ij(1) が厳密に成り立つ（実測で確認済み）。
  このため null アンサンブル（σ=1）を一度だけ作れば、閾値は σ^2 倍して全格子で再利用できる。

実行: python3 experiment9_fixed_fwer_control.py
出力: fwer_control_result.csv（格子点ごとの判定）
"""
import csv
import numpy as np
from scipy.signal import csd
from scipy.integrate import trapezoid

# ---------- パラメータ（実験2/4/8と同一の定義） ----------
N = 30
DT = 0.0005
CLIP_BOUND = 10.0
SAVE_DT = 0.01
FS = 100.0
F1, F2 = 0.5, 5.0
NPERSEG = 256
GAMMAS = [0.1, 0.2, 0.5, 1.0, 2.0]
C_MEMS = [0.5, 1.0, 2.0, 5.0, 10.0]
COUPLING = 1.0
D_NOISE = 0.05
T_RUN = 300.0
BURNIN = 100.0
SEED_MAIN = 42
N_NULL = 400         # null アンサンブル数（max統計量の分位推定）
N_SURR = 30           # 陪替標本数（p値推定、p の下限は 1/(N_SURR+1)=0.032）
N_SURR_HI = 99        # 動作点だけ使う高精度陪替数
ALPHA = 0.05


# ---------------------------------------------------------------- シミュレータ
def simulate(gamma, c_mem, T_run, burnin, seed, asym_gain):
    """修正版SPDE。asym_gain で J_asym のみをスケール（介入のスイッチに相当）。"""
    np.random.seed(seed)
    steps = int(T_run / DT)
    J_raw = np.random.randn(N, N) / np.sqrt(N)
    J_sym = COUPLING * 0.5 * (J_raw + J_raw.T)
    np.fill_diagonal(J_sym, 0.0)
    J_raw = np.random.randn(N, N) / np.sqrt(N)
    J_asym = COUPLING * 0.5 * (J_raw - J_raw.T) * asym_gain
    psi = np.random.randn(N) * 0.01
    Q = np.zeros(N)
    subsample = int(round(SAVE_DT / DT))
    saved_steps = steps // subsample
    history = np.zeros((saved_steps, N))
    saved_time = np.linspace(0, T_run, saved_steps)
    save_idx = 0
    diverged = False
    clip = 0
    for t in range(steps):
        xi = psi ** 3 - psi - np.dot(J_sym, psi) - np.dot(J_asym, psi)
        Q += (-gamma * Q + c_mem * xi) * DT
        psi += (-Q + np.sqrt(2.0 * D_NOISE * DT) * np.random.randn(N)) * DT
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


# ------------------------------------------------------------- Phase-1 指標
_IU = np.triu_indices(N, 1)


def phase1_integrals(traj):
    """全435ペアの帯域積分 I_ij と符号一貫性（数値実験2と同一の定義）。"""
    integrals = np.zeros((N, N))
    consistency = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            f, P = csd(traj[:, i], traj[:, j], fs=FS, nperseg=NPERSEG)
            m = (f >= F1) & (f <= F2)
            im = np.imag(P[m])
            integrals[i, j] = trapezoid(im, f[m])
            integrals[j, i] = -integrals[i, j]
            if len(im) > 0:
                consistency[i, j] = max(np.mean(im > 0), np.mean(im < 0))
    return integrals, consistency


def phase1_stats(traj):
    integrals, consistency = phase1_integrals(traj)
    vals = np.abs(integrals[_IU])
    cons = consistency[_IU]
    return integrals, vals, cons


# -------------------------------------------------------------------- null
def build_null_maxquantiles(n_t, n_rep, seed_offset=100000):
    """
    iid白色 null（σ=1）での max_j|I_j| 分布。旧実装の重複バグを修正し、
    試行ごとに異なるシードを使う。
    戻り値: (maxes[n_rep], vals[n_rep,435], cons[n_rep,435])
    """
    maxes = np.empty(n_rep)
    vals_all = np.empty((n_rep, N * (N - 1) // 2))
    cons_all = np.empty((n_rep, N * (N - 1) // 2))
    for r in range(n_rep):
        rng = np.random.default_rng(seed_offset + r)
        tr = rng.standard_normal((n_t, N))
        _, vals, cons = phase1_stats(tr)
        maxes[r] = vals.max()
        vals_all[r] = vals
        cons_all[r] = cons
    return maxes, vals_all, cons_all


def fwer_threshold(sigma, q_max_unit):
    """σ スケーリング則 I(σ)=σ^2 I(1) を用いて FWER 制御下の閾値を得る。"""
    return q_max_unit * sigma ** 2


def phase_randomize(x, rng):
    """
    振幅スペクトルを保存し位相のみランダム化する陪替（Theiler 法）。
    自己相関は厳密に保存され、時間反転非対称性のみが破壊される。
    """
    X = np.fft.rfft(x)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    ph[0] = 0.0                                   # DC は実数条件を保つため固定
    if X.shape[0] % 2 == 0:
        ph[-1] = 0.0                              # Nyquist も同様に固定
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def surrogate_pvalue(traj, obs_max, n_surr=None, base_seed=500000):
    """陪替データ（循環なし）における max_j|I_j| の上側確率。"""
    if n_surr is None:
        n_surr = N_SURR
    exceed = 0
    surr_max = np.empty(n_surr)
    for s in range(n_surr):
        rng = np.random.default_rng(base_seed + s)
        tr = phase_randomize(traj, rng)
        _, vals, _ = phase1_stats(tr)
        surr_max[s] = vals.max()
        if surr_max[s] >= obs_max:
            exceed += 1
    return (exceed + 1) / (n_surr + 1), surr_max


# ------------------------------------------------------------------ メイン
def main():
    print("=" * 108, flush=True)
    print("実験9: Phase-1 多重比較補正（FWER 制御）と null 再設計", flush=True)
    print("=" * 108, flush=True)
    n_t = int(round((T_RUN - BURNIN) / SAVE_DT))
    print(f"格子: gamma x c_mem = {len(GAMMAS)}x{len(C_MEMS)}, seed={SEED_MAIN}, "
          f"T={T_RUN}, burn-in={BURNIN}, n_t={n_t}", flush=True)

    # ---------- 0. null アンサンブル ----------
    # 閾値用と FWER 実測用に独立した null アンサンブルを使い分ける
    # （同一標本で閾値を作ると FWER を過小評価する）。
    print(f"\n[0] iid白色 null アンサンブル構築中 "
          f"（閾値用 {N_NULL} + 実測用 {N_NULL} realization）...", flush=True)
    maxes_a, pooled_a, _ = build_null_maxquantiles(n_t, N_NULL, seed_offset=100000)
    maxes_b, pooled_b, cons_b = build_null_maxquantiles(n_t, N_NULL, seed_offset=700000)
    q_max = float(np.percentile(maxes_a, 100 * (1 - ALPHA)))
    q99_pooled = float(np.percentile(pooled_a, 99))
    print(f"    max_j|I_j| の {100*(1-ALPHA):.0f}%分位（σ=1） = {q_max:.6f}", flush=True)
    print(f"    参考: ペア別 |I| の 99%分位（σ=1, {N_NULL*435} 標本） = {q99_pooled:.6f}", flush=True)
    print(f"    参考: 旧基準の FWER 理論値 = 1-0.99^435 = {1-0.99**435:.4f}", flush=True)

    # 実測 FWER: 独立した null アンサンブル B に、アンサンブル A 由来の閾値を適用
    old_flags = 0
    new_flags = 0
    for r in range(N_NULL):
        old_flags += int(np.sum((pooled_b[r] > q99_pooled) & (cons_b[r] > 0.85)) >= 1)
        new_flags += int(maxes_b[r] >= q_max)
    print(f"    旧基準の実測 FWER（純白ノイズ {N_NULL} 試行）= {old_flags/N_NULL:.4f} "
          f"（目標 {ALPHA}）", flush=True)
    print(f"    新基準の実測 FWER（max統計量, 同 {N_NULL} 試行）= {new_flags/N_NULL:.4f} "
          f"（目標 {ALPHA}）", flush=True)

    rows = []
    print("\n" + "=" * 108, flush=True)
    print("格子走査: ON（λ_asym=1）と OFF（λ_asym=0、記憶あり）の対応比較", flush=True)
    print("=" * 108, flush=True)
    print("{:>5} {:>6} | {:>10} {:>10} | {:>9} {:>9} | {:>6} {:>6} | {:>9} {:>8}".format(
        "γ", "c_mem", "maxI_ON", "maxI_OFF", "q99×σ²", "qmax×σ²", "旧ROI", "新ROI",
        "p_surr", "σ_pair"), flush=True)
    print("-" * 108, flush=True)

    for gi, gamma in enumerate(GAMMAS):
        for mi, cm in enumerate(C_MEMS):
            t_on, tr_on, clip_on, div_on = simulate(gamma, cm, T_RUN, BURNIN, SEED_MAIN, 1.0)
            t_off, tr_off, clip_off, div_off = simulate(gamma, cm, T_RUN, BURNIN, SEED_MAIN, 0.0)
            if div_on or div_off:
                print(f"{gamma:>5} {cm:>6} | diverged", flush=True)
                continue
            sig_on = float(tr_on.std())
            _, v_on, c_on = phase1_stats(tr_on)
            _, v_off, _ = phase1_stats(tr_off)
            max_on, max_off = float(v_on.max()), float(v_off.max())
            thr99 = q99_pooled * sig_on ** 2
            thrmax = fwer_threshold(sig_on, q_max)
            n_sig_old = int(np.sum((v_on > thr99) & (c_on > 0.85)))
            roi_old = bool(n_sig_old >= 1)
            roi_new = bool(max_on > thrmax)
            # 動作点（論文の ROI 条件 γ=0.1, c_mem=5.0）だけは陪替を増やして
            # p 値の解像度水浒を 1/(M+1) 以下にしない
            n_surr_pt = N_SURR_HI if (gamma == 0.1 and cm == 5.0) else N_SURR
            p_surr, _ = surrogate_pvalue(tr_on, max_on, n_surr=n_surr_pt,
                                          base_seed=500000 + 1000 * gi + mi)
            rows.append(dict(
                gamma=gamma, c_mem=cm, seed=SEED_MAIN, sigma_pair=sig_on,
                maxI_on=max_on, maxI_off=max_off, thr_q99=thr99, thr_qmax=thrmax,
                n_sig_old=n_sig_old, roi_old=roi_old, roi_new=roi_new,
                p_surr=p_surr, n_surr=n_surr_pt, clip_on=clip_on, clip_off=clip_off,
                q_max_unit=q_max, q99_pooled_unit=q99_pooled, n_null=N_NULL,
            ))
            print("{:>5} {:>6} | {:>10.5f} {:>10.5f} | {:>9.5f} {:>9.5f} | {:>6} {:>6} | "
                  "{:>9.4f} {:>8.4f}".format(
                      gamma, cm, max_on, max_off, thr99, thrmax,
                      str(roi_old), str(roi_new), p_surr, sig_on), flush=True)

    out = "fwer_control_result.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n書き出し: {out}  ({len(rows)} 行)", flush=True)

    # ---------- まとめ ----------
    old_roi = [r for r in rows if r["roi_old"]]
    new_roi = [r for r in rows if r["roi_new"]]
    print("\n" + "=" * 108, flush=True)
    print("まとめ", flush=True)
    print("=" * 108, flush=True)
    print(f"旧基準 ROI=True の格子点: {len(old_roi)}/{len(rows)}", flush=True)
    print(f"新基準 ROI=True の格子点: {len(new_roi)}/{len(rows)}", flush=True)
    for r in new_roi:
        print(f"   γ={r['gamma']}, c_mem={r['c_mem']}, p_surr={r['p_surr']:.4f}", flush=True)
    print(f"\nOFF 状態（記憶あり・循環なし）の max|I| 最大値: "
          f"{max(r['maxI_off'] for r in rows):.6f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
