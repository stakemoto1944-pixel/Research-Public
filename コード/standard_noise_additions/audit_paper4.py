# -*- coding: utf-8 -*-
"""論文(4) 数値実験(1) のコード（修正版SPDE, 7178dbe0-____1____.py）の更新式を積分器監査にかける。
対象行: noise = sqrt(2*D*dt)*randn ; psi += (-Q + noise)*dt
Q を含む2変数系の更新を忠実に再現（step_paper4）。Q駆動 xi は検査ごとに指定。"""
import sys, numpy as np
sys.path.insert(0, ".")
from integrator_audit import step_standard

def make_step_paper4():
    # 状態 = (psi, Q)。integrator_audit の step() は psi のみなので、Qを閉包で保持。
    def run_free(D=0.05, dt=5e-4, T=10.0, n=4000, seed=0):
        rng = np.random.default_rng(seed); psi = np.zeros(n); Q = np.zeros(n)
        for _ in range(int(T/dt)):
            xi = 0.0                                   # 駆動なし → Q=0
            Q += (-2.0*Q + 1.0*xi)*dt
            noise = np.sqrt(2.0*D*dt)*rng.standard_normal(n)
            psi += (-Q + noise)*dt                     # ← 論文(4)コードの行そのまま
        return psi.var(ddof=1)/(2*D*T)
    def run_ou(theta=1.0, D=0.05, dt=5e-4, T=40.0, burn=10.0, n=2000, seed=1):
        # 復元力を持つ系: Q を使わず、同じ「psi += (-drift + noise)*dt」の形で OU を再現
        rng = np.random.default_rng(seed); psi = np.zeros(n); acc=[]; b=int(burn/dt)
        for k in range(int(T/dt)):
            noise = np.sqrt(2.0*D*dt)*rng.standard_normal(n)
            psi += (-theta*psi + noise)*dt
            if k>=b and (k-b)%200==0: acc.append(psi.var(ddof=1))
        return np.mean(acc)/(D/theta)
    return run_free, run_ou

if __name__ == "__main__":
    rf, ro = make_step_paper4()
    print("論文(4)コードの更新式の監査（解析解との比。1.000 が正しい）")
    t1 = rf(); t2 = ro()
    print("T1 自由拡散 Var/(2Dt)     : %.3e   (標準EM期待値 1)" % t1)
    print("T2 OU 定常分散 /(D/θ)     : %.3e   (標準EM期待値 1)" % t2)
    for dt in (2e-3, 1e-3, 5e-4):
        r = rf(dt=dt, T=5.0, n=3000)
        print("  Δt=%.0e: T1=%.3e   T1/Δt²=%.3f" % (dt, r, r/dt**2))
    # 実コードによる直接確認: D=0.02 と D=0 の同一シードで ψ 軌道の差からノイズ寄与の大きさを見る
    import importlib.util
    spec = importlib.util.spec_from_file_location("p4", (sys.argv[1] if len(sys.argv)>1 else "experiment1_fixed_legacy.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    D0 = 0.02; dt = m.DT
    print("\n実コード: D=%.3f, Δt=%.4g" % (D0, dt))
    print("  1ステップのノイズ増分の標準偏差（コード）: sqrt(2DΔt)·Δt = %.3e" % (np.sqrt(2*D0*dt)*dt))
    print("  標準EMの増分標準偏差                    : sqrt(2DΔt)     = %.3e" % np.sqrt(2*D0*dt))
    print("  有効拡散係数 D_eff = D·Δt = %.3e （D=%.3f の %.4f 倍）" % (D0*dt, D0, dt))
    t, a, *_ = m.simulate_corrected(T_run=5.0, D_noise=D0)
    t, b, *_ = m.simulate_corrected(T_run=5.0, D_noise=0.0)
    print("  実コードで D=0.02 と D=0 の軌道差 RMS = %.3e （ほぼ決定論）; σ(traj)=%.4f" % (np.sqrt(np.mean((a-b)**2)), a.std()))
