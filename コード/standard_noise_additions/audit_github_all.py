# -*- coding: utf-8 -*-
"""GitHub公開コード(Research-Public/コード)の更新式を、ソース行そのものを実行して監査する。
自由拡散(Q=0, 駆動なし): Var(psi_T)/(2 D T) を測る。標準EMなら 1。"""
import re, sys, glob, os, numpy as np
ROOT = sys.argv[1]
D, DT, T, n = 0.02, 5e-4, 5.0, 4000
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "experiment*_fixed*.py"))):
    src = open(f, encoding="utf-8").read().splitlines()
    for i, ln in enumerate(src, 1):
        if re.match(r"\s*psi\s*\+=\s*\(-Q", ln):
            code = ln.strip().split("#")[0].strip()
            np.random.seed(0)
            ns = dict(np=np, Q=np.zeros(n), D=D, D_noise=D, D_NOISE=D, DT=DT, dt=DT, noise=None, N=n, N_spin=n, psi=np.zeros(n))
            # randn(N) は n 本並列で使うため N=n。noise 変数形式 (exp1) は直前行の定義を先に実行
            pre = src[i-2].strip() if "noise =" in src[i-2] else None
            for _ in range(int(T/DT)):
                if pre: exec(pre, ns)
                exec(code, ns)
            r = ns["psi"].var(ddof=1)/(2*D*T)
            rows.append((os.path.basename(f), i, code, r))
print("%-44s %4s  %-12s %s" % ("ファイル", "行", "Var/(2DT)", "判定"))
for fn, i, c, r in rows:
    print("%-44s %4d  %-12.3e %s" % (fn, i, r, "合格" if abs(r-1) < 0.05 else "不合格"))
print("\n検査した更新式（代表）:", rows[0][2]); print("Δt²=%.3e（比/Δt² ≈ %.3f）" % (DT**2, np.mean([r for *_, r in rows])/DT**2))
