import sys, importlib.util, numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "standard"
spec = importlib.util.spec_from_file_location("e4", "../experiment4_fixed_info_geometry.py")
E4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E4)
BASE = dict(coupling_scale=1.0, gamma=0.1, c_mem=5.0, D=0.05, T=300.0, burnin=100.0)
_, roi42, _, _ = E4.simulate(**BASE, seed=42, asym_gain=1.0)
cov = np.cov(roi42[:, :2].T)
res = {c: [] for c in ["ROI", "EQ", "DD", "GAUSS"]}
sig = {c: [] for c in res}
for s in range(20):
    for c in res:
        if c == "ROI": _, tr, _, d = E4.simulate(**BASE, seed=s, asym_gain=1.0)
        elif c == "EQ": _, tr, _, d = E4.simulate(**BASE, seed=s, asym_gain=0.0)
        elif c == "DD": _, tr, _, d = E4.simulate(1.0, 0.05, 2.0, 5.0, 300.0, 100.0, seed=s, asym_gain=1.0)
        else: tr = np.random.default_rng(s).multivariate_normal(np.zeros(2), cov, size=len(roi42))
        r = E4.compute_geometry(tr[:, :2], h_scale=1.0, n_eval=5000)
        res[c].append(r["lam_min"]); sig[c].append(float(np.std(tr[:, 0])))
    print("seed", s, "done", flush=True)
print(f"[{mode}] 20シード λ_min（点推定, n_eval=5000）")
for c, v in res.items():
    v = np.array(v)
    print(f"  {c:6s} λ_min 平均={v.mean():.5g} ±SE {v.std(ddof=1)/np.sqrt(len(v)):.3g}  中央値={np.median(v):.5g}  範囲[{v.min():.4g}, {v.max():.4g}]   σ1平均={np.mean(sig[c]):.4g}")
