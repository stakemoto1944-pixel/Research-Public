"""
実験12: ブロック長感度（$n_eval$=5000 = 論文と同一の解像度で 再測定）

診断: 論文の報告 CI [0.190, 0.208]（B=12, seed42, n_eval=5000）と
B=1000 ブロック bootstrap（seed42）の差が B に由来するのかを分解する。
raw replicate を保存し、各种 B の percentile 区間を後計算する。
"""
import importlib.util, os
import numpy as np
from multiprocessing import Pool

H = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("e4", os.path.join(H, "experiment4_fixed_info_geometry.py"))
E4 = importlib.util.module_from_spec(sp); sp.loader.exec_module(E4)
sp11 = importlib.util.spec_from_file_location("e11", os.path.join(H, "experiment11_fixed_bootstrap_ci.py"))
E11 = importlib.util.module_from_spec(sp11); sp11.loader.exec_module(E11)

def _init(d):
    E11._init(d)   # E11 側のグローバル _TRAJ を初期化
def _job(t):
    return E11._one(t)

if __name__ == "__main__":
    SEEDS = [0, 1, 42]
    B = 150
    trs = {}
    for s in SEEDS:
        _, tr, _, _ = E4.simulate(1.0, 0.05, 0.1, 5.0, 300.0, 100.0, seed=s, asym_gain=1.0)
        trs[("ROI", s)] = tr[:, :2]
    print("ブロック長感度（n_eval=5000 = 論文と同一, B={}, {}シード）".format(B, len(SEEDS)),
          flush=True)
    print("{:>6} {:>26} {:>26} {:>8}".format("L", "平均 CI [2.5,97.5]", "平均 CI 幅", "L=1比"),
          flush=True)
    base = None
    for L in [1, 64, 256]:
        tasks = [("ROI", s, L, 20_000_000 + 1000 * L + b, 5000)
                 for s in SEEDS for b in range(B)]
        with Pool(3, initializer=_init, initargs=(trs,)) as p:
            out = p.map(_job, tasks, chunksize=4)
        k = 0; los, his, ws = [], [], []
        for s in SEEDS:
            v = np.array([o["lam"] for o in out[k:k+B]]); k += B
            lo, hi = np.percentile(v, [2.5, 97.5]); los.append(lo); his.append(hi)
            ws.append(hi - lo)
        m = float(np.mean(ws))
        if L == 1:
            base = m
        print("{:>6} {:>26} {:>26} {:>8}".format(
            L, "[{:.4f}, {:.4f}]".format(np.mean(los), np.mean(his)),
            "{:.4f}".format(m), "{:.2f}x".format(m/base)), flush=True)
