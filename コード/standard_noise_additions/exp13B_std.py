import importlib.util, numpy as np
spec=importlib.util.spec_from_file_location("e4","../experiment4_fixed_info_geometry.py");E4=importlib.util.module_from_spec(spec);spec.loader.exec_module(E4)
M_GRID=[3000,5000,8000,12000,16000,20000];SEEDS=[0,1,2,3,4,42]
tr={}
for s in SEEDS:
    _,t,_,_=E4.simulate(1.0,0.05,0.1,5.0,300.0,100.0,seed=s,asym_gain=1.0); tr[s]=t
print("λmin の標本数 M 依存（標準ノイズ, ROI, 6シード, n_eval=5000）",flush=True)
print("%7s %10s %9s %8s %8s %s"%("M","平均","SE","最小","最大","ゼロ除外"),flush=True)
means=[]
for m in M_GRID:
    l=np.array([E4.compute_geometry(tr[s][:m],h_scale=1.0,n_eval=5000)["lam_min"] for s in SEEDS])
    means.append(l.mean())
    print("%7d %10.5f %9.5f %8.5f %8.5f %s"%(m,l.mean(),l.std(ddof=1)/np.sqrt(len(l)),l.min(),l.max(),l.min()>0),flush=True)
print("比 M=20000/M=3000 = %.2f ; 単調=%s"%(means[-1]/means[0],all(b>=a for a,b in zip(means,means[1:]))))
