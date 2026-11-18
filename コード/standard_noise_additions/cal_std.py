import numpy as np, importlib.util, sys, time
spec=importlib.util.spec_from_file_location("x10","../experiment10_fixed_estimator_calibration.py"); X=importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
# 新しい作業点: (名前, λmin真値, κ)
pts=[("ROI(20シード代表)",0.1737,1.55),("一致ガウス",0.1284,2.46),("EQ",0.997,1.22),("DD",6.6,1.65),("旧ROI点",0.207,4.83)]
print("標準ノイズ版の作業点での推定器較正（ガウス、n=20000、4シード平均）",flush=True)
print("%-18s %8s %5s | %9s %7s | %9s %7s"%("点","真λmin","κ","KDE","比","kNN","比"),flush=True)
rows=[]
for name,lm,k in pts:
    ks=[];nn=[]
    for sd in X.SEEDS:
        cov,tl,th=X.true_fisher_gauss(lm,lam_max=lm*k)
        tr=X.sample_gauss(cov,X.N_SAMP,sd)
        e=X.estimate(tr); ks.append(e['kde_lam_min']); nn.append(e['knn_lam_min'])
    a=np.nanmean(ks);b=np.nanmean(nn)
    print("%-18s %8.4f %5.2f | %9.4f %7.3f | %9.4f %7.2f"%(name,lm,k,a,a/lm,b,b/lm),flush=True)
