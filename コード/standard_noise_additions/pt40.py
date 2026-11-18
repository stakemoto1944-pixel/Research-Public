import importlib.util, numpy as np, csv
from scipy import stats
spec=importlib.util.spec_from_file_location("e4","../experiment4_fixed_info_geometry.py");E4=importlib.util.module_from_spec(spec);spec.loader.exec_module(E4)
BASE=dict(coupling_scale=1.0,gamma=0.1,c_mem=5.0,D=0.05,T=300.0,burnin=100.0)
_,roi42,_,_=E4.simulate(**BASE,seed=42,asym_gain=1.0); cov=np.cov(roi42[:,:2].T)
S=40;R={"ROI":[],"DD":[],"GAUSS":[]}
for s in range(S):
    _,tr,_,_=E4.simulate(**BASE,seed=s,asym_gain=1.0);R["ROI"].append(E4.compute_geometry(tr[:,:2],h_scale=1.0,n_eval=5000)["lam_min"])
    _,tr,_,_=E4.simulate(1.0,0.05,2.0,5.0,300.0,100.0,seed=s,asym_gain=1.0);R["DD"].append(E4.compute_geometry(tr[:,:2],h_scale=1.0,n_eval=5000)["lam_min"])
    tr=np.random.default_rng(s).multivariate_normal(np.zeros(2),cov,size=len(roi42));R["GAUSS"].append(E4.compute_geometry(tr,h_scale=1.0,n_eval=5000)["lam_min"])
    print("seed",s,flush=True)
with open("pt40_perseed.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["seed","ROI","DD","GAUSS"])
    for s in range(S): w.writerow([s,R["ROI"][s],R["DD"][s],R["GAUSS"][s]])
a={k:np.array(v) for k,v in R.items()}
for k,v in a.items(): print(k,"n=%d mean %.4f SE %.4f median %.4f range [%.3f,%.3f]"%(S,v.mean(),v.std(ddof=1)/np.sqrt(S),np.median(v),v.min(),v.max()))
for x,y in [("ROI","DD"),("ROI","GAUSS")]:
    t=stats.ttest_ind(a[x],a[y],equal_var=False); lt=stats.ttest_ind(np.log(a[x]),np.log(a[y]),equal_var=False); u=stats.mannwhitneyu(a[x],a[y],alternative="two-sided")
    print("%s vs %s: Welch t=%.2f p=%.3g | log-Welch t=%.2f p=%.3g | MWU p=%.3g"%(x,y,t.statistic,t.pvalue,lt.statistic,lt.pvalue,u.pvalue))
for s0,s1 in [(0,20),(20,40)]:
    t=stats.ttest_ind(a["ROI"][s0:s1],a["DD"][s0:s1],equal_var=False);print("seeds %d-%d ROI vs DD Welch p=%.3g"%(s0,s1,t.pvalue))
print("DD中央値/ROI中央値 = %.1f ; ROI<DD の割合(全ペア)=%.3f"%(np.median(a["DD"])/np.median(a["ROI"]),np.mean(a["ROI"][:,None]<a["DD"][None,:])))
