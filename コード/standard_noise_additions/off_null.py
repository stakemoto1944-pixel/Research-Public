import numpy as np, importlib.util, csv, sys
spec=importlib.util.spec_from_file_location("e9","exp9_grid_only.py");E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
pts=[(0.1,5.0),(0.1,10.0),(0.2,10.0),(0.5,10.0)]
N_OFF,N_ON=40,10
print("iid null (400) を構築...",flush=True)
mx,_,_=E.build_null_maxquantiles(20000,400)
qmax=mx.max(); q99=np.quantile(mx,0.99); print("qmax(unit)=%.5f q99=%.5f"%(qmax,q99),flush=True)
rows=[]
for g,c in pts:
    off=[];on=[];s2=[]
    for k in range(N_OFF):
        _,tr,_,_=E.simulate(g,c,300.0,100.0,1000+k,0.0); _,v,_=E.phase1_stats(tr); off.append(v.max()); s2.append(tr.var(axis=0).mean())
    for k in range(N_ON):
        _,tr,_,_=E.simulate(g,c,300.0,100.0,2000+k,1.0); _,v,_=E.phase1_stats(tr); on.append(v.max())
    off=np.array(off);on=np.array(on);s2=np.mean(s2)
    thr_iid=qmax*s2; thr99=q99*s2; emp95=np.quantile(off,0.95); empmax=off.max()
    print("γ=%.1f c=%.1f σ²=%.3f | OFF max|I| mean %.4f (min %.4f, max %.4f) | ON mean %.4f (min %.4f) | iid閾値(qmax) %.4f 超過OFF %d/%d | 経験null 95%%点 %.4f, ON>経験max %d/%d"%(
        g,c,s2,off.mean(),off.min(),off.max(),on.mean(),on.min(),thr_iid,(off>thr_iid).sum(),N_OFF,emp95,(on>empmax).sum(),N_ON),flush=True)
    rows.append([g,c,s2,off.mean(),off.min(),off.max(),on.mean(),on.min(),thr_iid,int((off>thr_iid).sum()),N_OFF,emp95,int((on>empmax).sum()),N_ON])
with open("OFF_null_result.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow("gamma c_mem sigma2 off_mean off_min off_max on_mean on_min thr_iid off_exceed n_off emp95 on_exceed n_on".split());w.writerows(rows)
