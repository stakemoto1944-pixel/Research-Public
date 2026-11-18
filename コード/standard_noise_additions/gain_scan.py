import numpy as np, importlib.util, csv
import lyap as L
spec=importlib.util.spec_from_file_location("e9","exp9_grid_only.py");E9=importlib.util.module_from_spec(spec);spec.loader.exec_module(E9)
spec=importlib.util.spec_from_file_location("e4","../experiment4_fixed_info_geometry.py");E4=importlib.util.module_from_spec(spec);spec.loader.exec_module(E4)
GAINS=[0.0,0.25,0.5,0.75,1.0,1.5];SEEDS=[0,1,2,3,4,5]
print("γ=0.1, c_mem=5, D: λ1 は D=0, 他は D=0.05（標準ノイズ）",flush=True)
print("%5s | %16s %9s | %16s %14s"%("gain","λ1(D=0) 平均±SE","正の割合","λmin 平均±SE","max|I| 平均"),flush=True)
rows=[]
for g in GAINS:
    ly=np.array([L.run(s,0.1,5.0,g) for s in SEEDS]); lm=[];mi=[]
    for s in SEEDS:
        _,tr,_,_=E4.simulate(1.0,0.05,0.1,5.0,300.0,100.0,seed=s,asym_gain=g)
        lm.append(E4.compute_geometry(tr[:,:2],h_scale=1.0,n_eval=3000)["lam_min"])
        _,v,_=E9.phase1_stats(tr); mi.append(v.max())
    lm=np.array(lm);mi=np.array(mi)
    print("%5.2f | %+7.3f ± %5.3f %9.2f | %8.4f ± %6.4f %14.4f"%(g,ly.mean(),ly.std(ddof=1)/np.sqrt(6),(ly>0).mean(),lm.mean(),lm.std(ddof=1)/np.sqrt(6),mi.mean()),flush=True)
    rows.append([g,ly.mean(),(ly>0).mean(),lm.mean(),lm.std(ddof=1)/np.sqrt(6),mi.mean()])
csv.writer(open("gain_scan_result.csv","w",newline="")).writerows([["gain","lyap_mean","lyap_pos_frac","lammin_mean","lammin_se","maxI_mean"]]+rows)
