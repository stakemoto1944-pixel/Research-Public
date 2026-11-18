import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='rerun_standard/'
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150})
O='figs/'
# Fig1 screening
gam=[0.1,0.2,0.5,1.0,2.0]
mx={0.1:[.04479,.17325,.68791,1.39306,1.22916],0.2:[.01844,.05756,.22894,.86362,1.14114],
 0.5:[.00109,.00771,.07772,.37,.76776],1.0:[.00081,.00145,.00315,.12084,.49216],2.0:[.00076,.00095,.00166,.00375,.0099]}
roi={0.1:[1]*5,0.2:[1]*5,0.5:[0,0,1,1,1],1.0:[0,0,0,1,1],2.0:[0]*5}
cm=[0.5,1,2,5,10]
fig,ax=plt.subplots(figsize=(7,4))
for k,g in enumerate(gam):
    v=np.array(mx[g]);f=np.array(roi[g],bool)
    ax.semilogy(cm,v,'-',color=f'C{k}',lw=1,alpha=.6)
    ax.semilogy(np.array(cm)[f],v[f],'o',color=f'C{k}',label=f'γ={g}')
    ax.semilogy(np.array(cm)[~f],v[~f],'o',mfc='white',color=f'C{k}')
ax.set_xscale('log');ax.set_xlabel('c_mem');ax.set_ylabel('max|I|')
ax.set_title('Stage 1 screening (filled: ROI flag True, open: False)')
ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(O+'Fig1_Phase1_screening_ROI.png',dpi=200)
# Fig2 four conditions
names=['ROI','EQ','DD','Gauss']
lam20=[0.1737,1.063,6.48,0.1294];se20=[0.0067,0.020,1.63,0.0006]
kap=[1.551,1.219,1.654,2.458];kse=[.237,.040,.174,.038]
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.8))
a.bar(names,lam20,yerr=se20,color=['C3','C0','C2','C7'],capsize=3)
a.set_yscale('log');a.set_ylabel('λ_min (KDE-Fisher)');a.set_title('λ_min, 20 seeds (mean ± SE)')
for i,v in enumerate(lam20): a.text(i,v*1.15,f'{v:.3g}',ha='center',fontsize=8)
a.set_ylim(0.05,20)
b.bar(names,kap,yerr=kse,color=['C3','C0','C2','C7'],capsize=3)
b.set_ylabel('κ');b.set_title('Condition number, 6 seeds');b.set_ylim(0,3.2)
fig.tight_layout();fig.savefig(O+'Fig2_FourConditions_metric.png',dpi=200)
# Fig3 sweeps
g=[.05,.1,.2,.3,.5,.75,1,1.5,2];lm=[.0996,.147,.252,.365,.471,.622,.948,4.48,12.5]
cs=[.2,.4,.6,.8,1,1.2,1.4,1.6,1.8];lc=[.863,.497,.388,.244,.147,.144,.117,.121,.078]
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.8))
a.semilogy(g,lm,'o-',color='C3');a.set_xlabel('γ');a.set_ylabel('λ_min');a.set_title('λ_min vs γ (c_s=1.0, 2 seeds)')
a.axvspan(0.2,0.5,color='gray',alpha=.15)
b.semilogy(cs,lc,'o-',color='C3');b.set_xlabel('c_s');b.set_title('λ_min vs c_s (γ=0.1)')
fig.tight_layout();fig.savefig(O+'Fig3_BoundarySweeps.png',dpi=200)
# Fig4 PCA
d=pd.read_csv(R+'高次元計量_result.csv')
m=d.groupby(['gamma','d']).lam.mean().unstack()
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.8))
for i,gm in enumerate(m.index): a.semilogy(m.columns,m.loc[gm],'o-',color=f'C{i}',label=f'γ={gm}')
a.set_xlabel('manifold dimension d');a.set_ylabel('λ_min(g_d)');a.legend(fontsize=7);a.set_title('all positive')
for gm,v3,v5 in [(0.1,1.04,.68),(0.2,1.04,.59),(0.5,1.03,.57)]:
    pass
gg=[0.1,0.2,0.5];b.plot(gg,[1.04,1.04,1.03],'o-',label='d=3');b.plot(gg,[.68,.59,.57],'s-',label='d=5')
b.axhline(1,color='k',lw=.8,ls='--');b.set_ylim(0,1.3);b.set_xlabel('γ');b.set_ylabel('λ_min emp / matched Gaussian');b.legend();b.set_title('seed 42')
fig.tight_layout();fig.savefig(O+'Fig4_HighDim_PCA.png',dpi=200)
# Fig5 axes
a_=pd.read_csv(R+'境界軸位相_result.csv')
fig,(a,b,c)=plt.subplots(1,3,figsize=(12,3.8))
A=a_[a_.axis=='A']
for gm in sorted(A.gamma.unique()):
    s=A[A.gamma==gm].groupby('D').lam_min.mean()
    a.semilogy(s.index,s.values,'o-',label=f'γ={gm}')
a.set_xscale('log');a.set_xticks([.05,.02,.01,.005]);a.set_xticklabels(['0.05','0.02','0.01','0.005']);a.minorticks_off();a.invert_xaxis();a.set_xlabel('D (→0)');a.set_ylabel('λ_min');a.legend(fontsize=8);a.set_title('(a) D→0')
B=a_[a_.axis=='B'].sort_values('c_mem')
b.semilogx(B.c_mem,B.lam_min,'o-',color='C3');b.set_xlabel('c_mem');b.set_title('(b) c_mem axis (γ=0.1)')
C=a_[a_.axis=='C'].groupby('gamma')[['g_tt','g_tt_G']].mean()
c.plot(C.index,C.g_tt,'o-',label='g_θθ');c.plot(C.index,C.g_tt_G,'s--',label='matched Gaussian')
c.set_xlabel('γ');c.set_title('(c) angular direction');c.legend(fontsize=8)
fig.tight_layout();fig.savefig(O+'Fig5_BoundaryAxes_Phase.png',dpi=200)
print(C)
