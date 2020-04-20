#%%
import hankel
import seaborn as sns
import scipy.linalg as sla
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%%

sns.set(style="white")
sns.despine()
#Compuation of SVD for N4SID algorithm based on 
#N4SID: Subspace Algorithms for the
#Identification of Combined
#Deterministic-Stochastic Systems

#%%load data
data = sio.loadmat('../ForIdentification.mat')
u = data['u'][()]
y = data['delta_T'][()]
nu = u.shape[1]
ny = y.shape[1]

#%%Choose Horizon
horizon = 30

Uf = hankel.constructHistorical(u,horizon,past=False)
Up = hankel.constructHistorical(u,horizon,past=True)
Yp = hankel.constructHistorical(y,horizon,past=True)
Yf = hankel.constructHistorical(y,horizon,past=False)

#%% max element
max_el = min(Yp.shape[0],Yf.shape[0])
max_el
Y = np.vstack((Yp[:max_el,:].T,Yf[:max_el,:].T))
U = np.vstack((Up[:max_el,:].T,Uf[:max_el,:].T))

#%% construct Hankel Matrix
Hank = np.vstack((U,Y))

#do QR decomposition
Q,R = np.linalg.qr(Hank.T)

Rt = R.T
Qt = Q.T
 

#Identifying block matrices
Rt11 = Rt[:(2*nu+ny)*horizon,:(2*nu+ny)*horizon]
Rt21 = Rt[-ny*horizon:,:(2*nu+ny)*horizon]

#%%solving for L Matrix
Lt = sla.solve_triangular(Rt11.T,Rt21.T)
L = Lt.T

#Identifying block matrices
L1 = L[:,:(nu*horizon)]
L3 = L[:,-(ny*horizon):]

#%% do SVD
Qt1 = Qt[:(2*nu+ny)*horizon,:]
Lmodified = np.hstack((L1,np.zeros((L1.shape[0],Rt11.shape[1]-(L1.shape[1]+L3.shape[1]))),L3))
InputToSVD = Lmodified@Rt11@Qt1
svd_res = sla.svd(InputToSVD)

#%% Plotting
fig_size = (30,15)
plt.rc('figure', titlesize=20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
fig = plt.figure(figsize=fig_size)
plt.semilogy(svd_res[1],'o',markersize=5, linewidth=0.5)
plt.grid()
plt.title('Singular Value Decomposition in N4SID procedure')
plt.xlabel('no')
plt.ylabel('s')
plt.tight_layout()
plt.savefig('N4SID_svd.pdf')
plt.show()

# %%
