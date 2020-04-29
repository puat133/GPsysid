#%%
import util
import a
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
sns.set_style('darkgrid')
from matplotlib import rc
import scipy.io as sio
import sippy
#%%
np.random.seed(1200)
data = sio.loadmat('ForIdentification.mat')
resampling = 48
ma_smoother = 24//2
tempIndex = 1
u = data['u'][::resampling,3];u=u[np.newaxis,:]
y = data['delta_T'][::resampling,tempIndex];y=y[np.newaxis,:]
y = y-y[:,-1]
u = u - np.mean(u,axis=1)
#%%
T = u.shape[1]
#%%
#Extend u and y
extension = 50 # 50% extension
y_extend = np.zeros((y.shape[0],(y.shape[1]*(100+extension)//100)))
u_extend = np.zeros((u.shape[0],(u.shape[1]*(100+extension)//100)))
shift = extension*y.shape[1]//200
y_extend[:,shift:-shift] = y
u_extend[:,shift:-shift] = u
y_ma = util.moving_average(y_extend.T,ma_smoother).T
plt.plot(y.T)
plt.plot(y_ma.T)
#%%
sys_id = sippy.system_identification(y.T,u.T,'N4SID'
                                    #  ,centering='InitVal'
                                    #  ,SS_p=horizon,SS_f=horizon
                                     ,SS_A_stability=True
                                     ,IC='BIC'
                                     ,SS_orders=[2,8]
                                     )

#%%
nx = sys_id.A_K.shape[0]
# iA = np.zeros((nx,nx))
# iB = np.zeros((nx,1))
iA = sys_id.A_K #np.random.randn(nx,nx)
iB = sys_id.B_K #np.ones((nx,1))
# iA = np.array([[0.0146,-0.1294],
#                 [-0.5902,-0.2214]])
# iB = np.array([[-0.1319],[0.7236]])
#%%
nbases=5
L = y_ma.shape[1]*10
steps = 10
sim = a.Simulate(steps,nx,u_extend,y_ma,nbases,L,PFweightNum=30)
sim.iA = np.zeros((nx,nx)) 
sim.iB = np.zeros((nx,1))
sim.burnInPercentage = 25
# %%
sim.run()

# %%
T_test = T
u_test = u_extend
# y_test = data['delta_T'][::resampling,tempIndex+7];y_test=y_test[np.newaxis,:];y_test = y_test-y_test[:,0]
y_test= y_ma

#%%
y_test_med,y_test_loQ,y_test_hiQ = sim.evaluate(y_test,u_test,Kn=5)
t = np.arange(T)
#%%
fig = plt.figure(figsize=(40,10))
plt.plot(y_test.T,color='k',linewidth=1,label='Ground Truth')
plt.plot(y_test_med[0,:],color='b',linewidth=0.5,label='Median')
# plt.fill_between(y_test_loQ[0,:],y_test_hiQ[0,:], color='b', alpha=.1, label=r'95 \% confidence')
plt.legend()
plt.show()


# %%


# %%
