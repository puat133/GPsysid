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
resampling = 25
tempIndex = 0
u = data['u'][::resampling,3];u=u[np.newaxis,:]
y = data['delta_T'][::resampling,tempIndex];y=y[np.newaxis,:]
y = y-y[:,0]
T = u.shape[1]
#%%
sys_id = sippy.system_identification(y.T,u.T,'N4SID'
                                    #  ,centering='InitVal'
                                    #  ,SS_p=horizon,SS_f=horizon
                                     ,SS_A_stability=True
                                     ,IC='BIC'
                                     ,SS_orders=[3,5]
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
L = 300
sim = a.Simulate(40,nx,u,y,nbases,L)
sim.iA = iA#np.zeros((2,2)) 
sim.iB = iB#np.zeros((2,1))
sim.burnInPercentage = 25
# %%
sim.run()

# %%
T_test = T
u_test = u
y_test = data['delta_T'][::resampling,tempIndex+7];y_test=y_test[np.newaxis,:];y_test = y_test-y_test[:,0]
y_test= y

#%%
y_test_med,y_test_loQ,y_test_hiQ = sim.evaluate(y_test,u_test,Kn=5)
t = np.arange(T)
fig = plt.figure(figsize=(40,10))
plt.plot(t,y_test.T,color='k',linewidth=1,label='Ground Truth')
plt.plot(t,y_test_med.flatten(),color='b',linewidth=0.5,label='Median')
plt.fill_between(t,y_test_loQ.flatten(),y_test_hiQ.flatten(), color='b', alpha=.1, label=r'95 \% confidence')
plt.legend()
plt.show()


# %%


# %%
