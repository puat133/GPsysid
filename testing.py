#%%
import util
import a
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

#%%
R = 0.1
stdev = np.sqrt(R)
T = 2000 # Number of dat points
tf = 2.5
t = np.arange(T)*tf/T
u = 2.5-5*np.random.rand(1,T)#np.sin(2*np.pi*t/10)+np.sin(2*np.pi*t/25)
y = np.zeros((1,T)) # allocation
xt = np.array([0.,0.])
for i in range(T):
    y[:,i] = a.TEST_G(xt) + stdev*np.random.randn()
    xt = a.TEST_F(xt,u[:,i])

#%%
iA = np.array([[0.0146,-0.1294],
                [-0.5902,-0.2214]])
iB = np.array([[-0.1319],[0.7236]])
#%%
test_dynamic = a.Dynamic(a.TEST_F,a.TEST_G,2,1,1)
sim = a.Simulate(500,test_dynamic,u,y,7,1.0,timeStep=T)
sim.iA = iA
sim.iB = iB
# %%
sim.run()

# %%
