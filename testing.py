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
T = 1000 # Number of dat points
tf = 2.5
t = np.arange(T)#*tf/T
u = 2.5-5*np.random.rand(1,T)#np.sin(2*np.pi*t/10)+np.sin(2*np.pi*t/25)
y = np.zeros((1,T)) # allocation
xt = np.zeros(2)
for i in range(T):
    y[:,i] = a.TEST_G(xt) + stdev*np.random.randn()
    xt = a.TEST_F(xt,u[:,i])

#%%
iA = np.array([[0.0146,-0.1294],
                [-0.5902,-0.2214]])
iB = np.array([[-0.1319],[0.7236]])
#%%
test_dynamic = a.Dynamic(a.TEST_F,a.TEST_G,2,1,1)
sim = a.Simulate(100,test_dynamic,u,y,7,1.0,timeStep=T)
sim.iA = np.zeros((2,2)) #iA
sim.iB = np.zeros((2,1))#iB
# %%
sim.run()

# %%
T_test = 500
t = np.arange(T_test)
u_test = np.sin(2*np.pi*t/10)+np.sin(2*np.pi*t/25)
u_test = u_test[np.newaxis,:]
y_test = np.zeros((1,T_test))
xt = np.zeros(2)
for i in range(T_test):
    y_test[:,i] = a.TEST_G(xt) #+ stdev*np.random.randn()
    xt = a.TEST_F(xt,u_test[:,i])
#%%
y_test_med,y_test_loQ,y_test_hiQ = sim.evaluate(y_test,u_test,Kn=5)

fig = plt.figure(figsize=(40,10))
plt.plot(t,y_test.T,color='k',linewidth=0.5)
plt.plot(t,y_test_med.flatten(),color='b',linewidth=0.5)
plt.fill_between(t,y_test_loQ.flatten(),y_test_hiQ.flatten(), color='b', alpha=.1)
plt.show()


# %%
