#%%
import util
import a
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%
R = 0.1
stdev = np.sqrt(R)
T = 2000 # Number of dat points
u = 2.5-5*np.random.rand(T) # input
y = np.zeros((1,T)) # allocation

xt = np.array([0.,0.])
for t in range(T):
    y[:,t] = a.TEST_G(xt) + stdev*np.random.randn()
    xt = a.TEST_F(xt,u[t])
#%%

test_dynamic = a.Dynamic(a.TEST_F,a.TEST_G,2,1,1)

# %%
