#%%
import util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

#some test functions
@util.njitSerial
def test_f(x,u):
    #[(x(1)/(1+x(1)^2))*sin(x(2)); x(2)*cos(x(2)) + x(1)*exp(-(x(1)^2+x(2)^2)/8) + u^3/(1+u^2+0.5*cos(x(1)+x(2)))];
    return np.array([
                    np.sin(x[1])*x[0]/(1.+x[0]*x[0]),
                    x[1]*np.cos(x[1]) + x[0]*np.exp(-(x[0]*x[0]+x[1]*x[1])/8) + u*u*u/(1.+ u*u + 0.5*np.cos(x[0]+x[1]))
                    ])
@util.njitSerial
def test_g(x):
    #x(1)/(1+0.5*sin(x(2))) + x(2)/(1+0.5*sin(x(1)));
    return np.array([x[0]/(1+0.5*np.sin(x[1]))  + x[1]/(1.+0.5*np.sin(x[0]))])



#%%
R = 0.1
stdev = np.sqrt(R)
T = 2000 # Number of dat points
u = 2.5-5*np.random.rand(T) # input
y = np.zeros((1,T)) # allocation

xt = np.array([0.,0.])
for t in range(T):
    y[:,t] = test_g(xt) + stdev*np.random.randn()
    xt = test_f(xt,u[t])




# %%
