#%% pade approximation of some GP_covariance and spectral functions
import numpy as np
import numba as nb
# import math
from scipy.special import factorial
from scipy.interpolate import pade

'''
Pade approximant [m/n] of SE. 
'''
def se_pade(m,n,s,ell):
    order = m+n
    
    #power
    p = np.arange(order,-1,-1)

    #intermediate variable
    c = np.power((np.square(ell)/2),p)/factorial(p)

    pade_res = pade(c,m,n)

    b = pade_res[0].coefficients
    a = pade_res[1].coefficients

    A = np.zeros(2*a.shape[0]-1)
    B = np.zeros(2*b.shape[0]-1)

    A[::2] = a
    B[::2] = b

    B = B*np.square(s)*np.sqrt(np.pi)*ell

    return A,B

