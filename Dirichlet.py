# Partial Differential Equations supporting functions
# The boundary condition here is a Dirichlet boundary condition
# for u(t,x) in a one dimensional domain (0,L)

import numpy as np
import scipy.linalg as sla
import numba as nb

CACHE = True
PARALLEL = False
FASTMATH = True
jitSerial = nb.jit(parallel=False,fastmath=FASTMATH,cache=CACHE)
jitParallel = nb.jit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitSerial = nb.njit(parallel=False,fastmath=FASTMATH,cache=CACHE)



@njitSerial
def basis(j,L,x):
    # if isinstance(j,np.ndarray):
    if j.ndim == 1: #j is a vector
        basis = np.sin(np.pi*j.reshape((-1, 1))*x/L)/np.sqrt(L/2)
    else: #single entry j
        basis = np.sin(np.pi*(j*x/L))/np.sqrt(L/2)

    return basis



'''
nbases is number of eigenfunction required.
L is spatial domain length
'''
@njitSerial
def eigenvalues(nbases,L):
    index = np.arange(nbases)+1
    eig = np.square((index*np.pi/L))
    return eig

'''
laplacian in the Fourier basis
'''
@njitSerial
def laplacian(nbases,L):
    eig = eigenvalues(nbases,L)
    Lap = -np.diag(eig)
    return Lap

'''
derivative in the Fourier basis
'''
@njitSerial
def derivative(nbases,L):
    der = np.zeros((nbases,nbases))
    for i in range(nbases):
        for j in range(nbases):
            if (i+j)%2 != 0:#odd case
                m = i+1
                n = j+1
                der[i,j] = (4*(m*n))/(L*(m*m - n*n))
            
    return der

@njitSerial
def construct_reference_temperature(temp_inlet,temp_outlet,positions):
    return temp_inlet + positions.reshape((-1,1))*(temp_outlet-temp_inlet) 