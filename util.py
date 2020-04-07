#%% Python implementation of systematic_resampling.m written by Andreas Svensson 2016


import numpy as np
import numba as nb
from numba.typed import List,Dict #numba typedList and typedDict
from scipy.stats import invwishart

njitParallel = nb.njit(parallel=True,fastmath=True)
njitSerial = nb.njit(parallel=False,fastmath=True)


'''
'''
@nb.jit()
def systematic_resampling(W,N):
    W /= np.sum(W)
    u = (1/N)*np.random.rand()
    idx = np.zeros(N)
    q = 0
    n = 0
    for i in range(N):
        while q<u:
            q += W[n]
            n +=1
        idx[i] = n
        u += 1/N

    return idx


# @njitSerial
def gibbsParam(Phi, Psi, Sigma, V, Lambda, l, T,I):
    M = np.zeros((V.shape[0],Phi.shape[0]))
    Vinv = np.linalg.solve(M,I)
    MVinv = M@Vinv
    Phibar = Phi + MVinv@M.T
    Psibar = Psi + MVinv
    Sigbar = Sigma + Vinv
    SigbarInv = np.linalg.solve(Sigbar,I)
    cov_M = Lambda+Phibar - (Psibar@SigbarInv@Psibar.T)
    cov_M_sym = 0.5*(cov_M+cov_M.T)
    Q = invwishart.pdf(cov_M_sym,df=T+l,scale=1.)
    X = np.random.randn(Phi.shape[0],V.shape[0])
    post_mean = Psibar@SigbarInv
    A =post_mean + np.linalg.cholesky(Q)@X@np.linalg.cholesky(SigbarInv)    
    return A,Q




@njitParallel
def onedim_basis(j,Li,xi):
    return np.sin(np.pi*j*(xi+Li)/(2*Li))/np.sqrt(Li)
        
@njitParallel
def basis(index,L,x):
    basis = np.zeros(index.shape[1],dtype=np.float64)
    for i in range(index.shape[1]):
        basis[i] = np.prod(onedim_basis(index[:,i],L,x))
    return basis

    

# @njitSerial
def expand(x,y):
    return np.kron(x,y)

@njitSerial
def power_kron(x,n):
    if n==0:
        if x.ndim == 1:
            y=np.ones(1,dtype=x.dtype)
        else:
            y=np.eye(1,dtype=x.dtype)
    else:
        y = x
        for _ in nb.prange(n-1):
            y = np.kron(y,x)        
    return y

@njitSerial
def eigen(index,L):
    lmb = np.zeros(index.shape[1])
    for i in range(index.shape[1]):
        lmb[i] =  np.sum((0.5*np.pi*L*index[:,i])**2)     

    return lmb

@njitSerial
def create_index(m,nbases):
    i = np.arange(nbases,dtype=np.int64)+1#start from 1
    e = np.ones(nbases,dtype=np.int64)
    index = np.zeros((m,nbases**m),dtype=np.int64)
    for j in range(m):
        index[j,:] = np.kron(np.kron(power_kron(e,j),i),power_kron(e,(m-1-j)))
    return index

@njitSerial
def spectrumRadial(eig,sf=1.,ell=1.):
    return sf*np.sqrt(2*np.pi)*ell*np.exp(-0.5*(np.pi*np.pi*ell*ell*eig*eig))





#Not implemented yet
# @njitSerial
# def spectrumMatern(eig,sf=1.,nu=1.5,ell=1.)