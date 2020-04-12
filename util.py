#%% Python implementation of systematic_resampling.m written by Andreas Svensson 2016


import numpy as np
import numba as nb
from numba.typed import List,Dict #numba typedList and typedDict
from scipy.stats import invwishart

jitSerial = nb.jit(fparallel=False,astmath=True)
jitParallel = nb.jit(parallel=True,fastmath=True)
njitParallel = nb.njit(parallel=True,fastmath=True)
njitSerial = nb.njit(parallel=False,fastmath=True)


'''
'''
@njitSerial
def systematic_resampling(W,N):
    W /= np.sum(W)
    u = (1/N)*np.random.rand()
    idx = np.zeros(N,dtype=np.int64)
    q = 0
    n = 0
    for i in nb.prange(N):
        while q<u:
            q += W[n]
            n +=1
        idx[i] = n#python start from 0
        u += 1/N

    return idx-1


#based on eq. 12 and 13 of "A flexible state–space model for learning nonlinear
# dynamical systems"  http://dx.doi.org/10.1016/j.automatica.2017.02.030
@njitSerial
def gibbParamPre(Phi, Psi, Sigma, vdiag, Lambda,I):
    # M = np.zeros((vdiag.shape[0],Phi.shape[0])) #this is stupid
    # MVinv = M/vdiag #@Vinv <--M is zero matrix
    Phibar = Phi # + MVinv@M.T <--M is zero matrix
    Psibar = Psi # + MVinv <--M is zero matrix
    Sigbar = Sigma + np.diag(1/vdiag)
    SigbarInv = np.linalg.solve(Sigbar,I)
    cov_M = Lambda+Phibar - (Psibar@SigbarInv@Psibar.T)
    cov_M_sym = 0.5*(cov_M+cov_M.T)
    return cov_M_sym,Psibar,SigbarInv


@njitSerial
def gibbParamPost(Phi,vdiag,Psibar,SigbarInv,Q):
    X = np.random.randn(Phi.shape[0],vdiag.shape[0])
    post_mean = Psibar@SigbarInv
    A =post_mean + np.linalg.cholesky(Q)@X@np.linalg.cholesky(SigbarInv)    
    return A


#V is assumed to be diagonal
# @jitSerial
def gibbsParam(Phi, Psi, Sigma, vdiag, Lambda, l, T,I):
    cov_M_sym,Psibar,SigbarInv = gibbParamPre(Phi, Psi, Sigma, vdiag, Lambda,I)
    Q = invwishart.rvs(T+l,cov_M_sym)
    A = gibbParamPost(Phi,vdiag,Psibar,SigbarInv,Q)
    return A,Q
    




@njitParallel
def onedim_basis(j,Li,xi):
    return np.sin(np.pi*j*(xi.T+Li)/(2*Li))/np.sqrt(Li)
        
@njitParallel
def basis(index,L,x):
    basis = np.ones((index.shape[1],x.shape[1]),dtype=np.float64)
    for i in nb.prange(index.shape[1]):
        temp = onedim_basis(index[:,i],L,x)
        for j in nb.prange(temp.shape[1]):
            basis[i,:] *= temp[:,j]
    return basis

    

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
    for i in nb.prange(index.shape[1]+1):#This start from one
        lmb[i] =  np.sum((0.5*np.pi*index[:,i]/L)**2)     

    return lmb

@njitSerial
def create_index(m,nbases):
    i = np.arange(nbases,dtype=np.int64)+1#start from 1
    e = np.ones(nbases,dtype=np.int64)
    index = np.zeros((m,nbases**m),dtype=np.int64)
    for j in nb.prange(m):
        index[j,:] = np.kron(np.kron(power_kron(e,j),i),power_kron(e,(m-1-j)))
    return index

# S_SE = @(w,ell) sqrt(2*pi.*ell.^2).*exp(-(w.^2./2).*ell.^2);
# V = 1000*diag(prod(S_SE(sqrt(lambda),1),2));
@njitSerial
def spectrumRadial(eig,sf=1.,ell=1.):
    return sf*np.sqrt(2*np.pi)*ell*np.exp(-0.5*(ell*ell*eig*eig))


@njitSerial
def evaluate_latest_model(iA,iB,A,index,L,x,u):
    uExpand = u*np.ones((1,x.shape[1]))
    xAug = np.vstack((x,uExpand))
    return iA@x + iB@uExpand+ A@basis(index,L,xAug)

@njitSerial
def compute_Phi_Psi_Sig(iA,iB,x_prim,index,L,u):
    #compute statistics
    linear_part = iA@x_prim[:,:-1] + iB@u[:,:-1]
    zeta = x_prim[:,1:] - linear_part
    xAug = np.vstack((x_prim,u))
    z = basis(index,L,xAug[:,:-1])
    Phi = zeta@zeta.T#np.outer(zeta,zeta)
    Psi = zeta@z.T#np.outer(zeta,z)
    Sig = z@z.T#np.outer(z,z)

    return Phi,Psi,Sig



        





#Not implemented yet
# @njitSerial
# def spectrumMatern(eig,sf=1.,nu=1.5,ell=1.)