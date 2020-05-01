#%% Python implementation of systematic_resampling.m written by Andreas Svensson 2016


import numpy as np
import numba as nb
from numba.typed import List,Dict #numba typedList and typedDict
from scipy.stats import invwishart
import scipy.fftpack as FFT

_LOG_2PI = np.log(2 * np.pi)
CACHE = True
PARALLEL = False
FASTMATH = True
jitSerial = nb.jit(parallel=False,fastmath=FASTMATH,cache=CACHE)
jitParallel = nb.jit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitSerial = nb.njit(parallel=False,fastmath=FASTMATH,cache=CACHE)


'''
OK
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
        idx[i] = n
        u += 1/N

    return idx-1#python start from 0


'''
Note: Matlab cholesky is transpose of numpy cholesky
'''
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

'''
Note: Matlab cholesky is transpose of numpy cholesky
'''
@njitSerial
def gibbParamPost(Phi,vdiag,Psibar,SigbarInv,Q):
    X = np.random.randn(Phi.shape[0],vdiag.shape[0])
    post_mean = Psibar@SigbarInv
    A =post_mean + np.linalg.cholesky(Q).T@X@np.linalg.cholesky(SigbarInv).T
    return A


#V is assumed to be diagonal
#@jitSerial
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

@njitParallel
def basis1D(index,L,x):
    basis = np.ones((index.shape[1]),dtype=np.float64)
    for i in nb.prange(index.shape[1]):
        basis[i] = np.prod(onedim_basis(index[:,i],L,x))
        # temp = onedim_basis(index[:,i],L,x)
        # for j in nb.prange(temp.shape[1]):
            # basis[i] *= temp[j]
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
    for i in nb.prange(index.shape[1]):
        lmb[i] =  np.sum(np.square((0.5*np.pi*index[:,i]/L)))    
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
def spectrumRadial(s,sf=1.,ell=1.):
    return sf*np.sqrt(2*np.pi)*ell*np.exp(-0.5*(ell*ell*np.square(s)))


@njitSerial
def evaluate_model(iA,iB,A,index,L,x,u):
    # uExpand = np.repeat(u,x.shape[1],axis=1)#u*np.ones((u.shape[0],x.shape[1]))
    xAug = np.vstack((x,u))
    return iA@x + iB@u+ A@basis(index,L,xAug)

@njitSerial
def evaluate_model_thin(iA,iB,A,index,L,x,u):
    xAug = np.concatenate((x,u))
    return iA@x + iB@u+ A@(basis1D(index,L,xAug))

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


@njitSerial 
def _logpdf(x, mean, Sigma):
    k = x.shape[0]
    z = x - mean
    C = np.linalg.cholesky(Sigma)
    log_det_cov = 2*np.sum(np.log(np.diag(C)))
    maha = np.sum(np.square(np.linalg.solve(C.T,z)), axis=0)
    return -0.5 * (k * _LOG_2PI + log_det_cov + maha)

@njitSerial
def mvnpdf(x,mean,Sigma): 
    return np.exp(_logpdf(x,mean,Sigma))

'''
Note: Matlab cholesky is transpose of numpy cholesky
'''
# @njitParallel
def runParticleFilter(Q,timeStep,k,a,PFweight,PFweightNum,iA,iB,A,index,xPF,nx,L,u,y,R):
    # N = a.shape[1]
    ny = y.shape[0]
    Qchol = np.linalg.cholesky(Q).T
    # for t in trange(timeStep,desc='SMC - {} th'.format(k+1)):
    for t in nb.prange(timeStep):
        if t>=1:
            uTile = np.tile(u[:,t-1][:,np.newaxis],PFweightNum)
            if k>0:
                a[t,:-1] = systematic_resampling(PFweight[t-1,:],PFweightNum-1)
                f = evaluate_model(iA,iB,A,index,L,xPF[a[t,:-1],:,t-1].T,uTile[:,:-1]) 
                
                xPF[:-1,:,t] = (f + Qchol@np.random.randn(nx,PFweightNum-1)).T
                
                f = evaluate_model(iA,iB,A,index,L,xPF[:,:,t-1].T,uTile)
                
                waN = PFweight[t-1,:]*mvnpdf(f,xPF[-1,:,t].reshape((-1,1)),Q)
                
                waN /= np.sum(waN)
                a[t,-1] = systematic_resampling(waN,1)
            else:
                a[t,:] = systematic_resampling(PFweight[t-1,:],PFweightNum)
                f = evaluate_model(iA,iB,A,index,L,xPF[a[t,:],:,t-1].T,uTile)
                xPF[:,:,t] = (f + Qchol@np.random.randn(nx,PFweightNum)).T


        log_w = np.sum(-0.5*np.square(xPF[:,-ny:,t]-y[:,t])/R,axis=1)
        PFweight[t,:] = np.exp(log_w-np.max(log_w)).flatten()
        PFweight[t,:] /= np.sum(PFweight[t,:])

    return a,PFweight,xPF







'''
compute cross corellation between two signals
using FFT
'''
def xcorr(x,y):
    n = x.shape[0]
    lags = np.arange(-n//2,n//2)
    xhat = FFT.fft(x)
    yhat = FFT.fft(y)
    c = xhat.conj()*yhat
    res = FFT.ifft(c).real
    return lags,np.concatenate((res[n//2:],res[:n//2]))

def moving_average(signal, window=3) :
    ret = np.cumsum(signal, axis=0, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ret =  ret[window - 1:] / window
    new_signal = np.ones_like(signal)*ret[-1]
    new_signal[:-window+1] = ret
    return new_signal


# def _save_object(f,obj):
#     excluded_matrix = []
#     for key,value in obj.__dict__.items():
#         if isinstance(value,int) or isinstance(value,float) or isinstance(value,str) or isinstance(value,bool):
#             f.create_dataset(key,data=value)
#             continue
#         elif isinstance(value,np.ndarray):
#             if key in excluded_matrix:
#                 continue
#             else:
#                 if value.ndim >0:
#                     f.create_dataset(key,data=value,compression='gzip')
#                 else:
#                     f.create_dataset(key,data=value)
        

def _save_object(f,obj,end_here=False):
    excluded_matrix = []
    if isinstance(obj,dict):
        dict_items = obj.items()
    else:
        dict_items = obj.__dict__.items()
    for key,value in dict_items:
        if isinstance(value,int) or isinstance(value,float) or isinstance(value,str) or isinstance(value,bool):
            f.create_dataset(key,data=value)
            continue
        elif isinstance(value,np.ndarray):
            if key in excluded_matrix:
                continue
            else:
                if value.ndim >0:
                    f.create_dataset(key,data=value,compression='gzip')
                else:
                    f.create_dataset(key,data=value)
        
        else:
            if not end_here:
                
                # typeStr = str(type(value))
                if isinstance(value,nb.typed.typedlist.List) or isinstance(value,list):
                    grp = f.create_group(key)
                    res_dct = {str(i): value[i] for i in range(len(value))} 
                    _save_object(grp,res_dct,end_here=True)
                    


                        
