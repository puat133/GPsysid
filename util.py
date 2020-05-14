#%% mix of utility functions to serve GP -system identifications, based on 
#%% 1. Python implementation of systematic_resampling.m written by Andreas Svensson 2016


import numpy as np
import numba as nb
from scipy.special import factorial
from scipy.interpolate import pade
import scipy.linalg as sla
import control.matlab as ctmat
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
    

'''
We use convention that the domain is given in [0,L]
in these basis function
'''
@njitParallel
def onedim_basis(j,Li,xi):
    # return np.sin(np.pi*j*(xi.T+Li)/(2*Li))/np.sqrt(Li)
    return np.sin(np.pi*j*(xi.T)/(Li))/np.sqrt(Li)
        
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

'''
We use convention that the domain is given in [0,L]
in these basis function
'''
@njitSerial
def eigen(index,L):
    lmb = np.zeros(index.shape[1])
    for i in nb.prange(index.shape[1]):
        # lmb[i] =  np.sum(np.square((0.5*np.pi*index[:,i]/L)))    
        lmb[i] =  np.sum(np.square((np.pi*index[:,i]/L)))    
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
    return np.square(sf)*np.sqrt(2*np.pi)*ell*np.exp(-0.5*(ell*ell*np.square(s)))

@njitSerial
def covarianceRadial(diff_x,sf=1.,ell=1.):
    return np.square(sf)*ell*np.exp(-0.5*(np.square(diff_x)/np.square(ell)))



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
                    


'''
SE spectral function using Pad`e approximation
equivalent to se_pade of Simo Sarkka
constructed using scipy
m is the polynomial order of the numerator
n is the polynomial order of the denominator
'''                 
def se_pade(m,n,s,ell):
    order = m+n
    
    #power
    p = np.arange(order+1)

    #intermediate variable
    c = np.power((-np.square(ell)/2),p)/factorial(p)

    #simo's implementation of pade seems different to the scipy one
    #it seems that simo take account negativity of c in his pade_approx.m
    #I dont know which is one correct
    pade_res = pade(c,n,m)

    a = pade_res[0].coefficients
    b = pade_res[1].coefficients

    

    A = np.zeros(2*a.shape[0]-1)
    B = np.zeros(2*b.shape[0]-1)

    A[::2] = a
    B[::2] = b

    A = A*np.square(s)*np.sqrt(2*np.pi)*ell
    return np.poly1d(A),np.poly1d(B)


'''
[F,L,q,H,Pinf] = ratspec_to_ss(A,B,rtype)

convert A(s)/B(s) into state space
A and B are numpy.poly1d object

Is this similar to tf2ss from scipy?
'''
def ratspec_to_ss(A,B,controllable=True):
    q = A(0)/B(0)

    LA = A.c/(np.power(1j,np.arange(len(A.c)-1,-1,-1)))
    LB = B.c/(np.power(1j,np.arange(len(B.c)-1,-1,-1)))

    #only take real coefficient from denumerator
    LA = np.poly1d(LA.real)
    LB = np.poly1d(LB.real)



    #strangely np.roots seems flipped between real and imag
    ra = LA.roots[LA.roots<0]
    rb = LB.roots[LB.roots<0]

    GA = np.poly(ra)
    GB = np.poly(rb)

    GA = np.poly1d(GA.real)
    GB = np.poly1d(GB.real)

    GA.c /= GA.c[-1]
    GB.c /= GB.c[-1]

    GA.c /= GB.c[0]
    GB.c /= GB.c[0]

    F = np.zeros((len(GB.c)-1,len(GB.c)-1),dtype=np.float64)
    L = np.zeros((len(GB.c)-1,1),dtype=np.float64)
    H = np.zeros((1,len(GB.c)-1),dtype=np.float64)
    if controllable:
        
        F[-1,:] = -GB.c[-1:0:-1]
        F[:-1,1:] = np.eye(len(GB.c)-2)
        L[-1,0] = 1
        H[0,:len(GA.c)] = GA.c[-1::-1]

    else:

        F[:,-1] = -GB.c[-1:0:-1]
        F[1:,:-1] = np.eye(len(GB.c)-2)
        L[:len(GA.c),0] = GA.c[-1::-1]
        H[0,-1] = 1
        
    
    # Pinf = lyapchol(F,L*np.sqrt(q))#this is lower triagular
    # Pinf = Pinf@Pinf.conj().T
    Pinf = sla.solve_lyapunov(F,-q*L@L.T)
    return F,L,q,H,Pinf




def covariance_approximation(tau,F,L,q,H,Pinf=None):
    if Pinf == None:
        Pinf = sla.solve_lyapunov(F,-q*L@L.T)
        

    approximated_cov = np.zeros(tau.shape[0])
    approximated_cov[tau>=0] = np.array([ H@Pinf@sla.expm(tau_t*F).T@H.T for tau_t in tau[tau>=0]]).flatten() 
    approximated_cov[tau<0] = np.array([ H@sla.expm(-tau_t*F)@Pinf@H.T for tau_t in tau[tau<0]]).flatten() 
    
    return approximated_cov.flatten()

# function cov_approx = ss_cov(tau,F,L,q,H)

#     %Pinf = are(F',zeros(size(F)),L*q*L');
# %    Pinf = lyap(F,F',L*q*L');
#     Pinf = lyapchol(F,L*sqrt(q));
#     Pinf = Pinf' * Pinf;

#     % Initialize covariance
#     cov_approx = zeros(size(tau));
  
#     % Evaluate positive parts
#     cov_approx(tau >= 0) = arrayfun(@(taut) H*Pinf*expm(taut*F)'*H',tau(tau >= 0));
  
#     % Evaluate negative parts
#     cov_approx(tau < 0) = arrayfun(@(taut) H*expm(-taut*F)*Pinf*H',tau(tau < 0));
# end
    



    

