#Partial Differential Equations supporting functions
# The boundary condition here is a mixed one 
# for u(t,x) in a one dimensional domain (0,L)
# u(t,0) = 0
# u_z(t,L) = 0

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
def expm_eps_less_than(delta):
    epsilon = 8
    p = 0
    while epsilon>delta:
        p +=1
        epsilon /= (2**4)*(2*p+3)*(2*p+1)
    
    return p

@njitSerial
def expm(A,delta=1e-10):
    j = max(0,np.int(1+np.log2(np.linalg.norm(A,np.inf))))
    A = A/(2**j)
    q = expm_eps_less_than(delta)
    # print(q)
    n = A.shape[0]
    I = np.eye(n)
    D = I
    N = I
    X = I
    c = 1
    sign = 1
    for k in range(1,q+1):
        c = c*(q-k+1)/((2*q - k+ 1)*k)
        X = A@X
        N = N+ c*X
        sign =  -1*sign
        D = D+ sign*c*X
    
    # F,LU = torch.solve(N,D) #torch notion is different than np.linalg.solve
    F = np.linalg.solve(D,N)
    for _ in range(j):
        F = F@F
    
    return F


'''
j is index
L is the length of domain
x is the spatial variable
'''
@njitSerial
def basis(j,L,x):
    # if isinstance(j,np.ndarray):
    if j.ndim == 1: #j is a vector
        basis = np.sin(np.pi*(2*j.reshape((-1, 1))-1)*x/L)/np.sqrt(L/2)
    else: #single entry j
        basis = np.sin(np.pi*(2*j-1)*x/L)/np.sqrt(L/2)

    return basis



'''
nbases is number of eigenfunction required.
L is spatial domain length
'''
@njitSerial
def eigenvalues(nbases,L):
    index = np.arange(nbases)
    eig = np.square((2*index-1)*np.pi/(2*L))
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
    # L_per_pi = L/np.pi
    for i in range(nbases):
        for j in range(nbases):
            # if i != j:
            if (i+j)%2 == 0:#even case
                der[i,j] = (2*i-j)/(L*(i+j-1))
            else:
                der[i,j] = (2*i-j)/(L*(i-j))
    return der



def _save_object(f,obj,end_here=False,excluded_matrix=[],special_types = []):
    # excluded_matrix = ['H','Ht','I','In','H_t_H','Imatrix','Dmatrix','ix','iy','y','ybar']
    
    for key,value in obj.__dict__.items():
        if isinstance(value,int) or isinstance(value,float) or isinstance(value,str) or isinstance(value,bool):
            f.create_dataset(key,data=value)
            continue
        # elif isinstance(value,cp.core.core.ndarray):
        #     if key in excluded_matrix:
        #         continue
        #     else:
        #         if value.ndim >0:
        #             f.create_dataset(key,data=cp.asnumpy(value),compression='gzip')
        #         else:
        #             f.create_dataset(key,data=cp.asnumpy(value))
        elif isinstance(value,np.ndarray):
            if value.ndim >0:
                f.create_dataset(key,data=value,compression='gzip')
            else:
                f.create_dataset(key,data=value)
            continue
        else:
            if not end_here:
                
                typeStr = str(type(value))
                if isinstance(value,list):
                    for i in range(len(value)):
                        grp = f.create_group(key + ' {0}'.format(i))
                        _save_object(grp,value[i],end_here=True)
                elif typeStr in special_types:
                    grp = f.create_group(key)
                    _save_object(grp,value,end_here=True)

            

    




