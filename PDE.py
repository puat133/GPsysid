#Partial Differential Equations supporting functions
# The boundary condition here is a mixed one 
# for u(t,x) in a one dimensional domain (0,L)
# u(t,0) = 0
# u_z(t,L) = 0
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
j is index
L is the length of domain
x is the spatial variable
'''
def basis(j,L,x):
    if isinstance(j,np.ndarray):
        if j.ndim == 1: #j is a vector
            basis = np.sin(np.pi*(2*j[:,np.newaxis]-1)*x/L)/np.sqrt(L/2)
    else: #single entry j
        basis = np.sin(np.pi*(2*j-1)*x/L)/np.sqrt(L/2)

    return basis



'''
nbases is number of eigenfunction required.
L is spatial domain length
'''
def eigenvalues(nbases,L):
    eig = np.empty(nbases)
    for i in range(nbases):
        eig[i] = np.square((2*i-1)*np.pi/(2*L))

    return eig

'''
laplacian in the Fourier basis
'''
def laplacian(nbases,L):
    eig = eigenvalues(nbases,L)
    Lap = -np.diag(eig)
    return Lap

'''
derivative in the Fourier basis
'''
def derivative(nbases,L):
    der = np.zeros((nbases,nbases))
    L_per_pi = L/np.pi
    for i in range(nbases):
        for j in range(nbases):
            if (i+j)%2 == 0:#even case
                der[i,j] = L_per_pi/(i+j-1)
            else:
                der[i,j] = L_per_pi/(i-j)
    return der



class DiscreteLinearDynamic():
    def __init__(self,A,B,C,state_init=None):
        self.A = A
        self.B = B
        self.C = C
        if state_init==None:
            self.state = np.zeros(A.shape[0])
        else:
            self.state = state_init
        self.output = C@self.state
        self.history_length = 1000
        self.history = np.zeros((self.history_length,self.A.shape[0]))
        self.index = 0

    
    def propagate(self,u):
        self.state = self.A@self.state + self.B@u
        self.index += 1
        self.output = self.C@self.state
        

    def recordstate(self):
        self.history[self.index,:] = self.state
        

class KalmanFilter(DiscreteLinearDynamic):
    def __init__(self,A,B,C,Q,R,state_init=None,P_init=None):
        super().__init__(A,B,C,state_init)
        self.Q = Q
        self.R = R
        if P_init == None:
            self.P = np.eye(self.A.shape[0])
        else:
            self.P = P_init

        self.P_history = np.empty((self.history_length,self.P.shape[0],self.P.shape[1]))
        self.S_history = np.empty((self.history_length,self.R.shape[0],self.R.shape[1]))

    def record_S(self,S_input):
        self.S_history[self.index,:,:] = S_input
    

    '''
    This implement a naive kalman filter prediction step
    '''
    def predictive(self,u):
        #do predictive step
        super().propagate(u)
        self.P = self.A@self.P@self.A.T + self.Q

        #only record the prediction
        self.recordstate()

    '''
    This implement a naive Kalman filter update step
    implementing using cholesky factorization could be faster
    '''
    def update(self,y):
        yTilde = y - C@self.state
        S = self.C@self.P@self.C.T + self.R

        K = np.linalg.solve(self.P@self.C.T,S)
        self.state = self.state + K@yTilde
        self.P = self.P - K@S@K.T
        self.record_S(S)

    def propagate(self,u,y):
        self.predictive(u)
        self.update(y)
            

    




