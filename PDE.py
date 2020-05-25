#Partial Differential Equations supporting functions
# The boundary condition here is a mixed one 
# for u(t,x) in a one dimensional domain (0,L)
# u(t,0) = 0
# u_z(t,L) = 0
import torch
import numpy as np
PI = torch.acos(torch.zeros(1))



'''
j is index
L is the length of domain
x is the spatial variable
'''
def basis(j,L,x):
    if isinstance(j,torch.Tensor):
        if j.ndim == 1: #j is a vector
            basis = torch.sin(PI*(2*j.view(j.shape[0],1)-1)*x/L)/torch.sqrt(L/2)
    else: #single entry j
        basis = torch.sin(PI*(2*j-1)*x/L)/torch.sqrt(L/2)

    return basis



'''
nbases is number of eigenfunction required.
L is spatial domain length
'''
def eigenvalues(nbases,L):
    index = torch.arange(nbases)
    eig = torch.square((2*index-1)*PI/(2*L))
    return eig

'''
laplacian in the Fourier basis
'''
def laplacian(nbases,L):
    eig = eigenvalues(nbases,L)
    Lap = -torch.diag(eig)
    return Lap

'''
derivative in the Fourier basis
'''
def derivative(nbases,L):
    der = torch.zeros((nbases,nbases))
    L_per_pi = L/PI
    for i in range(nbases):
        for j in range(nbases):
            if i != j:
                if (i+j)%2 == 0:#even case
                    der[i,j] = L_per_pi/(i+j-1)
                else:
                    der[i,j] = L_per_pi/(i-j)
    return der


'''
Simple Class to contain the abstraction of a discrete linear dynamic
'''
class DiscreteLinearDynamic():
    def __init__(self,A,B,C,state_init=None):
        self.A = A
        self.B = B
        self.C = C
        if state_init==None:
            self.state = torch.zeros(A.shape[0])
        else:
            self.state = state_init
        self.output = C@self.state
        self.history_length = 1000
        self.history = torch.zeros((self.history_length,self.A.shape[0]))
        self.index = 0

    
    def propagate(self,u):
        self.state = self.A@self.state + self.B@u#.view(-1)
        self.index += 1
        self.output = self.C@self.state
        

    def recordstate(self):
        self.history[self.index,:] = self.state.squeeze()
        
'''
Discrete KalmanFilter Class, inheriting the DiscreteLinearDynamic
'''
class KalmanFilter(DiscreteLinearDynamic):
    def __init__(self,A,B,C,Q,R,state_init=None,P_init=None):
        super().__init__(A,B,C,state_init)
        self.Q = Q
        self.R = R
        if P_init == None:
            self.P = torch.eye(self.A.shape[0])
        else:
            self.P = P_init

        self.P_history = torch.empty((self.history_length,self.P.shape[0],self.P.shape[1]))
        self.S_history = torch.empty((self.history_length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = torch.empty((self.history_length,self.C.shape[0]))

    def record_S(self,S_input):
        self.S_history[self.index,:,:] = S_input

    
    def record_yTilde(self,yTilde_input):
        self.yTilde_history[self.index,:] = yTilde_input

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
        yTilde = y - self.C@self.state
        S = self.C@self.P@self.C.T + self.R

        K = torch.cholesky_solve(self.C@self.P,S).T
        self.state = self.state + K@yTilde
        self.P = self.P - K@S@K.T
        self.record_S(S)
        self.record_yTilde(yTilde)

    '''
    Propagate Kalman Filter
    '''    
    def propagate(self,u,y):
        self.predictive(u)
        self.update(y)
            

    




