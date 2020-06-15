#Partial Differential Equations supporting functions
# The boundary condition here is a mixed one 
# for u(t,x) in a one dimensional domain (0,L)
# u(t,0) = 0
# u_z(t,L) = 0
import torch
import numpy as np
PI = torch.acos(torch.zeros(1)).float()


def expm_eps_less_than(delta):
    epsilon = 8
    p = 0
    while epsilon>delta:
        p +=1
        epsilon /= (2**4)*(2*p+3)*(2*p+1)
    
    return p


def expm(A,delta=1e-10):
    # j = max(0,(1+torch.log2(torch.norm(A,float('inf')))).int())
    if(A.requires_grad):
        An = A.detach().numpy()
    else:
        An = A.numpy()
    j = max(0,np.int(1+np.log2(np.linalg.norm(An,np.inf))))
    A = A/(2**j)
    q = expm_eps_less_than(delta)
    # print(q)
    n = A.shape[0]
    I = torch.eye(n)
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
    
    F,LU = torch.solve(N,D) #torch notion is different than np.linalg.solve
    for _ in range(j):
        F = F@F
    
    return F


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
        self.__length = 1000
        self.history = torch.zeros((self.__length,self.A.shape[0]))
        self.index = 0

    @property
    def length(self):
        return self.__length
    
    @length.setter
    def length(self,value):
        self.__length = value
        self.history = torch.zeros((self.__length,self.A.shape[0]))

    
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
    def __init__(self,A,B,C,Q,R,length=1000,state_init=None,P_init=None):
        super().__init__(A,B,C,state_init)
        self.Q = Q
        self.R = R
        super(KalmanFilter, self.__class__).length.fset(self, length)
        if P_init == None:
            self.P = torch.eye(self.A.shape[0])
        else:
            self.P = P_init

        self.P_history = torch.empty((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = torch.zeros((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = torch.empty((self.length,self.C.shape[0]))

        self.P_history[0,:,:] = self.P


    @property
    def length(self):
        return super(KalmanFilter,self).length

    @length.setter
    def length(self,value):
        super(KalmanFilter, self.__class__).length.fset(self, value)
        self.P_history = torch.empty((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = torch.empty((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = torch.empty((self.length,self.C.shape[0]))


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
        S = 0.5*(S+S.T)

        K = torch.cholesky_solve(self.C@self.P,S).T
        self.state = self.state + K@yTilde
        self.P = self.P - K@S@K.T
        self.P = 0.5*(self.P+self.P.T)
        self.record_S(S)
        self.record_yTilde(yTilde)

    '''
    Propagate Kalman Filter
    '''    
    def propagate(self,u,y):
        self.predictive(u)
        self.update(y)
            

    




