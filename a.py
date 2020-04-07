import numpy as np
import numba as nb
import util
from numba.typed import List,Dict #numba typedList and typedDict
from scipy.stats import multivariate_normal as mvn

njitParallel = nb.njit(parallel=True,fastmath=True)
njitSerial = nb.njit(parallel=False,fastmath=True)


class Dynamic:
    def __init__(self,dynamicFun,measFun,nx,nu,ny):
        self.__dynamicFun = dynamicFun
        self.__measFun = measFun
        self.__nx = nx
        self.__nu = nu
        self.__ny = ny
        self.__x = np.zeros(nx)
        self.__y = np.zeros(ny)
        


    def oneStep(self,x,u):
        return  self.__dynamicFun(x,u)


    def measure(self,x):
        return self.__measFun(x)

    
    @property
    def nx(self):
        return self.__nx

    @property
    def nu(self):
        return self.__nu

    @property
    def ny(self):
        return self.__ny

class Simulate:
    def __init__(self,steps,dynamic,u,y,nbases,L,R=0.1,timeStep=2000,PFweightNum=30):
        self.__dynamic = dynamic
        self.__steps = steps
        self.__nbases = nbases #assumed to be equal to all x and u
        self.__A = np.zeros(self.nx,self.nbases)
        self.__Q = np.eye(dynamic.nx)
        self.__models = List()
        self.__models.append((self.__A,self.__Q))
        self.__index = util.create_index(self.nx+self.nu,self.nbases)
        self.__L = L
        self.__timeStep = timeStep
        self.__PFweightNum = PFweightNum
        self.__PFweight = np.zeros((self.__timeStep,self.__PFweightNum))
        self.__a = self.__PFweight.copy()#I dont know what a is
        self.__xPF = np.zeros((self.__dynamic.nx,self.__PFweightNum,self.__timeStep))
        self.__u = u
        self.__y = y
        self.__R = R

        
    
    def addmodel(self,newA,newQ):
        self.__models.append((newA,newQ))

    #Assume that observation function gives the last element of x
    def observe(self,t):
        return self.__xPF[-1,:,t]

    
    @property
    def get_latest_model(self):
        return self.__models[-1]

    @property
    def nx(self):
        return self.__dynamic.nx

    @property
    def nu(self):
        return self.__dynamic.nu

    @property
    def ny(self):
        return self.__dynamic.ny

    @property
    def nbases(self):
        return self.__nbases
    
    @property
    def steps(self):
        return self.__steps

    @property
    def u(self):
        return self.__u

    @property
    def y(self):
        return self.__y

    @property
    def R(self):
        return self.__R



    def evaluate_latest_model(self,x,u):
        A,Q = self.__models[-1]
        return A@util.basis(self.__index,self.__L,np.concatenate((x,u)))


    def __runParticleFilter(self,k):
        Q = self.__models[-1][1]
        Qchol = np.linalg.cholesky(Q)
        for t in range(self.__timeStep):
            if t>=1:
                if k>0:
                    self.__a[t,:-1] = util.systematic_resampling(self.__PFweight[t-1,:],self.__PFweightNum-1)
                    f = self.evaluate_latest_model(self.__xPF[:,self.__a[t,:-1],t-1],self.u[:,t-1])
                    self.__xPF[:,:-1,t] = f + Qchol@np.random.randn(self.nx,self.__timeStep-1)
                    waN = self.__PFweight[t-1,:]*mvn.pdf(f,self.__xPF[:,-1,t],Q)
                    waN /= np.sum(waN)
                    self.__a[t,-1] = util.systematic_resampling(waN,1)
                else:
                    self.__a[t,:] = util.systematic_resampling(self.__PFweight[t-1,:],self.__PFweightNum)
                    f = self.evaluate_latest_model(self.__xPF[:,self.__a[t,:-1],t-1],self.u[:,t-1])
                    self.__xPF[:,:-1,t] = f + Qchol@np.random.randn(self.nx,self.__timeStep-1)


            log_w = -0.5*(self.observe(t)-self.y)^2/self.R
            self.__PFweight[t,:] = np.exp(log_w-np.max(log_w))
            self.__PFweight[t,:] /= np.sum(self.__PFweight[t,:])




    def run(self):
        x_prim = np.zeros((self.nx,1,self.__timeStep))
        for k in range(self.__steps):
            self.__runParticleFilter(k)


            star = util.systematic_resampling(self.__PFweight[-1,:],1)
            x_prim[:,1,-1] = self.__xPF[:,star,-1]

            

            

            






