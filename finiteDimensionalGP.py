import numpy as np
import numba as nb
import GPutils as util
import h5py
from tqdm import trange
from numba.typed import List,Dict #numba typedList and typedDict
from scipy.stats import multivariate_normal as mvn

njitParallel = nb.njit(parallel=True,fastmath=True)
njitSerial = nb.njit(parallel=False,fastmath=True)


#some test functions, required to make a class declaration (wierd)

# f_true = @(x,u) [(x(1)/(1+x(1)^2))*sin(x(2)); x(2)*cos(x(2)) + x(1)*exp(-(x(1)^2+x(2)^2)/8) + u^3/(1+u^2+0.5*cos(x(1)+x(2)))];
# g_true = @(x) x(1)/(1+0.5*sin(x(2))) + x(2)/(1+0.5*sin(x(1)));
#NARENDA \& LI dynamics
@util.njitSerial
def TEST_F(x,u):
    return np.array([
                    np.sin(x[1])*(x[0]/(1.+x[0]*x[0])),
                    x[1]*np.cos(x[1]) + x[0]*np.exp(-(x[0]*x[0]+x[1]*x[1])/8) + u[0]*u[0]*u[0]/(1.+ u[0]*u[0] + 0.5*np.cos(x[0]+x[1]))
                    ])
NB_TYPE_DYN_FUN = nb.typeof(TEST_F)
@util.njitSerial
def TEST_G(x):
    #x(1)/(1+0.5*sin(x(2))) + x(2)/(1+0.5*sin(x(1)));
    return np.array([x[0]/(1+0.5*np.sin(x[1]))  + x[1]/(1.+0.5*np.sin(x[0]))])
NB_TYPE_MEAS_FUN = nb.typeof(TEST_G)

DEFAULT_LIST = List()
DEFAULT_LIST.append((np.random.randn(2,2),np.random.randn(2,2)))
NB_TYPE_LIST = nb.typeof(DEFAULT_LIST)

  

spec=[ ('__nx',nb.int64),
       ('__nu',nb.int64),
       ('__ny',nb.int64),
       ('__x',nb.float64[::1]),
       ('__y',nb.float64[::1]),
       ('__dynamicFun',NB_TYPE_DYN_FUN),
       ('__measFun',NB_TYPE_MEAS_FUN),
]

@nb.jitclass(spec)
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


NB_TYPE_DYNAMIC = nb.deferred_type()
NB_TYPE_DYNAMIC.define(Dynamic.class_type.instance_type)

class Simulate:
    def __init__(self,steps,nx,u,y,nbases,L,R=0.1,PFweightNum=30):
        
        
        # self.__dynamic = dynamic
        self.__u = u
        self.__y = y
        self.__nu = self.__u.shape[0]
        self.__ny = self.__y.shape[0]
        self.__nx = nx
        self.__steps = steps
        self.__nbases = nbases #assumed to be equal to all x and u
        self.__A = np.zeros((self.__nx,self.nbases**(self.__nx+self.__nu)),dtype=np.float64,order='C')
        self.__Q = np.eye(self.__nx)
        # self.__models = List()
        # self.__models.append((self.__A,self.__Q))
        self.__As = []
        self.__Qs = []
        self.__As.append(self.__A)
        self.__Qs.append(self.__Q)

        self.__index = util.create_index(self.__nx+self.__nu,self.nbases)
        self.__L = L
        self.__timeStep = self.__y.shape[1]
        self.__PFweightNum = PFweightNum
        
        #DefaultZero
        self.__PFweightZero = np.zeros((self.__timeStep,self.__PFweightNum),dtype=np.float64,order='C')
        self.__aZero = np.zeros((self.__timeStep,self.__PFweightNum),dtype=np.int64)  #I dont know what a is
        self.__xPFZero = np.zeros((self.__PFweightNum,self.__nx,self.__timeStep),dtype=np.float64,order='C')
        
        
        self.__PFweight =  self.__PFweightZero
        self.__a = self.__aZero
        self.__xPF = self.__xPFZero
        self.__x_prim = np.zeros((self.__nx,self.__timeStep),dtype=np.float64,order='C')
        
        self.__R = R
        self.__iA = np.zeros((self.__nx,self.__nx),dtype=np.float64,order='C')
        self.__iB = np.zeros((self.__nx,self.__nu),dtype=np.float64,order='C')
        self.__iC = np.zeros((self.__ny,self.__nx),dtype=np.float64,order='C')
        self.__I = np.eye(self.nbases**(self.__nx+self.__nu))
        self.__LambdaQ = np.eye(self.__nx)
        self.__lQ = 100
        self.__lambda = util.eigen(self.__index,self.__L)
        self.__ell = 1.
        self.__Vgain = 1000
        self.__V = util.spectrumRadial(np.sqrt(self.__lambda),self.__Vgain,self.__ell)
        self.__burnInPercentage = 1
        self.x_test_sim = None
        self.y_test_sim = None
        
    def save(self,file_name):
        with h5py.File(file_name,'w') as f:
            util._save_object(f,self)    
        
    
    def addmodel(self,newA,newQ):
        self.__models.append((newA,newQ))

    #Assume that observation function gives the last ny elements of x
    def observe(self,x):
        # return x[-self.__ny:]
        return self.__iC@x

    def setToZero(self):
        self.__PFweight = self.__PFweightZero.copy()
        self.__a = self.__aZero.copy()
        self.__xPF = self.__xPFZero.copy()


    @property
    def burnInPercentage(self):
        return self.__burnInPercentage

    @property
    def PFweight(self):
        return self.__PFweight

    @property
    def V(self):
        return self.__V

    @property
    def a(self):
        return self.__a

    @property
    def xPF(self):
        return self.__xPF

    @property
    def x_prim(self):
        return self.__x_prim

    @property
    def A(self):
        return self.__A

    @property
    def Q(self):
        return self.__Q

    @property
    def L(self):
        return self.__L
    
    @property
    def index(self):
        return self.__index
    
    @property
    def latest_model(self):
        return self.__models[-1]

    @property
    def nx(self):
        return self.__nx

    @property
    def nu(self):
        return self.__nu

    @property
    def ny(self):
        return self.__ny

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

    @property
    def iA(self):
        return self.__iA

    @property
    def iB(self):
        return self.__iB

    @property
    def iC(self):
        return self.__iC

    @property
    def ell(self):
        return self.__ell

    @property
    def lQ(self):
        return self.__lQ
   

    @property
    def Vgain(self):
        return self.__Vgain

    @Vgain.setter
    def Vgain(self,value):
        self.__Vgain = value
        self.__V = util.spectrumRadial(np.sqrt(self.__lambda),self.__Vgain,self.__ell)

    @lQ.setter
    def lQ(self,value):
        self.__lQ = value

    @ell.setter
    def ell(self,value):
        self.__ell = value
        self.__V = util.spectrumRadial(np.sqrt(self.__lambda),self.__Vgain,self.__ell)

    @R.setter
    def R(self,value):
        self.__R = value

    @iA.setter
    def iA(self,value):
        self.__iA = value

    @iC.setter
    def iC(self,value):
        self.__iC = value

    @iB.setter
    def iB(self,value):
        self.__iB = value

    @burnInPercentage.setter
    def burnInPercentage(self,value):
        self.__burnInPercentage = value

    # def __evaluate_latest_model(self,x,u):
    #     return util.evaluate_latest_model(self.__iA,self.__iB,self.__A,self.__index,self.__L,x,u)


    def __runParticleFilter(self,k):
        self.__a,self.__PFweight,self.__xPF = util.runParticleFilter(self.__Q,self.__timeStep,k,self.__a,self.__PFweight,self.__PFweightNum,  
                        self.__iA,self.__iB,self.__A,self.__index,self.__xPF,self.__nx,self.__L,self.__u,
                        self.y,self.__R)
        # Qchol = np.linalg.cholesky(self.__Q)
        # # for t in trange(self.__timeStep,desc='SMC - {} th'.format(k+1)):
        # for t in range(self.__timeStep):
        #     if t>=1:
        #         if k>0:
        #             self.__a[t,:-1] = util.systematic_resampling(self.__PFweight[t-1,:],self.__PFweightNum-1)
        #             f = self.__evaluate_latest_model(self.__xPF[:,self.__a[t,:-1],t-1],self.__u[:,t-1])
        #             self.__xPF[:,:-1,t] = f + Qchol@np.random.randn(self.__nx,self.__PFweightNum-1)
        #             f = self.__evaluate_latest_model(self.__xPF[:,:,t-1],self.__u[:,t-1])
        #             waN = self.__PFweight[t-1,:]*util.mvnpdf(f.T,self.__xPF[:,-1,t-1],self.__Q)#mvn.pdf(f.T,self.__xPF[:,-1,t-1],self.__Q)
        #             waN /= np.sum(waN)
        #             self.__a[t,-1] = util.systematic_resampling(waN,1)
        #         else:
        #             self.__a[t,:] = util.systematic_resampling(self.__PFweight[t-1,:],self.__PFweightNum)
        #             f = self.__evaluate_latest_model(self.__xPF[:,self.__a[t,:-1],t-1],self.__u[:,t-1])
        #             self.__xPF[:,:-1,t] = f + Qchol@np.random.randn(self.__nx,self.__PFweightNum-1)


        #     log_w = -0.5*(self.observe(t)-self.y[:,t])**2/self.R
        #     self.__PFweight[t,:] = np.exp(log_w-np.max(log_w))
        #     self.__PFweight[t,:] /= np.sum(self.__PFweight[t,:])

    def __update_statistics(self):
        Phi,Psi,Sig = util.compute_Phi_Psi_Sig(self.__iA,self.__iB,self.__x_prim,self.__index,self.__L,self.__u)
        self.__A,self.__Q = util.gibbsParam(Phi,Psi,Sig,self.__V,self.__LambdaQ,self.__lQ,self.__timeStep-1,self.__I)
        # self.__models.append((self.__A,self.__Q))
        self.__As.append(self.__A)
        self.__Qs.append(self.__Q)



    def run(self):
        
        for k in trange(self.__steps,desc='Simulation'):
            #This probabiy not required
            self.setToZero()
            
            #Initialization
            if k>0:
                self.__xPF[-1,:,:] = self.__x_prim
            
            self.__PFweight[0,:] = 1.
            self.__PFweight[0,:] /= np.sum(self.__PFweight[0,:])
            
            # CPF with ancestor sampling
            self.__xPF[:-1,:,0] = 0


            self.__runParticleFilter(k)
            star = util.systematic_resampling(self.__PFweight[-1,:],1)
            self.__x_prim[:,-1] = self.__xPF[star,:,-1].flatten()

            #loop from the back
            for t in np.flip(np.arange(1,self.__timeStep)):
                star = self.__a[t,star]
                self.__x_prim[:,t-1] = self.__xPF[star,:,t-1].flatten()

            # print('Sampling. k = {}/{}'.format(k,self.__steps))

            self.__update_statistics()

    def evaluate(self,yTest,uTest,Kn=1):
        # ny = yTest.shape[0]
        burn_in = (self.__burnInPercentage*self.__steps)//100
        remain_step = self.__steps - burn_in
        eval_timeSteps = yTest.shape[1]
        self.x_test_sim = np.zeros((remain_step*Kn,self.__nx,eval_timeSteps),dtype=np.float64,order='C')
        # if yTest.ndim > 1:
        self.y_test_sim = np.zeros((remain_step*Kn,self.__ny,eval_timeSteps),dtype=np.float64,order='C')
        # else:
            # y_test_sim = np.zeros((eval_timeSteps,remain_step*Kn),dtype=np.float64,order='C')
        if isinstance(self.__R,np.ndarray):
            if self.__R.ndim ==2:
                Rchol = np.linalg.cholesky(self.__R)
        else:
            Rchol = np.sqrt(self.__R)
        
        if uTest.ndim == 1:
            uTest = uTest[np.newaxis,:]
        for k in trange(remain_step,desc='Evaluation'):
            # self.__A,self.__Q = self.__models[k+burn_in]
            
            self.__A = self.__As[k+burn_in]
            self.__Q = self.__Qs[k+burn_in]

            Qchol = np.linalg.cholesky(self.__Q)
            
            for kn in range(Kn):
                ki = k*Kn+ kn
                for t in range(eval_timeSteps-1):
                    self.x_test_sim[ki,:,t+1] = util.evaluate_model_thin(self.__iA,self.__iB,self.__A,self.__index,self.__L,self.x_test_sim[ki,:,t],uTest[:,t])+Qchol@np.random.randn(self.__nx)
                    # y_test_sim[ki,:,t] = x_test_sim[ki,t,-1] + Rchol*np.random.randn()
                    #Rchol@np.random.randn(self.ny)
                    self.y_test_sim[ki,:,t] = self.observe(self.x_test_sim[ki,:,t]) + Rchol*np.random.randn(self.__ny)


        
        return self.x_test_sim,self.y_test_sim



        



            



            

            






