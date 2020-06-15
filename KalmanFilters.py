import numpy as np
import scipy.linalg as sla
import numba as nb

import pathlib
import os
import h5py
import datetime
from tqdm import trange

import GPutils as utils
import Nonlinear_fun as NL

from abc import ABC, abstractmethod

CACHE = True
PARALLEL = False
FASTMATH = True
jitSerial = nb.jit(parallel=False,fastmath=FASTMATH,cache=CACHE)
jitParallel = nb.jit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitSerial = nb.njit(parallel=False,fastmath=FASTMATH,cache=CACHE)

'''
Simple Class to contain the abstraction of a discrete linear dynamic
'''
spec1=[ ('A',nb.float64[:,:]),
       ('B',nb.float64[:,:]),
       ('H',nb.float64[:,:]),
       ('history',nb.float64[:,:]),
       ('state',nb.float64[:]),
       ('output',nb.float64[:]),
       ('__length',nb.int64),
       ('index',nb.int64),       
]

@nb.jitclass(spec1)
class DiscreteLinearDynamic():
    def __init__(self,A,B,H,state_init=None):
        self.A = A
        self.B = B
        self.H = H
        if state_init==None:
            self.state = np.zeros(A.shape[0])
        else:
            self.state = state_init
        self.output = H@self.state
        self.__length = 1000
        self.history = np.zeros((self.__length,self.A.shape[0]))
        self.index = 0

    @property
    def length(self):
        return self.__length
    
    @length.setter
    def length(self,value):
        self.__length = value
        self.history = np.zeros((self.__length,self.A.shape[0]))

    
    def propagate(self,u):
        self.state = self.A@self.state + self.B@u#.view(-1)
        self.index += 1
        self.output = self.H@self.state
        

    def recordstate(self):
        self.history[self.index,:] = self.state.squeeze()
        
'''
Discrete KalmanFilter Class, inheriting the DiscreteLinearDynamic
'''
spec2=[ ('A',nb.float64[:,:]),
       ('B',nb.float64[:,:]),
       ('H',nb.float64[:,:]),
       ('Q',nb.float64[:,:]),
       ('R',nb.float64[:,:]),
       ('P',nb.float64[:,:]),
       ('P_history',nb.float64[:,:,:]),
       ('S_history',nb.float64[:,:,:]),
       ('history',nb.float64[:,:]),
       ('measured_y_hist',nb.float64[:,:]),
       ('yTilde_history',nb.float64[:,:]),
       ('state',nb.float64[:]),
       ('output',nb.float64[:]),
       ('__length',nb.int64),
       ('index',nb.int64),       
]
@nb.jitclass(spec2)
class KalmanFilter():
    def __init__(self,A,B,H,Q,R,length=1000):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        # if state_init==None:
        self.state = np.zeros(A.shape[0])
        # else:
            # self.state = state_init
        self.output = H@self.state
        self.__length = length
        self.history = np.zeros((self.__length,self.A.shape[0]))
        self.index = 0
        self.measured_y_hist = np.zeros((self.__length,self.H.shape[0]))

        
        
        # if P_init == None:
        self.P = np.eye(self.A.shape[0])
        # else:
            # self.P = P_init

        self.P_history = np.empty((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = np.zeros((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = np.empty((self.length,self.H.shape[0]))

        self.P_history[0,:,:] = self.P


    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self,value):
        self.__length = value
        self.history = np.zeros((self.__length,self.A.shape[0]))
        self.P_history = np.empty((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = np.empty((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = np.empty((self.length,self.H.shape[0]))
        self.measured_y_hist = np.zeros((self.__length,self.H.shape[0]))

    def recordstate(self):
        self.history[self.index,:] = self.state

    def record_S(self,S_input):
        self.S_history[self.index,:,:] = S_input

    def record_P(self,P_input):
        self.P_history[self.index,:,:] = P_input
    
    # def record_yTilde(self,yTilde_input):
    #     self.yTilde_history[self.index,:] = yTilde_input

    '''
    This implement a naive kalman filter prediction step
    '''
    def predictive(self,u):
        #do predictive step
        self.state = self.A@self.state + self.B@u#.view(-1)
        self.index += 1
        self.output = self.H@self.state

        self.P = self.A@self.P@self.A.T + self.Q

        #only record the prediction
        self.recordstate()
        

    '''
    This implement a naive Kalman filter update step
    implementing using cholesky factorization could be faster
    '''
    def update(self,y):
        yTilde = y - self.H@self.state
        S = self.R + self.H@self.P@self.H.T 
        # S = 0.5*(S+S.T)

        K = np.linalg.solve(S,self.H@self.P).T
        self.state = self.state + K@yTilde
        self.P = self.P - K@S@K.T
        # self.P = 0.5*(self.P+self.P.T)
        
        self.record_S(S)
        # self.record_P(self.P)
        
        # self.record_yTilde(yTilde)

    '''
    Propagate Kalman Filter
    '''    
    def propagate(self,u,y):
        self.predictive(u)
        self.update(y)

    #assuming that no u is involved
    def propagate_till_end(self,u):
        for i in range(self.measured_y_hist.shape[0]-1):
            self.propagate(u,self.measured_y_hist[i,:])



'''
Discrete Extended KalmanFilter Class: Jitclass does not support inheritance :(
'''
specEKF=[ ('nstate',nb.int64),
        ('ninput',nb.int64),
        ('f',nb.typeof(NL.catalyst_dynamics_discrete)),
        ('P_dyn',nb.typeof(NL.P_dynamics_discrete)),
       
       ('H',nb.float64[:,:]),
       ('Q',nb.float64[:,:]),
       ('R',nb.float64[:,:]),
       ('S',nb.float64[:,:]),
       ('state',nb.float64[:]),
       ('output',nb.float64[:]),
       ('__length',nb.int64),
       ('history',nb.float64[:,:]),
       ('index',nb.int64),
       ('measured_y_hist',nb.float64[:,:]),
       ('input_hist',nb.float64[:,:]),
       ('P',nb.float64[:,:]),
       ('P_history',nb.float64[:,:,:]),
       ('S_history',nb.float64[:,:,:]),
       ('yTilde_history',nb.float64[:,:]),

       ('parameters',nb.typeof(NL.parameters)),
       ('matrices',nb.typeof(NL.matrices))
]
@nb.jitclass(specEKF)
class ExtendedKalmanFilter():
    def __init__(self,nstate,ninput,parameters,matrices,f,P_dyn,length=1000):
        self.nstate = nstate
        self.ninput = ninput
        self.f = f
        self.P_dyn = P_dyn
        self.H = matrices['H']
        self.Q = matrices['Q']
        self.R = matrices['R']
        self.S = self.R
        # if state_init==None:
        self.state = np.zeros(self.nstate)
        # else:
            # self.state = state_init
        self.output = self.H@self.state
        self.__length = length
        self.history = np.zeros((self.__length,self.nstate))
        self.index = 0
        self.measured_y_hist = np.zeros((self.__length,self.H.shape[0]))
        self.input_hist = np.zeros((self.__length,self.ninput))

        
        
        # if P_init == None:
        self.P = np.eye(self.nstate)
        # else:
            # self.P = P_init

        self.P_history = np.empty((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = np.zeros((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = np.empty((self.length,self.H.shape[0]))
        self.P_history[0,:,:] = self.P

        self.parameters = parameters
        self.matrices = matrices
        
        


    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self,value):
        self.__length = value
        self.history = np.zeros((self.__length,self.nstate))
        self.P_history = np.zeros((self.length,self.P.shape[0],self.P.shape[1]))
        self.S_history = np.zeros((self.length,self.R.shape[0],self.R.shape[1]))
        self.yTilde_history = np.zeros((self.length,self.H.shape[0]))
        self.measured_y_hist = np.zeros((self.__length,self.H.shape[0]))

    def recordstate(self):
        self.history[self.index,:] = self.state

    def record_S(self,S_input):
        self.S_history[self.index,:,:] = S_input

    def record_P(self,P_input):
        self.P_history[self.index,:,:] = P_input
    
    # def record_yTilde(self,yTilde_input):
    #     self.yTilde_history[self.index,:] = yTilde_input

    '''
    This implement a naive kalman filter prediction step
    '''
    def predictive(self,u):
        #do predictive step
        self.state = self.f(self.parameters,self.state,u) 
        self.index += 1
        self.output = self.H@self.state

        
        self.P = self.P_dyn(self.parameters,self.matrices,self.state,self.P)
        self.P = 0.5*(self.P+self.P.T)
        #only record the prediction
        self.recordstate()
        

    '''
    This implement a naive Kalman filter update step
    implementing using cholesky factorization could be faster
    '''
    def update(self,y):
        update_failed = False
        try:
            yTilde = y - self.H@self.state
            self.S = self.R + self.H@self.P@self.H.T 
            

            K = np.linalg.solve(self.S,self.H@self.P).T
            self.state = self.state + K@yTilde
            self.P = self.P - K@self.S@K.T
            
            
            self.record_S(self.S)
        except Exception : #np.linalg.LinAlgError:
            update_failed = True
        return update_failed
        
        

    '''
    Propagate Kalman Filter
    '''    
    def propagate(self,u,y):
        self.predictive(u)
        self.update(y)

    #assuming that no u is involved
    def propagate_till_end(self):   
        for i in range(self.measured_y_hist.shape[0]-1):
            self.propagate(self.input_hist[i,:],self.measured_y_hist[i,:])
            

    '''
    Propagate Kalman Filter
    '''    
    def propagate_with_check(self,u,y):
        self.predictive(u)
        fail = self.sanity_check()
        if not fail:
            update_failed = self.update(y)
            if not update_failed:
                fail = self.sanity_check()
            else:
                fail = update_failed
        
        return fail


    #assuming that no u is involved
    def propagate_till_end_with_check(self):
        fail = False
        for i in range(self.measured_y_hist.shape[0]-1):
            fail = self.propagate_with_check(self.input_hist[i,:],self.measured_y_hist[i,:])
            if fail:
                break
        return fail

        

    '''
    Check that there are no element in the state that are inf, nan, or negative as these values are always positive
    '''
    def sanity_check(self):
        cond_1 = np.any(np.isnan(self.state))
        cond_2 = np.any(np.isinf(self.state))
        cond_3 = np.any(self.state<0.)
        cond_4 = np.any(np.isnan(self.S))
        cond_5 = np.any(np.isnan(self.S))
        return cond_1 or cond_2 or cond_3 or cond_4 or cond_5



'''
dummy class to handle simulation data
'''
class SimulationBase(ABC):
    def __init__(self,args):
        
        #if load folder is not empty
        if args.load:
            self.init_with_load(args)
        
        else:
            self.init_without_load(args)

        

        
    @abstractmethod
    def get_new_sample(self,current_sample,randw):
        return

    @abstractmethod
    def compute_neg_loglikelihood(self,sample):
        return 

    @abstractmethod
    def adapt_step(self,partial_acceptance_rate):
        return

    @abstractmethod
    def load_from_file(self,file):
        pass

    @abstractmethod
    def how_many_params(self):
        return

    @abstractmethod
    def analyze(self,skip=1):
        pass

    @abstractmethod
    def plotSamples(self,step_plot=10,hist_bin=10):
        pass

    @abstractmethod
    def plotAnalysisResult(self,results):
        pass
        
    def init_with_load(self,args):
        self.determineRelativePath()
        self.simResultPath = self.relativePath/args.load
        self.load_from_file(str(self.simResultPath/'result.hdf5'))
        #Limit the neg_log_likelihood
        self.lower_limit_neg_log_likelihood = -np.inf
    
    def init_without_load(self,args):
        #decompose the argument
        self.sampling_period = args.sampling_period # again assumption
        self.samples_length = args.samples_num
        self.step_size = args.initial_step_size
        self.adaptation_steps = args.adaptation_steps
        self.feedback_gain = args.feedback_gain
        self.target_acceptance = args.target_acceptance
        self.random_seed = args.seed
        self.burn = args.burn
        self.determineRelativePath()
        self.createFolder(args.folderName)
        
        # self.nparams = self.how_many_params()
        self.randw = np.random.randn(self.samples_length,self.nparams)
        self.log_uniform = np.log(np.random.rand(self.samples_length))
        self.samples = np.zeros((self.samples_length,self.nparams))
        # self.nbases = args.bases_num
        np.random.seed(args.seed)
        sample_init,success = self.initialize_from_folder(args.init)
        if not success:
            sample_init = self.get_new_sample(self.samples[0,:],self.randw[0,:])
        
        self.samples[0,:] = sample_init
        
        
        self.accepted = 0
        #Limit the neg_log_likelihood
        self.lower_limit_neg_log_likelihood = -np.inf
        


    def determineRelativePath(self):
        
        if 'WRKDIR' in os.environ:
            relativePath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult_Neste'
        elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult_Neste').exists():
            relativePath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult/Neste')
        else:
            relativePath = pathlib.Path.home() / 'Documents' / 'SimulationResult_Neste'
        
        self.relativePath = relativePath

        #also return?
        return relativePath

    def createFolder(self,folderName):
        if not folderName: #empty string are false
            folderName = 'Results-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M_%S')
        else:
            folderName = folderName
        
        simResultPath = self.relativePath/folderName

        if not simResultPath.exists():
            simResultPath.mkdir()
        self.simResultPath = simResultPath

        #also return?
        return simResultPath

    def initialize_from_folder(self,target_folder):
        success = False
        sample_init = None    
        if target_folder:
            init_folder = self.relativePath /target_folder
            file_name = 'result.hdf5'
            init_file_path = init_folder/file_name

        
            with h5py.File(init_file_path,mode='r') as file:
                print('loading initial state from {}'.format(init_file_path))
                chain = file['samples'][()]
                sample_init = chain[-1,:]
                success = True

        return sample_init,success

    def train(self):
        # sample_init = np.mean(samples[burned_index:],axis=1)
        partial_acceptance = 0
        latest_accepted_samples_neg_likelihood,_ = self.compute_neg_loglikelihood(self.samples[0,:])
        with trange(1,self.samples_length) as t:
            t.set_description('running MCMC, with acceptance percentage = {:e}, loss = {:e}, step size = {:e}'.format(0,latest_accepted_samples_neg_likelihood, self.step_size))
            # for i in trange(1,samples_length-1,desc='running MCMC, with acceptance rate = {}'.format(accepted/i)):
            for i in t:
                new_sample = self.get_new_sample(self.samples[i-1,:],self.randw[i-1,:])
                

                #compute negloglikelihood for new sample
                new_neg_log_likelihood,_ = self.compute_neg_loglikelihood(new_sample)

                #if is -/+ inf continue
                if np.isinf(new_neg_log_likelihood) or new_neg_log_likelihood<self.lower_limit_neg_log_likelihood:#this added to avoid pathologic 
                    self.samples[i,:] = self.samples[i-1,:]
                    continue
                

                logRatio = latest_accepted_samples_neg_likelihood - new_neg_log_likelihood

                #compare
                if logRatio>self.log_uniform[i]:
                    latest_accepted_samples_neg_likelihood = new_neg_log_likelihood
                    self.samples[i,:] = new_sample    
                    self.accepted += 1
                    partial_acceptance += 1
                else:
                    self.samples[i,:] = self.samples[i-1,:]
                    
                    

                #adapt step_size
                if i%self.adaptation_steps == 0:
                    self.step_size = self.adapt_step(partial_acceptance/self.adaptation_steps)
                    #set back to zero
                    partial_acceptance = 0
                    t.set_description('running MCMC, with acceptance percentage = {:e}, loss = {:e}, step size = {:e}'.format(self.accepted*100/i,latest_accepted_samples_neg_likelihood, self.step_size))
        

    def save(self):
        # excluded_matrix = ['']
        with h5py.File(str(self.simResultPath/'result.hdf5'),'w') as f:
            utils._save_object(f,self)

    
