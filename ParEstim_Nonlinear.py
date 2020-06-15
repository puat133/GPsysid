import numpy as np

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import sys
import os
import pandas as pd
import scipy.linalg as sla
import GPutils as utils
import numba_settings as nbs
import data
import argparse
import h5py

from tqdm import trange


import Nonlinear_fun as NL
import ParEstim_LinearLatent as peLinear
import PDE_numpy as PDE
import KalmanFilters as KF
import Dirichlet
 


#some important indexes



# @nbs.njitParallel
def runEKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history):
    ekf = constructEKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history)
    ekf.propagate_till_end()
    ekf.yTilde_history = ekf.measured_y_hist - ekf.history@ekf.H.T
    return ekf

# @nbs.njitParallel
def test_EKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history):
    ekf = constructEKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history)
    fail = ekf.propagate_till_end_with_check()
    return ekf,fail

# @nbs.njitParallel
def negLogLikelihoodFromParameters(nstate,ninput,parameters,matrices,measurement_history,input_history):
    ekf,fail = test_EKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history)
    # negLogLikelihood = np.sum(np.square(ekf.yTilde_history))#
    if not fail:
        ekf.yTilde_history = ekf.measured_y_hist - ekf.history@ekf.H.T
        negLogLikelihood = peLinear.negLogLikelihood(ekf.yTilde_history,ekf.S_history)
    else:
        negLogLikelihood = np.inf
    return negLogLikelihood,0

# @nbs.njitParallel
def constructEKF_from_parameters(nstate,ninput,parameters,matrices,measurement_history,input_history):
    ekf = KF.ExtendedKalmanFilter(nstate,ninput,parameters,matrices,\
                                NL.catalyst_dynamics_discrete,NL.P_dynamics_discrete,\
                                measurement_history.shape[0])
    ekf.P = matrices['P_init']
    ekf.state = parameters['InitialState']
    ekf.history[0,:] = ekf.state
    ekf.measured_y_hist = measurement_history
    ekf.input_hist = input_history
    return ekf







    

'''
dummy class to handle simulation data
'''
class EKFBased_ParameterEstimationSimulation(KF.SimulationBase):
    def __init__(self,args):
        
        
        if args.load:
            super().init_with_load(args)
            args.sampling_period = self.sampling_period
        
        else:
            self.nbases = args.bases_num
            self.n_levels = NL.N_LEVELS
            self.n_states = 4*self.n_levels
            self.n_inputs = 3
            self.__Start_Index_For_PDE_Constants = 0
            self.__PDE_Constants_numbers = 9
            self.__Start_Index_For_Q_diag_numbers = self.__PDE_Constants_numbers
            self.__Q_diag_numbers = self.n_states
            self.__Start_Index_For_P_diag_numbers = self.__PDE_Constants_numbers+self.__Q_diag_numbers
            self.__P_diag_numbers = self.n_states
            self.__R_scale_index = self.__PDE_Constants_numbers+self.__Q_diag_numbers+self.__P_diag_numbers
            
            
            self.__Initial_state_numbers = 0#self.n_levels*2 #only initialize the aromatics and sulfurs
            self.__Start_Index_For_Initial_State = self.__PDE_Constants_numbers+self.__Q_diag_numbers+self.__P_diag_numbers+1

            self.nparams = self.how_many_params()
            self.stdev = np.ones(self.nparams)
            self.stdev[:self.__PDE_Constants_numbers] = 1e0*np.ones(self.__PDE_Constants_numbers)
            # self.stdev[self.__Start_Index_For_Initial_State:] = 1e-2*np.ones(self.__Initial_state_numbers)

            super().init_without_load(args)

            self.measurement_history = data.y_hist_first_ma
            self.input_history = data.input_hist
            

            
            

            
        O = np.zeros((self.n_levels,self.n_levels))
        I = np.eye(self.n_levels)
        self.parameters = NL.parameters
        self.matrices = NL.matrices
        #Set parameters that are fixed
        self.parameters['Delta_z'] = data.Delta_z[0,:]
        self.parameters['dt'] = np.array([args.sampling_period])
        self.matrices['D1'] = NL.firstSpatialDerivativeMatrix(self.parameters)
        self.matrices['D2'] = NL.secondSpatialDerivativeMatrix(self.parameters)
        self.matrices['H'] = np.hstack([O,O,I,O])
            

    def assign_parameters_and_matrices(self,sample):
        self.parameters['PDEconstants'] = np.exp(sample[self.__Start_Index_For_PDE_Constants:(self.__Start_Index_For_PDE_Constants+self.__PDE_Constants_numbers)])
        self.matrices['Q'] = np.diag(np.exp(sample[self.__Start_Index_For_Q_diag_numbers:(self.__Start_Index_For_Q_diag_numbers+self.__Q_diag_numbers)]))
        self.matrices['P_init'] = np.diag(np.exp(sample[self.__Start_Index_For_P_diag_numbers:(self.__Start_Index_For_P_diag_numbers+self.__P_diag_numbers)]))
        self.matrices['R'] = np.exp(sample[self.__R_scale_index])*np.eye(self.n_levels)
        self.parameters['InitialState'] = np.concatenate((np.ones(self.n_levels)*self.input_history[0,0],np.ones(self.n_levels)*self.input_history[0,1],self.measurement_history[0,:],np.ones(self.n_levels)))
        

    def find_a_good_initial_sample(self):
        original_step_size = sim.step_size
        sim.step_size=1.
        success = False
        print('Finding a good initial sample')
        i = 0
        while not success:
            i += 1
            test_sample = self.get_new_sample(self.samples[0,:],np.random.randn(self.samples.shape[1]))
            self.samples[0,:] = test_sample
            self.assign_parameters_and_matrices(test_sample)
            ekf,fail = test_EKF_from_parameters(self.n_states,self.n_inputs,self.parameters,self.matrices,self.measurement_history,self.input_history)
            print('Testing for {}-th time, is it Fail : {}\n'.format(i,fail))
            success = not fail
        sim.step_size = original_step_size
        return ekf

        


    #Implementation of abstract methods     
    def get_new_sample(self,current_sample,randw):
        return np.sqrt(1-np.square(self.step_size))*current_sample + self.step_size*self.stdev*randw

    def compute_neg_loglikelihood(self,sample):
        self.assign_parameters_and_matrices(sample)
        return negLogLikelihoodFromParameters(self.n_states,self.n_inputs,self.parameters,self.matrices,self.measurement_history,self.input_history)

    def adapt_step(self,partial_acceptance_rate):
        return peLinear.adapt_step_sqrtBeta(self.step_size,partial_acceptance_rate,\
                                        feedback_gain=self.feedback_gain,target_acceptance=self.target_acceptance)
        

    def how_many_params(self):
        #the parameters are the pde constants, Q_diagonals, P_diagonals, Initial_consitions, R_scale,
        return self.__PDE_Constants_numbers+self.__Q_diag_numbers+self.__P_diag_numbers+self.n_states+self.__Initial_state_numbers+1
                    

    def plotSamples(self,step_plot=10,hist_bin=10):
        params_collections = ['PDE constants', 'Q diagonal entries', 'P diagonal entries', 'R scale', 'Initial state elements']
        params_numbers = np.array([self.__PDE_Constants_numbers,self.__Q_diag_numbers,self.__P_diag_numbers,1,self.__Initial_state_numbers])
        params_start_indexes = np.concatenate([[0],np.cumsum(params_numbers[:-1])])
        burned_index = self.samples.shape[0]*self.burn//100 
        for i in range(len(params_collections)):
            if i>0:#Plot only the PDE constants at the moment
                continue
            nparams = params_numbers[i]
            f, ax = plt.subplots(nparams,nparams,figsize=(9*(nparams-1),9*(nparams-1))) #plot relation of each parameter samples in two dimensional scatter plot
            
            
            for j in range(nparams):
                ax[j,j].hist(self.samples[burned_index::step_plot,params_start_indexes[i]+j],hist_bin)
                for k in range(j+1,nparams):
                    ax[j,k].scatter(self.samples[burned_index::step_plot,params_start_indexes[i]+k],\
                        self.samples[burned_index::step_plot,params_start_indexes[i]+j],alpha=0.1)

            
            

            f.savefig(str(self.simResultPath/'{} samples.png'.format(params_collections[i])))
            plt.close(f)

    def load_from_file(self,f):
        with h5py.File(f,mode='r') as file:
            #reload from files
            self.__Initial_state_numbers = file['_EKFBased_ParameterEstimationSimulation__Initial_state_numbers'][()]
            self.__PDE_Constants_numbers = file['_EKFBased_ParameterEstimationSimulation__PDE_Constants_numbers'][()]
            self.__P_diag_numbers = file['_EKFBased_ParameterEstimationSimulation__P_diag_numbers'][()]
            self.__Q_diag_numbers = file['_EKFBased_ParameterEstimationSimulation__Q_diag_numbers'][()]
            self.__R_scale_index = file['_EKFBased_ParameterEstimationSimulation__R_scale_index'][()]
            self.__Start_Index_For_Initial_State = file['_EKFBased_ParameterEstimationSimulation__Start_Index_For_Initial_State'][()]
            self.__Start_Index_For_PDE_Constants = file['_EKFBased_ParameterEstimationSimulation__Start_Index_For_PDE_Constants'][()]
            self.__Start_Index_For_P_diag_numbers = file['_EKFBased_ParameterEstimationSimulation__Start_Index_For_P_diag_numbers'][()]
            self.__Start_Index_For_Q_diag_numbers = file['_EKFBased_ParameterEstimationSimulation__Start_Index_For_Q_diag_numbers'][()]
            
            
            
            self.accepted = file['accepted'][()]
            self.adaptation_steps = file['adaptation_steps'][()]
            self.burn = file['burn'][()]
            self.feedback_gain = file['feedback_gain'][()]
            self.input_history = file['input_history'][()]
            self.log_uniform = file['log_uniform'][()]
            self.measurement_history = file['measurement_history'][()]
            self.n_inputs = file['n_inputs'][()]
            self.n_levels = file['n_levels'][()]
            self.n_states = file['n_states'][()]
            self.nparams = file['nparams'][()]
            self.random_seed = file['random_seed'][()]
            self.randw = file['randw'][()]
            self.samples = file['samples'][()]
            self.samples_length = file['samples_length'][()]
            self.stdev = file['stdev'][()]
            self.sampling_period = file['sampling_period'][()]
            self.step_size = file['step_size'][()]
            self.target_acceptance = file['target_acceptance'][()]
        
    '''
    from generated sample
    '''
    def analyze(self,skip=1,validate=True):
        time_length = self.measurement_history.shape[0]
        burned_index = self.samples.shape[0]*self.burn//100 #burn first 5 percent data
        samples = self.samples[burned_index::skip,:]
        samples_length = samples.shape[0]
        output_num = self.measurement_history.shape[1]
        state_hist = np.zeros((samples_length,time_length,self.n_states))
        yTilde_hist = np.zeros((samples_length,time_length,output_num))
        S_hist = np.zeros((samples_length,time_length,output_num,output_num))        
        neglog_hist = np.zeros((samples_length))
        if validate:
            self.measurement_history = data.y_hist_second_ma
            self.parameters['Delta_z'] = data.Delta_z[1,:]
        
        
        for i in range(samples_length):
            self.assign_parameters_and_matrices(samples[i,:])
            ekf = constructEKF_from_parameters(self.n_states,self.n_inputs,self.parameters,self.matrices,self.measurement_history,self.input_history)
            ekf.propagate_till_end()
            ekf.yTilde_history = ekf.measured_y_hist - ekf.history@ekf.H.T
            state_hist[i,:,:] = ekf.history
            yTilde_hist[i,:,:] = ekf.yTilde_history
            S_hist[i,:,:,:] = ekf.S_history
            neglog_hist[i] = peLinear.negLogLikelihood(ekf.yTilde_history,ekf.S_history)

        #return a dictionary
        results = {'state_hist':state_hist,
                'yTilde_hist':yTilde_hist,
                'S_hist':S_hist,
                'neglog_hist':neglog_hist,
                'validate':validate
                }
        return results

    def plotAnalysisResult(self,results):
        state_hist_mean = np.mean(results['state_hist'],axis=0)
        state_hist_std = np.std(results['state_hist'],axis=0)
        time = np.arange(state_hist_mean.shape[0])
        i=0
        y_axis = ['weight \%', 'weight \%', r'$^\circ C$', '-']

        for state_name in ['sulfur','aromatics','temperature','activity']:
            
            for j in range(self.n_levels):
                plt.figure(figsize=(15,10))
                if state_name=='temperature':
                    if j>0:
                        delta_T_meas = self.measurement_history[:,j]-self.measurement_history[:,j-1]
                        delta_T_est = state_hist_mean[:,i*self.n_levels+j]-state_hist_mean[:,i*self.n_levels+j-1]
                        
                    else:
                        delta_T_meas = self.measurement_history[:,j]-self.input_history[:,2]
                        delta_T_est = state_hist_mean[:,i*self.n_levels+j]-self.input_history[:,2]

                    plt.plot(time,delta_T_meas,linewidth=0.5,color='r')
                    plt.plot(time,delta_T_est,linewidth=0.5)
                    plt.fill_between(time,delta_T_est-2*state_hist_std[:,i*self.n_levels+j],\
                                        delta_T_est+2*state_hist_std[:,i*self.n_levels+j], 
                                color='b', alpha=0.1)
                else:
                    plt.plot(time,state_hist_mean[:,i*self.n_levels+j],linewidth=0.5)
                    plt.fill_between(time,state_hist_mean[:,i*self.n_levels+j]-2*state_hist_std[:,i*self.n_levels+j],\
                                        state_hist_mean[:,i*self.n_levels+j]+2*state_hist_std[:,i*self.n_levels+j], 
                                color='b', alpha=0.1)
                # plt.tight_layout()
                plt.title(state_name+' for level {}'.format(j+1))
                plt.ylabel(y_axis[i])
                plt.xlabel('Days')
                plt.savefig(str(self.simResultPath/'{}-{}.png'.format(state_name,j+1)))
                plt.close()
            i +=1




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-step-size',default=1.,type=float,help='Initial step size of MCMC, Default=1.')
    parser.add_argument('--feedback-gain',default=2.1,type=float,help='MCMC feedback, Default=2.1')
    # parser.add_argument('--R-scale',default=1e-2,type=float,help='Constant multiplier to R related to Kalman filter, Default=1e-2')
    parser.add_argument('--target-acceptance',default=0.4,type=float,help='MCMC target acceptance rate, Default=0.5')
    parser.add_argument('--sampling-period',default=1e-3,type=float,help='sampling period for PDE, Default=1e-3')
    parser.add_argument('--adaptation-steps',default=10,type=int,help='adaptation-steps number to reevaluate the MCMC step size, Default=10')
    parser.add_argument('--samples-num',default=100,type=int,help='MCMC samples number, Default=1000')
    parser.add_argument('--seed',default=0,type=int,help='Random seed, Default=0')
    parser.add_argument('--bases-num',default=10,type=int,help='Number of bases for each stata/input, Default=5')
    parser.add_argument('--burn',default=25,type=int,help='Burn percentage, Default=25')
    parser.add_argument('--folderName',default="",type=str,help='folder name, Default=')
    parser.add_argument('--init',default="",type=str,help='folder contains a chains to initialize, Default=')
    parser.add_argument('--load',default="",type=str,help='folder contains a full simulation results to load and analyze, Default=')
    args = parser.parse_args()
    
    # main(args)
    sim = EKFBased_ParameterEstimationSimulation(args)

    #if only used for analysis
    if args.load:
        results = sim.analyze(skip=10)
        sim.plotAnalysisResult(results)
        sim.plotSamples(hist_bin=100)
    else:
        # trial = 10
        # for i in range(trial):
        #     ekf = sim.find_a_good_initial_sample()
        #     f,ax = plt.subplots(2,2,figsize=(80,40))
        #     ax[0,0].plot(ekf.history[:,:sim.n_levels],linewidth=0.5)
        #     ax[0,1].plot(ekf.history[:,sim.n_levels:2*sim.n_levels:],linewidth=0.5)
        #     ax[1,0].semilogy(ekf.history[:,2*sim.n_levels:3*sim.n_levels],linewidth=0.5)
        #     ax[1,1].plot(ekf.history[:,3*sim.n_levels:],linewidth=0.5)
        #     plt.tight_layout()
        #     plt.show()
        
        if not args.init:          
            ekf = sim.find_a_good_initial_sample()

        sim.train()
        results = sim.analyze(skip=10)
        sim.plotAnalysisResult(results)
        sim.save()
        sim.plotSamples(hist_bin=100)