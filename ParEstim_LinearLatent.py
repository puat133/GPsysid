import numpy as np

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import sys
import os
import pandas as pd
import scipy.linalg as sla
import GPutils as utils
import numba as nb
import numba_settings as nbs
import data
import argparse
import h5py

from tqdm import trange



import PDE_numpy as PDE
import KalmanFilters as KF
import Dirichlet
 

#some important indexes
Last_index_for_ODE_const_logparam = 9
Index_for_P_scale = 10
Index_for_R_scale = 11
Start_index_for_initial_Kalman_state = Index_for_R_scale+1

@nbs.njitSerial
def constructDiscreteSystemMatrices(log_parameters,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H=None):
    #the original parameters is in the log space to assume that the parameters are positive
    parameters = np.exp(log_parameters)

    D_T = parameters[0]
    C_T = parameters[1]
    K_3 = parameters[2]
    D_a = parameters[3]
    C_a = parameters[4]
    K_2 = parameters[5]
    D_v = parameters[6]
    C_v = parameters[7]
    E_v = parameters[8]
    H_v = parameters[9]

    A = np.vstack((
                    np.hstack((D_T*Lap-C_T*Der, O, K_3*H_v*I)),\
                    np.hstack((O,D_a*Lap-C_a*Der,-K_2*H_v*I)),\
                    np.hstack((O,O,D_v*Lap-C_v*Der))))
    
    B = np.vstack((O,O,E_v*I)) 

    # if H == None:
    #     index = np.arange(1,nbases+1)
    #     Phi = PDE.basis(index,L,measurement_points)
    #     Oz = np.zeros((measurement_points.shape[0],2*nbases))
    #     H = H = np.hstack((Phi.T,Oz)) #((Phi.T,Oz),dim=1)

    F = PDE.expm(A*sampling_period)

    inv = np.linalg.solve(A,F-np.eye(nbases*3))
    G = -inv@B

    return F,G,H

@nbs.njitParallel
def negLogLikelihood(yTilde_hist,S_hist):
    loss = 0
    for i in nb.prange(1,yTilde_hist.shape[0]):
        # loss += yTilde_hist[i,:]@yTilde_hist[i,:]
        loss += yTilde_hist[i,:]@np.linalg.solve(S_hist[i,:,:],yTilde_hist[i,:])+ np.linalg.slogdet(S_hist[i,:,:])[1]       
    return loss

@nbs.njitParallel
def runKalmanFilterFromParameters_with_P_init(parameters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H,P_init):
    F,G,H= constructDiscreteSystemMatrices(parameters[:Last_index_for_ODE_const_logparam+1],nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H)
    Q = G@G.T
    Q = 0.5*(Q.T+Q)
    R = np.eye(measurement_points.shape[0])*np.exp(parameters[Index_for_R_scale])
    G_u = np.zeros((nbases*3,1))
    kalman = KF.KalmanFilter(F,G_u,H,Q,R,y_hist_normalized.shape[1])
    kalman.P = P_init
    kalman.state = parameters[Start_index_for_initial_Kalman_state:]
    kalman.measured_y_hist = y_hist_normalized.T ## this need to be transposed
    u = np.array([0.])
    kalman.propagate_till_end(u)
    kalman.yTilde_history = kalman.measured_y_hist - kalman.history@kalman.H.T
    return kalman

@nbs.njitSerial
def runKalmanFilterFromParameters(parameters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H):
    P_init = np.eye(3*nbases)*np.exp(parameters[Index_for_P_scale])
    kalman = runKalmanFilterFromParameters_with_P_init(parameters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H,P_init)
    return kalman

@nbs.njitSerial
def negLogLikelihoodFromParameters(parameters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H):
    kalman = runKalmanFilterFromParameters(parameters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H)
    return negLogLikelihood(kalman.yTilde_history,kalman.S_history),kalman.P

@nbs.njitSerial
def adapt_step_sqrtBeta(current_step_size,current_acceptance_rate,feedback_gain=2.1,target_acceptance=0.234):
        #based on Algorithm 2 of: Computational Methods for Bayesian Inference in Complex Systems: Thesis   
        current_step_size = max(1e-17,current_step_size*np.exp(feedback_gain*(current_acceptance_rate-target_acceptance)))
        return min(1.,current_step_size)

    

'''
dummy class to handle simulation data
'''
class KalmanBased_ParameterEstimationSimulation(KF.SimulationBase):
    def __init__(self,args):
        
        
        if args.load:
            super().init_with_load(args)
        
        else:
            self.nbases = args.bases_num
            self.nparams = self.how_many_params()
            super().init_without_load(args)

            self.measurement_history = data.y_hist_dirichlet#data.y_hist_normalized
            self.measurement_position = data.training_pole_temperature_positions

            self.index = np.arange(1,self.nbases+1)
            self.Der = Dirichlet.derivative(self.nbases,data.L)
            self.Lap = Dirichlet.laplacian(self.nbases,data.L)
            self.Phi = Dirichlet.basis(self.index,data.L,self.measurement_position)

            self.O = np.zeros_like(self.Der)
            self.I = np.eye(self.Der.shape[0])
            
            self.Oz = np.zeros((self.measurement_position.shape[0],2*self.nbases))
            self.H = np.hstack((self.Phi.T,self.Oz))

            #Experimental
            # self.P_stationary = np.zeros((self.samples_length,3*self.nbases,3*self.nbases))


    #Implementation of abstract methods     
    def get_new_sample(self,current_sample,randw):
        return np.sqrt(1-np.square(self.step_size))*current_sample + self.step_size*randw

    def compute_neg_loglikelihood(self,sample):
        return negLogLikelihoodFromParameters(sample,self.measurement_history,\
                                                self.nbases,data.L,self.measurement_position,\
                                                self.sampling_period,self.Der,self.Lap,self.O,self.I,\
                                                    self.H)

    def adapt_step(self,partial_acceptance_rate):
        return adapt_step_sqrtBeta(self.step_size,partial_acceptance_rate,\
                                        feedback_gain=self.feedback_gain,target_acceptance=self.target_acceptance)
        

    def how_many_params(self):
        return 12 + 3*self.nbases
                    

    def plotSamples(self,step_plot=10,hist_bin=10):
        nparams = Last_index_for_ODE_const_logparam+1
        f, ax = plt.subplots(nparams,nparams,figsize=(9*(nparams-1),9*(nparams-1))) #plot relation of each parameter samples in two dimensional scatter plot
        burned_index = self.samples.shape[0]*self.burn//100 #burn first 5 percent data
        
        for i in range(nparams):
            ax[i,i].hist(self.samples[burned_index::step_plot,i],hist_bin)
            for j in range(i+1,nparams):
                ax[i,j].scatter(self.samples[burned_index::step_plot,j],self.samples[burned_index::step_plot,i],alpha=0.1)

        
        # for i in range(nparams-1):
        #     for j in range(i+1,nparams):
        #         ax[i,j-1].scatter(samples[burned_index::step_plot,j],samples[burned_index::step_plot,i],alpha=0.1)

        f.savefig(str(self.simResultPath/'result.png'))
        plt.close(f)

    def load_from_file(self,file):
        with h5py.File(file,mode='r') as file:
            #reload from files
            self.Der = file['Der'][()]
            self.Lap = file['Lap'][()]
            self.O = file['O'][()]
            self.I = file['I'][()]
            self.H = file['H'][()]
            self.Oz = file['Oz'][()]
            self.Phi = file['Phi'][()]
            self.accepted = file['accepted'][()]
            self.adaptation_steps = file['adaptation_steps'][()]
            self.burn = file['burn'][()]
            self.feedback_gain = file['feedback_gain'][()]
            self.index = file['index'][()]
            self.log_uniform = file['log_uniform'][()]
            self.nbases = file['nbases'][()]
            self.nparams = file['nparams'][()]
            self.random_seed = file['random_seed'][()]
            self.randw = file['randw'][()]
            self.samples = file['samples'][()]
            self.samples_length = file['samples_length'][()]
            self.sampling_period = file['sampling_period'][()]
            self.step_size = file['step_size'][()]
            self.target_acceptance = file['target_acceptance'][()]
            self.measurement_history = file['measurement_history'][()]
            self.measurement_position = file['measurement_position'][()]
            self.P_stationary = file['P_stationary'][()]
    '''
    from generated sample
    '''
    def analyze(self,skip=1):
        time_length = self.measurement_history.shape[1]
        burned_index = self.samples.shape[0]*self.burn//100 #burn first 5 percent data
        samples = self.samples[burned_index::skip,:]
        # P_stationary = self.P_stationary[burned_index::skip,:,:]
        samples_length = samples.shape[0]
        state_num = 3*self.nbases
        output_num = self.measurement_history.shape[0]
        state_hist = np.zeros((samples_length,time_length,state_num))
        yTilde_hist = np.zeros((samples_length,time_length,output_num))
        S_hist = np.zeros((samples_length,time_length,output_num,output_num))
        P_hist = np.zeros((samples_length,time_length,state_num,state_num))
        neglog_hist = np.zeros((samples_length))
        
        
        for i in range(samples_length):
            parameters = samples[i,:]  
            kalman = runKalmanFilterFromParameters(parameters,self.measurement_history,\
                                                    self.nbases,data.L,self.measurement_position,\
                                                    self.sampling_period,self.Der,self.Lap,self.O,\
                                                    self.I,self.H)#,P_stationary[i,:,:])
            state_hist[i,:,:] = kalman.history
            P_hist[i,:,:,:] = kalman.P_history
            S_hist[i,:,:] = kalman.S_history
            yTilde_hist[i,:,:] = kalman.yTilde_history
            neglog_hist[i] = negLogLikelihood(kalman.yTilde_history,kalman.S_history)

        #return a dictionary
        results = {'state_hist':state_hist,
                'yTilde_hist':yTilde_hist,
                'S_hist':S_hist,
                'P_hist':P_hist,
                'neglog_hist':neglog_hist
                }
        return results

    def plotAnalysisResult(self,results):
        time_length = self.measurement_history.shape[1]
        recorded_y = data.temp_reference.T+self.measurement_history.T
        output_num = recorded_y.shape[1]
        t_index = np.arange(recorded_y.shape[0])
        time_start = 5
        yTilde_hist_mean = np.mean(results['yTilde_hist'],axis=0)
        yTilde_hist_std = np.std(results['yTilde_hist'],axis=0)
        estimated_mean = recorded_y+yTilde_hist_mean
        
        #temperature output
        for i in range(output_num-1):
            f = plt.figure(figsize=(15,10))
            delta_T_ref = recorded_y[time_start:,i+1]-recorded_y[time_start:,i]
            delta_T_est = estimated_mean[time_start:,i+1]-estimated_mean[time_start:,i]
            plt.plot(t_index[time_start:],delta_T_ref,'-r',linewidth=0.25,markersize=1)
            plt.plot(t_index[time_start:],delta_T_est,'-b',linewidth=0.25,markersize=1)
            plt.fill_between(t_index[time_start:],delta_T_est-2*yTilde_hist_std[time_start:,i],
                        delta_T_est+2*yTilde_hist_std[time_start:,i], 
                        color='b', alpha=0.1)
            plt.tight_layout()
            f.savefig(str(self.simResultPath/'output_{}.png'.format(i)))
            plt.close(f)

        
        #draw temperature profile for different times
        estimated_points = np.linspace(0.,1.,100,endpoint=True)
        temp_reference = Dirichlet.construct_reference_temperature(data.temp_inlet,data.temp_outlet,estimated_points)
        # Phi = PDE.basis(self.index,data.L,estimated_points).T
        Phi = Dirichlet.basis(self.index,data.L,estimated_points).T
        temp_fourier_history = results['state_hist'][:,:,:self.nbases]

        #add the reference first when reconstruct the temperature profile
        temperature = temp_reference.T+np.tensordot(temp_fourier_history,Phi,axes=([2],[1]))
        temp_mean = np.mean(temperature,axis=0)
        temp_std = np.std(temperature,axis=0)

        time_skip = 50
        for j in range(time_length//time_skip):
            i = j*time_skip
            f = plt.figure(figsize=(15,10))
            plt.plot(self.measurement_position,recorded_y[i,:],'-*r',linewidth=0.25,markersize=1)
            plt.plot(estimated_points,temp_mean[i,:],'-b',linewidth=0.25,markersize=1)
            plt.fill_between(estimated_points,temp_mean[i,:]-2*temp_std[i,:],
                        temp_mean[i,:]+2*temp_std[i,:], 
                        color='b', alpha=0.1)
            f.savefig(str(self.simResultPath/'temperature_Profile_t={}.png'.format(i)))
            plt.close(f)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-step-size',default=1.,type=float,help='Initial step size of MCMC, Default=1.')
    parser.add_argument('--feedback-gain',default=2.1,type=float,help='MCMC feedback, Default=2.1')
    # parser.add_argument('--R-scale',default=1e-2,type=float,help='Constant multiplier to R related to Kalman filter, Default=1e-2')
    parser.add_argument('--target-acceptance',default=0.5,type=float,help='MCMC target acceptance rate, Default=0.5')
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
    sim = KalmanBased_ParameterEstimationSimulation(args)

    #if only used for analysis
    if args.load:
        results = sim.analyze(skip=100)
        sim.plotAnalysisResult(results)
        sim.plotSamples(hist_bin=100)
    else:
        sim.train()
        results = sim.analyze(skip=100)
        sim.plotAnalysisResult(results)
        sim.save()
        sim.plotSamples(hist_bin=100)