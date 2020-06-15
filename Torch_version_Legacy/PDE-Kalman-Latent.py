import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
import pandas as pd
import datetime
import torch.nn as nn
import scipy.linalg as sla
import pathlib
import argparse
import GPutils as utils
import h5py
from tqdm import trange
from torch.utils.data import Dataset, DataLoader

#import from parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import PDE
Dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def constructDiscreteSystemMatrices(log_parameters,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H=None):
    #the original parameters is in the log space to assume that the parameters are positive
    parameters = torch.exp(log_parameters)

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

    A = torch.cat((
                    torch.cat((D_T*Lap-C_T*Der, O, K_3*H_v*I),dim=1),\
                    torch.cat((O,D_a*Lap-C_a*Der,-K_2*H_v*I),dim=1),\
                    torch.cat((O,O,D_v*Lap-C_v*Der),dim=1)))
    
    B = torch.cat((O,O,E_v*I))

    if H == None:
        index = torch.arange(1,nbases+1)
        Phi = PDE.basis(index,L,measurement_points)
        Oz = torch.zeros((measurement_points.shape[0],2*nbases))
        H = torch.cat((Phi.T,Oz),dim=1)

    F = PDE.expm(A*sampling_period)

    inv, _ = torch.solve(F-torch.eye(nbases*3),A)
    G = -inv@B

    return F,G,H

def negLogLikelihood(yTilde_hist,S_hist):
    loss = 0
    for i in range(yTilde_hist.shape[0]):
        loss += yTilde_hist[i,:]@S_hist[i,:,:]@yTilde_hist[i,:]
    
    return loss

def negLogLikelihoodFromParameters(log_parmeters,y_hist_normalized,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H=None,R_scale=1.):
    F,G,H= constructDiscreteSystemMatrices(log_parmeters,nbases,L,measurement_points,sampling_period,Der,Lap,O,I,H)
    Q = G@G.T
    Q = 0.5*(Q.T+Q)
    R = torch.eye(measurement_points.shape[0])*R_scale
    G_u = torch.zeros((nbases*3,1))
    kalman = PDE.KalmanFilter(F,G_u,H,Q,R,length=y_hist_normalized.shape[1])
    u = torch.tensor([0.])
    for i in range(kalman.length-1):
        kalman.propagate(u,y_hist_normalized[:,i])

    return negLogLikelihood(kalman.yTilde_history,kalman.S_history)

def adapt_step_sqrtBeta(current_step_size,current_acceptance_rate,feedback_gain=2.1,target_acceptance=0.234):
        #based on Algorithm 2 of: Computational Methods for Bayesian Inference in Complex Systems: Thesis
        
        return current_step_size*np.exp(feedback_gain*(current_acceptance_rate-target_acceptance))

def run(step_size,samples_length,adaptation_steps,feedback_gain,target_acceptance,nbases,y_hist_normalized,L,normalized_height,sampling_period,Der,Lap,O,I,H,R_scale):
    
    sample_init = torch.randn(10)
    # sample_init = torch.mean(samples[burned_index:],axis=1)
    samples = torch.empty((samples_length,sample_init.shape[0]))
    samples[0,:] = sample_init
    randw = torch.randn((samples_length,sample_init.shape[0]))
    log_uniform = torch.log(torch.rand(samples_length))
    accepted = 0
    partial_acceptance = 0
    
    latest_accepted_samples_neg_likelihood = negLogLikelihoodFromParameters(samples[0,:],y_hist_normalized,nbases,L,normalized_height[0,:],sampling_period,Der,Lap,O,I,H=H,R_scale=R_scale)
    with trange(1,samples_length) as t:
        t.set_description('running MCMC, with acceptance percentage = {}'.format(0.))
        # for i in trange(1,samples_length-1,desc='running MCMC, with acceptance rate = {}'.format(accepted/i)):
        for i in t:
            #pCN proposal
            new_sample = np.sqrt(1-np.square(step_size))*samples[i-1,:] + step_size*randw[i-1,:]

            #compute negloglikelihood for new sample
            new_neg_log_likelihood = negLogLikelihoodFromParameters(new_sample,y_hist_normalized,nbases,L,normalized_height[0,:],sampling_period,Der,Lap,O,I,H=H,R_scale=R_scale)

            #if is -/+ inf continue
            if torch.isinf(new_neg_log_likelihood):
                samples[i,:] = samples[i-1,:]
                continue

            logRatio = latest_accepted_samples_neg_likelihood - new_neg_log_likelihood

            #compare
            if logRatio>log_uniform[i]:
                latest_accepted_samples_neg_likelihood = new_neg_log_likelihood
                samples[i,:] = new_sample    
                accepted += 1
                partial_acceptance += 1
            else:
                samples[i,:] = samples[i-1,:]
                

            #adapt step_size
            if i%adaptation_steps == 0:
                step_size = adapt_step_sqrtBeta(step_size,partial_acceptance/adaptation_steps,feedback_gain=feedback_gain,target_acceptance=target_acceptance)
                #set back to zero
                partial_acceptance = 0
                t.set_description('running MCMC, with acceptance percentage = {}'.format(accepted*100/i))
    
    return samples

def save(file_name,object):
    with h5py.File(file_name,'w') as f:
        if isinstance(object,np.ndarray):
            if object.ndim >0:
                f.create_dataset('chain',data=object,compression='gzip')
            else:
                f.create_dataset('chain',data=object)

def createFolder(args):
    if not args.folderName: #empty string are false
        folderName = 'Results-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M_%S')
    else:
        folderName = args.folderName

    if 'WRKDIR' in os.environ:
        simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult_Neste'/folderName
    elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult_Neste').exists():
        simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult/Neste')/folderName
    else:
        simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult_Neste'/folderName
    if not simResultPath.exists():
        simResultPath.mkdir()

    return simResultPath

def plotSamples(samples,burn_percentage,simResultPath):
    nparams = samples.shape[1]
    f, ax = plt.subplots(nparams-1,nparams-1,figsize=(9*(nparams-1),9*(nparams-1))) #plot relation of each parameter samples in two dimensional scatter plot
    burned_index = samples.shape[0]//burn_percentage #burn first 5 percent data
    step_plot=10
    for i in range(nparams-1):
        for j in range(i+1,nparams):
            ax[i,j-1].scatter(samples[burned_index::step_plot,i],samples[burned_index::step_plot,j],alpha=0.1)

    f.savefig(str(simResultPath/'result.png'))
    plt.close(f)


def main(args):
    #decompose the argument
    sampling_period = args.sampling_period # again assumption
    nbases = args.bases_num
    samples_length = args.samples_num
    step_size = args.initial_step_size
    adaptation_steps = args.adaptation_steps
    feedback_gain = args.feedback_gain
    target_acceptance = args.target_acceptance
    R_scale = args.R_scale
    torch.manual_seed(args.seed)

    simResultPath = createFolder(args)
    

    temperature_columns = np.array(
    [['TI8585','TI8553','TI8554','TI8555','TI8556','TI8557','TI8558','TI8559', 'TIZ8578A'],
     ['TI8585','TI8560','TI8561','TI8562','TI8563','TI8564','TI8565','TI8566', 'TIZ8578A'],
     ['TI8585','TI8567','TI8568','TI8569','TI8570','TI8571','TI8572','TI8573', 'TIZ8578A']],dtype=object)

    tc_heights =np.array([[7600,6550,5500,4450,3400,2350,1300],
                [7250,6250,5150,4100,3050,2000,950],
                [6900,5850,4800,3750,2700,1650,600]])

    tc_total_height=8000 #assumption
    normalized_height = (tc_total_height-tc_heights)/tc_total_height
    normalized_height = torch.from_numpy(normalized_height).float()

    
    L = torch.tensor(1.)

    #Load data
    df_raw = pd.read_hdf('Data/timeseries_complete.hdf5',key='KAAPO_hour_15_16_17_18_19_complete')
    df_raw = df_raw[(df_raw.index < "2017-03-26") & (df_raw.index > "2015-07-14")]
    df_lab = pd.read_hdf('Data/Laboratory.hdf5',key='Laboratory').interpolate()
    df_lab = df_lab[(df_lab.index < "2017-03-26") & (df_lab.index > "2015-07-14")]
    df = pd.concat([df_raw, df_lab], axis=1)
    df = df.resample('d').median() #resample daily or weekly

    y_hist = torch.tensor((df[temperature_columns[0,1:-1]]).values,device=device).float()
    temp_inlet = torch.tensor(df['TI8585'].values,device=device).float()
    y_hist_normalized = y_hist.T - temp_inlet
    

    #Construct some of the required matrix
    Der = PDE.derivative(nbases,L)
    Lap = PDE.laplacian(nbases,L)
    O = torch.zeros_like(Der)
    I = torch.eye(Der.shape[0])
    index = torch.arange(1,nbases+1)
    Phi = PDE.basis(index,L,normalized_height[0,:])
    Oz = torch.zeros((normalized_height.shape[1],2*nbases))
    H = torch.cat((Phi.T,Oz),dim=1)

    samples = run(step_size,samples_length,adaptation_steps,feedback_gain,target_acceptance,nbases,y_hist_normalized,L,normalized_height,sampling_period,Der,Lap,O,I,H,R_scale)
    save(str(simResultPath/'result.hdf5'),samples.numpy())
    plotSamples(samples,args.burn,simResultPath)

    

    
    
    
    



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-step-size',default=1.,type=float,help='Initial step size of MCMC, Default=1.')
    parser.add_argument('--feedback-gain',default=2.1,type=float,help='MCMC feedback, Default=2.1')
    parser.add_argument('--R-scale',default=1e-2,type=float,help='Constant multiplier to R related to Kalman filter, Default=1e-2')
    parser.add_argument('--target-acceptance',default=0.5,type=float,help='MCMC target acceptance rate, Default=0.5')
    parser.add_argument('--sampling-period',default=1e-3,type=float,help='sampling period for PDE, Default=1e-3')
    parser.add_argument('--adaptation-steps',default=10,type=int,help='adaptation-steps number to reevaluate the MCMC step size, Default=10')
    parser.add_argument('--samples-num',default=10000,type=int,help='MCMC samples number, Default=1000')
    parser.add_argument('--seed',default=0,type=int,help='Random seed, Default=0')
    parser.add_argument('--bases-num',default=10,type=int,help='Number of bases for each stata/input, Default=5')
    parser.add_argument('--burn',default=25,type=int,help='Burn percentage, Default=25')
    parser.add_argument('--folderName',default="",type=str,help='folder name, Default=')
    args = parser.parse_args()
    main(args)