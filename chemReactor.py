#%%
import util
import a
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
sns.set_style('darkgrid')
from matplotlib import rc
import scipy.io as sio
import sippy
import argparse
import parser_help as ph
import pathlib
import datetime
import os
#%%
'''
resampling = 
'''
def chemReactorGP(path,randSeed=0,resampling=5,ma_smoother=14,
                    data_extension_percentage=50,
                    minSS_orders=5,maxSS_orders=8,useLinear=True,
                    samples_num=1000,particles_num=30,
                    bases_num=4,ratio_L=1,Kn=10,
                    burnPercentage=25,lQ=100,ell=1.,Vgain=1e3):
    np.random.seed(randSeed)
    data = sio.loadmat('ForIdentification.mat')
    # resampling = 5
    # ma_smoother = 
    # tempIndex = 0
    u = data['u'][::resampling,[-2,-1]];u = u.T#u=u[np.newaxis,:]
    y = data['delta_T'][::resampling,:3];y = y.T#y=y[np.newaxis,:]
    #%%
    y = y-y[:,-1][:,np.newaxis]
    y = y/(np.max(y)-np.min(y)) #scaled to 0-1
    u = u - np.mean(u,axis=1)[:,np.newaxis]
    #%%
    T = u.shape[1]
    #%%
    #Extend u and y
    extension = data_extension_percentage # 50% extension
    y_extend = np.zeros((y.shape[0],(y.shape[1]*(100+extension)//100)))
    u_extend = np.zeros((u.shape[0],(u.shape[1]*(100+extension)//100)))
    shift = extension*y.shape[1]//200
    y_extend[:,shift+1:-shift] = y
    u_extend[:,shift+1:-shift] = u
    y_ma = util.moving_average(y_extend.T,ma_smoother).T
    plt.plot(y.T)
    plt.plot(y_ma.T)
    # %%
    # T_test = T
    u_test = u_extend
    y_test= y_ma
    # t = np.arange(T)
    #%%
    sys_id = sippy.system_identification(y.T,u.T,'N4SID'
                                        #  ,centering='InitVal'
                                        #  ,SS_p=horizon,SS_f=horizon
                                        ,SS_A_stability=True
                                        ,IC='BIC'
                                        ,SS_orders=[minSS_orders,maxSS_orders]
                                        )

    #%%
    nx = sys_id.A.shape[0]
    iA = sys_id.A #np.random.randn(nx,nx)
    iB = sys_id.B #np.ones((nx,1))
    #%%
    # nbases=4
    L = y.shape[1]//ratio_L
    steps = samples_num
    sim = a.Simulate(steps,nx,u_extend,y_ma,bases_num,L,PFweightNum=particles_num)
    if useLinear:
        sim.iA = iA
        sim.iB = iB

    sim.burnInPercentage = burnPercentage
    sim.lQ = lQ #for prior QR
    sim.ell = ell
    sim.Vgain = Vgain
    sim.run()


    #%%
    y_test_med,y_test_loQ,y_test_hiQ = sim.evaluate(y_test,u_test,Kn=Kn)

    sim.save(str(simResultPath/'result.hdf5'))
    #%%
    for i in range(sim.ny):
        fig = plt.figure(figsize=(20,10))
        plt.plot(y_test[i,:],color='k',linewidth=1,label='Ground Truth')
        plt.plot(y_test_med[i,:],color='b',linewidth=0.5,label='Median')
        plt.fill_between(np.arange(y_test_loQ.shape[1]),y_test_loQ[i,:],y_test_hiQ[i,:], color='b', alpha=.1, label=r'95 \% confidence')
        plt.legend()
        filename = str(simResultPath/'delta_T_{}.png'.format(i))
        plt.savefig(filename)


#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--randSeed',default=0,type=int,help='random Seed number, Default=0')
    parser.add_argument('--resampling',default=5,type=int,help='resampling the timeseries data (daily), Default=5')
    parser.add_argument('--ma-smoother',default=14,type=int,help='Smoothen the output data before processing, Default=14')
    parser.add_argument('--extension',default=50,type=int,help='data is extended before and after so that at both end they are zero, Default=50')
    parser.add_argument('--minSS-orders',default=3,type=int,help='Minimum SS orders, Default=3')
    parser.add_argument('--maxSS-orders',default=8,type=int,help='Maximum SS orders, Default=8')
    parser.add_argument('--samples-num',default=1000,type=int,help='MCMC samples number, Default=1000')
    parser.add_argument('--particles-num',default=30,type=int,help='Particles number for particle filter, Default=30')
    parser.add_argument('--bases-num',default=5,type=int,help='Number of bases for each stata/input, Default=5')
    parser.add_argument('--ratio-L',default=1,type=int,help='L reciprocal ratio, Default=1')
    parser.add_argument('--Kn',default=10,type=int,help='How many times a sample repeated in evaluation, Default=10')
    parser.add_argument('--burn',default=25,type=int,help='Burn percentage, Default=25')
    parser.add_argument('--lQ',default=1e2,type=float,help='lQ constant, Default=1e2')
    parser.add_argument('--ell',default=1.,type=float,help='ell constant, Default=1')
    parser.add_argument('--Vgain',default=1e3,type=float,help='Vgain constant, Default=1e3')
    ph.add_boolean_argument(parser,'useLinear',default=True,messages='Whether to N4SID Linear Dynamic as mean in prior, Default=True')

    args = parser.parse_args()

    folderName = 'NESTE_GP_SYSID-'+ datetime.datetime.now().strftime('%d-%b-%Y_%H_%M_%S')
    if 'WRKDIR' in os.environ:
        simResultPath = pathlib.Path(os.environ['WRKDIR']) / 'SimulationResult'/folderName
    elif 'USER' in os.environ and pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult').exists():
        simResultPath = pathlib.Path('/scratch/work/'+os.environ['USER']+'/SimulationResult')/folderName
    else:
        simResultPath = pathlib.Path.home() / 'Documents' / 'SimulationResult'/folderName
    if not simResultPath.exists():
        simResultPath.mkdir()

    chemReactorGP(simResultPath,randSeed=args.randSeed,resampling=args.resampling,ma_smoother=args.ma_smoother,
                    data_extension_percentage=args.extension,
                    minSS_orders=args.minSS_orders,maxSS_orders=args.maxSS_orders,useLinear=args.useLinear,
                    samples_num=args.samples_num,particles_num=args.particles_num,
                    bases_num=args.bases_num,ratio_L=args.ratio_L,Kn=args.Kn,
                    burnPercentage=args.burn,lQ=args.lQ,ell=args.lQ,Vgain=args.Vgain)

    