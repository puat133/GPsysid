import numpy as np
import scipy.linalg as sla
import numba as nb
from numba.typed import Dict #numba typedDict
import numba.types as types 
import numba_settings as nbs



parameters = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

matrices = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:,:],
)
END_FOR_D_INDEX=4
N_LEVELS = 7

@nbs.njitParallel
def firstSpatialDerivativeMatrix(parameters):
    #this is bad!
    n_levels=N_LEVELS
    Delta_z = parameters['Delta_z']
    M = np.zeros((n_levels,n_levels))

    for i in nb.prange(n_levels):
        M[i,i] = 1/Delta_z[i]
        if i>0:
            M[i,i-1] = -1/Delta_z[i]

    return M


@nbs.njitParallel
def secondSpatialDerivativeMatrix(parameters):
    #this is bad!
    n_levels=7
    Delta_z = parameters['Delta_z']
    M = np.zeros((n_levels,n_levels))

    for i in nb.prange(n_levels):
        Dz2 = Delta_z[i]*Delta_z[i]
        M[i,i] = 1/Dz2
        if i>0:
            M[i,i-1] = -2/Dz2
            if i>1:
                M[i,i-2] = 1/Dz2
    return M

@nbs.njitParallel
def constructJacobian(parameters,matrices,x):
    #Load parameters
    learned_parameters = parameters['PDEconstants']
    d = learned_parameters[:END_FOR_D_INDEX]
    K = learned_parameters[END_FOR_D_INDEX:]

    #load helper matrices
    D1 = matrices['D1']#first spatial derivative
    D2 = matrices['D2']#second spatial derivative

    n_levels = N_LEVELS
    #get the current states
    s = x[:n_levels]
    a = x[n_levels:2*n_levels]
    T = x[2*n_levels:3*n_levels]
    c = x[3*n_levels:]

    Jac = np.zeros((x.shape[0],x.shape[0]))

    #compute some of the nonlinearities partial derivatives
    rS_ = rS(s,T,c,d)
    rA_ = rA(a,T,d)
    drSds_ = drSds(s,T,c,d)
    drSdT_ = drSdT(s,T,c,d)
    drSdc_ = drSdc(s,T,c,d)
    drAda_ = drAda(a,T,d)
    drAdT_ = drAdT(a,T,d)


    #Fill s
    Jac[:n_levels,:n_levels] =  -K[0]*D1 -K[1]*np.diag(drSds_*c)
    Jac[:n_levels,2*n_levels:3*n_levels] = -K[1]*np.diag(drSdT_*c)
    Jac[:n_levels,3*n_levels:] = -K[1]*np.diag(drSdc_*c+rS_)

    #Fill a
    Jac[n_levels:2*n_levels,n_levels:2*n_levels] =  -K[0]*D1 - K[1]*np.diag(drAda_*c)
    Jac[n_levels:2*n_levels,2*n_levels:3*n_levels] = -K[1]*np.diag(drAdT_*c)
    Jac[n_levels:2*n_levels,3*n_levels:] = - K[1]*np.diag(rA_)

    #Fill T
    Jac[2*n_levels:3*n_levels,n_levels:2*n_levels] = K[3]*np.diag(drAda_*c)
    Jac[2*n_levels:3*n_levels,2*n_levels:3*n_levels] = -K[2]*D1 + K[3]*np.diag(drAdT_*c)
    Jac[2*n_levels:3*n_levels,3*n_levels:] = -K[3]*np.diag(rA_)

    #Fill c
    Jac[3*n_levels:,:n_levels] = K[4]*np.diag(drSds_)
    Jac[3*n_levels:,2*n_levels:3*n_levels] = K[4]*np.diag(drSdT_)
    Jac[3*n_levels:,3*n_levels:] = K[4]*np.diag(drSdc_)

    #Fill s
    # Jac[:n_levels,:n_levels] = K[0]*D1 - K[1]*D2 -K[2]*np.diag(drSds_*c)
    # Jac[:n_levels,2*n_levels:3*n_levels] = -K[2]*np.diag(drSdT_*c)
    # Jac[:n_levels,3*n_levels:] = -K[2]*np.diag(drSdc_*c+rS_)

    # #Fill a
    # Jac[n_levels:2*n_levels,n_levels:2*n_levels] = K[3]*D1 - K[1]*D2 - K[2]*np.diag(drAda_*c)
    # Jac[n_levels:2*n_levels,2*n_levels:3*n_levels] = - K[2]*np.diag(drAdT_*c)
    # Jac[n_levels:2*n_levels,3*n_levels:] = - K[2]*np.diag(rA_)

    # #Fill T
    # Jac[2*n_levels:3*n_levels,n_levels:2*n_levels] = -K[6]*np.diag(drAda_*c)
    # Jac[2*n_levels:3*n_levels,2*n_levels:3*n_levels] = K[4]*D1 - K[5]*D2 - K[6]*np.diag(drAdT_*c)
    # Jac[2*n_levels:3*n_levels,3*n_levels:] = - K[6]*np.diag(rA_)

    # #Fill c
    # Jac[3*n_levels:,:n_levels] = - K[7]*np.diag(drSdc_)
    # Jac[3*n_levels:,2*n_levels:3*n_levels] = - K[7]*np.diag(drSdT_)
    # Jac[3*n_levels:,3*n_levels:] = - K[7]*np.diag(drSdc_)
    return Jac

@nbs.njitParallel
def P_dynamics_continuous(parameters,matrices,x,P):
    Jac = constructJacobian(parameters,matrices,x)
    dPdt = Jac@P + P@Jac.T + matrices['Q']
    return dPdt
        
            
            
                


# @torch.jit.script
@nbs.njitParallel
def catalyst_dynamics(parameters,x,u):
    #
    
    learned_parameters = parameters['PDEconstants']
    d = learned_parameters[:END_FOR_D_INDEX]
    K = learned_parameters[END_FOR_D_INDEX:]
    
    #fixed parameters
    n_levels = N_LEVELS
    Delta_z = parameters['Delta_z']

    
    
    s_in = u[0] #we assume that we are working in a mass fraction mode, the assumption is that molar fraction of sulfur or aromatics are constants times the weight fractions
    a_in = u[1]
    T_in = u[2]

    #get the current states
    s = x[:n_levels]
    a = x[n_levels:2*n_levels]
    T = x[2*n_levels:3*n_levels]
    c = x[3*n_levels:]
    
    #create new state
    dxdt = np.zeros_like(x)
    dsdt = dxdt[:n_levels] #This is only view
    dadt = dxdt[n_levels:2*n_levels]
    dTdt = dxdt[2*n_levels:3*n_levels]
    dcdt = dxdt[3*n_levels:]

    rS_ = rS(s,T,c,d)
    rA_ = rA(a,T,d)
    
    for i in nb.prange(n_levels):
        if i == 0:
            s_i_min_1 = s_in
            a_i_min_1 = a_in
            T_i_min_1 = T_in
            
            # s_i_min_2 = s_in
            # a_i_min_2 = a_in
            # T_i_min_2 = T_in
        else:
            s_i_min_1 = s[i-1]
            a_i_min_1 = a[i-1]
            T_i_min_1 = T[i-1]
            # if i==1:
            #     s_i_min_2 = s_in
            #     a_i_min_2 = a_in
            #     T_i_min_2 = T_in
            # else:
            #     s_i_min_2 = s[i-2]
            #     a_i_min_2 = a[i-2]
            #     T_i_min_2 = T[i-2]
        
        Delta_s = (s[i] - s_i_min_1)/Delta_z[i]
        Delta_a = (a[i] - a_i_min_1)/Delta_z[i]
        Delta_T = (T[i] - T_i_min_1)/Delta_z[i]
        
        
        # Delta2_s = (s[i] - 2*s_i_min_1 + s_i_min_2)/(Delta_z[i]*Delta_z[i])
        # Delta2_a = (a[i] - 2*a_i_min_1 + a_i_min_2)/(Delta_z[i]*Delta_z[i])
        # Delta2_T = (T[i] - 2*T_i_min_1 + T_i_min_2)/(Delta_z[i]*Delta_z[i])
        
        
        # rS_i = rS(s[i],T[i],c[i],d)
        # rA_i = rA(a[i],T[i],d)
        
        dsdt[i] = - K[0]*Delta_s - K[1]*rS_[i]*c[i]
        dadt[i] = - K[0]*Delta_a - K[1]*rA_[i]*c[i]
        dTdt[i] = - K[2]*Delta_T + K[3]*rA_[i]*c[i] # multiplier to K[6] should be positive
        dcdt[i] = K[4]*rS_[i]

        # dsdt[i] = K[0]*Delta2_s - K[1]*Delta_s - K[2]*rS_[i]*c[i]
        # dadt[i] = K[3]*Delta2_a - K[1]*Delta_a - K[2]*rA_[i]*c[i]
        # dTdt[i] = K[4]*Delta2_T - K[5]*Delta_T + K[6]*rA_[i]*c[i] # multiplier to K[6] should be positive
        # dcdt[i] = K[7]*rS_[i]                
    return dxdt



@nbs.njitSerial
def runggeKuttaNextStep(fun,parameters,x,u):
    dt = parameters['dt']
    k1 = fun(parameters,x,u)
    k2 = fun(parameters,x+0.5*dt*k1,u)
    k3 = fun(parameters,x+0.5*dt*k2,u)
    k4 = fun(parameters,x+dt*k3,u)
    return x+dt*(k1+2*k2+2*k3+k4)/6

@nbs.njitSerial
def runggeKuttaNextStep_Matrices(P_fun,parameters,matrices,x,P):
    dt = parameters['dt']
    k1 = P_fun(parameters,matrices,x,P)
    k2 = P_fun(parameters,matrices,x,P+0.5*dt*k1)
    k3 = P_fun(parameters,matrices,x,P+0.5*dt*k2)
    k4 = P_fun(parameters,matrices,x,P+dt*k3)
    return P+dt*(k1+2*k2+2*k3+k4)/6

@nbs.njitSerial
def catalyst_dynamics_discrete(parameters,x,u):
    return runggeKuttaNextStep(catalyst_dynamics,parameters,x,u)

@nbs.njitSerial
def P_dynamics_discrete(parameters,matrices,x,P):
    return runggeKuttaNextStep_Matrices(P_dynamics_continuous,parameters,matrices,x,P)



'''
Aromatic reaction rate
'''
@nbs.njitSerial
def rS(s,T,c,d):
    res =  -d[2]*np.exp(d[3]/T)*s*c
    return res

'''
Sulphur reaction rate
'''
@nbs.njitSerial
def rA(a,T,d):
    res =  d[0]*np.exp(d[1]/T)*a #corresponds to eq (5), but the denominator is equal to one, -Q is positive and E is positive, but (-Q)-E is still positive
    return res


#Some derivatives of rS and rA to compute Jacobian
@nbs.njitSerial
def drSds(s,T,c,d):
    res =  -d[2]*np.exp(d[3]/T)*c
    return res

@nbs.njitSerial
def drSdT(s,T,c,d):
    res =  d[2]*np.exp(d[3]/T)*d[3]/(T*T)*s*c
    return res
@nbs.njitSerial
def drSdc(s,T,c,d):
    res =  -d[2]*np.exp(d[3]/T)*s
    return res

@nbs.njitSerial
def drAda(a,T,d):
    res =  d[0]*np.exp(d[1]/T)
    return res

@nbs.njitSerial
def drAdT(a,T,d):
    res =  -d[0]*np.exp(d[1]/T)*d[1]/(T*T)*a
    return res

