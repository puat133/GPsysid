function [ model ] = saem_param_nonreg( Phi, Psi, Sigma, T)
%SAEM_PARAM
% Andreas Svensson, 2016

    Q = (Phi-(Psi/Sigma)*Psi')/T;
    A = Psi/Sigma;

    model.A = A;
    model.Q = Q;

end

