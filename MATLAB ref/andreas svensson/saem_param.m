function [ model ] = saem_param( Phi, Psi, Sigma, V, Lambda, l, T, m )
%SAEM_PARAM
% Andreas Svensson, 2016


    nb = size(V,1);
    nx = size(Phi,1);
    
    M = zeros(nx,nb);
    
    Phibar = Phi + (M/V)*M';
    Psibar = Psi +  M/V;
    Sigbar = Sigma + inv(V);
    Q = (Lambda+Phibar-(Psibar/Sigbar)*Psibar')/(nx+T+l+m+1);
    A = Psibar/Sigbar;

    model.A = A;
    model.Q = Q;

end

