function [ marg_lok_lik ] = compute_marginal_likelihood( Phi, Psi, Sigma, V, Lambda, l, N )
%COMPUTE_MARGINAL_LIKELIHOOD
% Andreas Svensson, 2016

    nb = size(V,1);
    nx = size(Phi,1);
    
    M = zeros(nx,nb);
    
    Phibar = Phi + (M/V)*M';
    Psibar = Psi +  M/V;
    Sigbar = Sigma + inv(V);
    
    Lambda_post = Lambda+Phibar-(Psibar/Sigbar)*Psibar'; l_post = l+N;
    
    gamma_lnx_prior     = log(pi)*(((nx)-1)*(nx)/4) + sum(gammaln(l/2 + (1 -(1:(nx)))/2));
    gamma_lnx_posterior = log(pi)*(((nx)-1)*(nx)/4) + sum(gammaln(l_post/2 + (1 -(1:(nx)))/2));
    
    marg_log_lik_fr_lik   = log(2*pi)*(-N/2);
    marg_log_lik_fr_post  = -log(2)*nx*l_post/2 - gamma_lnx_posterior + log(det(Lambda_post))*l_post/2 - log(2*pi)*nx*nb/2 + log(det(Sigbar))*nx/2;
    marg_log_lik_fr_prior = -log(2)*nx*l/2      - gamma_lnx_prior     + log(det(Lambda))^l/2           - log(2*pi)*nx*nb/2 - log(det(V))*nx/2;
   
    marg_lok_lik = marg_log_lik_fr_lik + marg_log_lik_fr_prior - marg_log_lik_fr_post;


end

